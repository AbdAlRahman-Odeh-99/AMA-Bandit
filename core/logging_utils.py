# -*- coding: utf-8 -*-
"""
core/logging_utils.py

One shared home for everything about a RUN that is not the run's numbers:
identity, provenance, console/file logging, fine-grained timing, progress
heartbeats, crash-resilient row checkpointing, per-cell failure isolation,
and the optional per-round trace sidecar.

=== Why this exists ===
Before this module every runner did its observability with bare `print()`
to stdout, and recorded exactly three timings per row -- train, inference,
seed. Under PBS that meant:

  * stdout is captured into one .o file, buffered, and interleaved with
    every other message, so a 10-hour job gave no usable progress signal;
  * nothing tied a results_*.xlsx back to the code that produced it (no
    commit, no argv, no library versions, no job id);
  * a walltime kill at seed 8/10 threw away all eight completed seeds,
    because rows only reached disk in the final save_results_to_excel;
  * one bad cell (a degenerate LP, a missing class in a split) killed the
    whole sweep;
  * "why is full-space acquisition slow?" was unanswerable -- train_time_sec lumps the
    acquisition policy, the arm-reward replay and the dual update together.

Every piece below exists to fix exactly one of those.

=== The five things a runner uses ===

1. setup_run(...) -> RunContext
   Called ONCE from an entry point's __main__, before any work. Assigns the
   run_id, creates logs/ and results/, installs the logging handlers, and
   writes the provenance manifest. Everything else finds it via get_run().

2. get_logger(__name__) / logging
   Replaces print. INFO goes to console AND file; DEBUG to the file only.

3. tick("t_acquisition") / cell_timers()
   Fine-grained timing. `cell_timers()` opens a fresh bucket for one sweep
   cell and yields the dict it will fill; any `tick(name)` executed inside
   it -- including deep inside core.lp_colgen, which knows nothing about
   the runner -- accumulates into that bucket. That indirection is the
   whole point: instrumenting the pricing loop costs one `with` statement
   and NO signature change anywhere between it and the runner.

4. Progress(total).step()
   Heartbeat with ETA, so a running job says where it is.

5. guard(cell) + run.emit_row(row)
   Per-cell failure isolation and append-as-you-go checkpointing. Rows land
   in results/{run_id}.rows.jsonl the moment they are produced.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

__all__ = [
    "RunContext", "setup_run", "get_run", "get_logger",
    "tick", "timed", "cell_timers", "add_time", "bump", "current_timers",
    "set_timing_enabled", "timing_enabled",
    "Progress", "guard", "CellOutcome",
    "TIMER_KEYS", "COUNTER_KEYS", "TIMING_COLUMNS", "excel_timing_columns",
    "timing_row", "read_rows_jsonl", "read_manifest", "fmt_hms",
]


# ─────────────────────────────────────────────────────────────────────────
# The timing vocabulary.
#
# ONE list, defined here, consumed by the runners (as row keys), by
# run_proposed_methods (as UNIFIED_COLUMNS entries) and by the instrumented
# core modules (as tick names). Adding a bucket means adding it here and
# adding one `with tick(...)` -- nothing else in the pipeline needs editing,
# and nothing can drift out of sync because there is no second copy.
#
# The three ORIGINAL timings -- train_time_sec / inference_time_sec /
# seed_time_sec -- are NOT in this list and are NOT touched. They remain
# exactly what they always were, computed the same way, in the same columns.
# These are a strictly finer decomposition living alongside them.
# ─────────────────────────────────────────────────────────────────────────
TIMER_KEYS = (
    "t_data_load",        # run-level: load_dataset_as_numpy (once per run)
    "t_init_centers",     # Stage 1 / centre initialisation
    "t_acquisition",      # per-round subset choice: chain LP, oracle LP,
                          # argmax, or Hedge. Ticked inside
                          # core.acquisition_policies, so it covers BOTH
                          # methods and EVERY acquisition mode for free.
    "t_reward_update",    # per-round empirical arm-reward update (the
                          # containment replay is the expensive one)
    "t_dual_update",      # per-round OMD dual step
    "t_predict",          # per-round prediction + scoring
    "t_inference_solve",  # inference-phase colgen solve, total
    "t_pricing",          #   ... of which: branch-and-bound pricing
    "t_master_lp",        #   ... of which: restricted-master linprog
    "t_inference_sample", # inference-phase physical sampling + scoring loop
)

COUNTER_KEYS = (
    "n_colgen_iters",     # column-generation iterations (master solves)
    "n_lp_solves",        # per-round LP solves during training
    "n_guard_failures",   # cells that raised and were isolated
)

# Row keys, in the order they should appear in a workbook.
TIMING_COLUMNS = TIMER_KEYS + COUNTER_KEYS


def _excel_name(key: str) -> str:
    """t_master_lp -> 'T Master LP (s)';  n_colgen_iters -> 'N Colgen Iters'."""
    words = key.split("_")
    pretty = " ".join(w.upper() if w in ("lp", "ucb") else w.capitalize() for w in words)
    return f"{pretty} (s)" if key.startswith("t_") else pretty


#: {row_key: "Excel Column Name"} -- consumed by run_proposed_methods to
#: extend UNIFIED_COLUMNS without hardcoding a second copy of the names.
EXCEL_TIMING_COLUMNS = {k: _excel_name(k) for k in TIMING_COLUMNS}


def excel_timing_columns():
    """Ordered list of the Excel column names for the timing block."""
    return [EXCEL_TIMING_COLUMNS[k] for k in TIMING_COLUMNS]


def timing_row(buckets, keys=TIMING_COLUMNS):
    """Materialise a timer bucket into a flat row fragment.

    Missing buckets become NaN rather than 0.0 -- deliberately. 0.0 asserts
    "this ran and took no time"; NaN says "this run did not measure it",
    which is the truth for a mode that never enters that code path (e.g.
    t_pricing on a --skip-inference run). Averaging a NaN column with
    np.nanmean then reports the modes that DID measure it, instead of
    silently halving their mean against zeros.
    """
    buckets = buckets or {}
    return {k: float(buckets[k]) if k in buckets else float("nan") for k in keys}


# ─────────────────────────────────────────────────────────────────────────
# Timing: a thread-local stack of accumulator dicts.
#
# The stack is what lets `tick` be callable from anywhere without plumbing.
# core/lp_colgen.py's pricing routine ticks "t_pricing" with no idea whether
# a runner is listening; if one is (cell_timers is open), the time lands in
# that cell's bucket, and if none is, it lands in the run-level bucket, and
# if timing is off entirely the context manager is a no-op that costs one
# boolean test.
# ─────────────────────────────────────────────────────────────────────────
_local = threading.local()
_TIMING_ENABLED = True


def set_timing_enabled(flag: bool) -> None:
    """Master switch for the fine-grained timers (--no-fine-timers).

    Off, `tick` degrades to a bare generator with one boolean test -- the
    per-round buckets (t_predict, t_reward_update, t_dual_update) fire tens
    of thousands of times per cell, so the escape hatch exists for anyone
    who would rather not pay two perf_counter calls for each.
    """
    global _TIMING_ENABLED
    _TIMING_ENABLED = bool(flag)


def timing_enabled() -> bool:
    return _TIMING_ENABLED


def _stack():
    st = getattr(_local, "stack", None)
    if st is None:
        st = _local.stack = [{}]      # index 0 is the run-level bucket
    return st


def current_timers():
    """The bucket `tick` is currently writing into."""
    return _stack()[-1]


def add_time(name: str, seconds: float) -> None:
    b = current_timers()
    b[name] = b.get(name, 0.0) + float(seconds)


def bump(name: str, k=1) -> None:
    """Increment a COUNTER bucket (n_colgen_iters and friends)."""
    if not _TIMING_ENABLED:
        return
    b = current_timers()
    b[name] = b.get(name, 0) + k


@contextmanager
def tick(name: str):
    """Accumulate this block's wall time into bucket `name`.

    Re-entrant and exception-safe: the elapsed time is recorded in a finally
    block, so a cell that dies mid-solve still reports how long it spent
    before dying -- which is usually the interesting part.
    """
    if not _TIMING_ENABLED:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        add_time(name, time.perf_counter() - t0)


def timed(name: str):
    """Decorator form of `tick`, for instrumenting a whole function.

    Preferred over wrapping a function body in `with tick(...)` when the
    function already exists and works: a decorator adds one line and
    re-indents nothing, so the diff cannot silently change control flow
    inside a numerical routine. Used on the acquisition policies in
    core.acquisition_policies, which is how t_acquisition gets measured for
    BOTH methods and EVERY acquisition mode without either method module
    being touched.

    NOTE on nesting: `tick` accumulates per name, so a decorated function
    calling another function decorated with the SAME name would count the
    inner span twice. The convention that avoids it: only the leaf policy
    routines in core carry the t_acquisition decorator, and no caller ever
    opens a t_acquisition tick of its own.
    """
    def deco(fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            if not _TIMING_ENABLED:
                return fn(*a, **kw)
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                add_time(name, time.perf_counter() - t0)
        return wrapper
    return deco


@contextmanager
def cell_timers(inherit=()):
    """Open a fresh bucket for ONE sweep cell and yield it.

    Anything ticked inside -- at any call depth -- accumulates here instead
    of in the run-level bucket. `inherit` names run-level buckets to copy in
    (t_data_load is the one that matters: it is measured once per run but
    belongs on every row, so each row can be read as a self-contained
    account of where its time went).
    """
    run_bucket = _stack()[0]
    bucket = {k: run_bucket[k] for k in inherit if k in run_bucket}
    _stack().append(bucket)
    try:
        yield bucket
    finally:
        _stack().pop()


# ─────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────
def _sh(cmd, cwd=None):
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                             timeout=10, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def _git_info(cwd=None):
    """Commit, branch, dirty flag, and the short stat of uncommitted work.

    `dirty` is the field that matters and the one a manual note always gets
    wrong: an .xlsx produced from a working tree with uncommitted edits is
    NOT reproducible from its commit hash, and this is the only place that
    records the difference.
    """
    cwd = str(cwd or Path.cwd())
    commit = _sh(["git", "rev-parse", "HEAD"], cwd)
    if commit is None:
        return {"git_available": False}
    status = _sh(["git", "status", "--porcelain"], cwd) or ""
    return {
        "git_available": True,
        "git_commit": commit,
        "git_commit_short": commit[:12],
        "git_branch": _sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd),
        "git_dirty": bool(status.strip()),
        "git_dirty_files": [ln[3:] for ln in status.splitlines()][:50],
        "git_describe": _sh(["git", "describe", "--always", "--dirty", "--tags"], cwd),
    }


def _package_versions():
    names = ("numpy", "scipy", "pandas", "openpyxl", "scikit-learn", "numba")
    try:
        from importlib.metadata import version
    except Exception:                                    # pragma: no cover
        return {}
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except Exception:
            out[n] = None
    return out


def _scheduler_env():
    """PBS/SLURM identifiers, so a workbook can be traced back to its job.

    Kept generic on purpose -- the same manifest should mean something if
    these runs ever move off Gadi.
    """
    keys = ("PBS_JOBID", "PBS_JOBNAME", "PBS_O_QUEUE", "PBS_NCPUS", "PBS_VMEM",
            "PBS_O_WORKDIR", "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_CPUS_ON_NODE")
    return {k: os.environ.get(k) for k in keys if os.environ.get(k)}


def _job_tag():
    """Short, filesystem-safe identifier for THIS execution."""
    jid = os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID")
    if jid:
        return "job" + jid.split(".")[0]
    return f"{socket.gethostname().split('.')[0]}-{os.getpid()}"


# ─────────────────────────────────────────────────────────────────────────
# JSON encoding for numpy
# ─────────────────────────────────────────────────────────────────────────
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _dump(obj):
    return json.dumps(obj, default=_json_default)


# ─────────────────────────────────────────────────────────────────────────
# RunContext
# ─────────────────────────────────────────────────────────────────────────
_ACTIVE = None


class RunContext:
    """Identity + open file handles + manifest for one execution.

    One per process, reachable from anywhere via get_run(). Nothing here is
    required: every helper in this module checks for None and degrades to a
    no-op, so a runner imported into a notebook works unchanged.
    """

    def __init__(self, run_id, run_dir, results_dir, log_dir,
                 manifest, trace_rounds=False, file_logging=True,
                 json_sidecars=True):
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.results_dir = Path(results_dir)
        self.log_dir = Path(log_dir)
        self.manifest = dict(manifest)
        self.trace_rounds = bool(trace_rounds)
        self.file_logging = bool(file_logging)
        self.json_sidecars = bool(json_sidecars)

        self.rows_path = self.results_dir / f"{run_id}.rows.jsonl"
        self.trace_path = self.results_dir / f"{run_id}.trace.jsonl"
        self.manifest_path = self.results_dir / f"{run_id}.manifest.json"
        self.log_path = self.log_dir / f"{run_id}.log"

        self.t_start = time.time()
        self.n_rows = 0
        self.failures = []
        self._lock = threading.Lock()
        self._rows_fh = None
        self._trace_fh = None
        self._finalized = False

    # -- checkpointing ---------------------------------------------------
    def emit_row(self, row):
        """Append one finished sweep-cell row to the JSONL checkpoint.

        Flushed on every write. That is the entire point: a walltime kill,
        an OOM, or a node failure leaves every row produced up to that
        instant on disk and readable, instead of leaving nothing at all
        because the workbook is only written after the last seed.
        """
        with self._lock:
            if not self.json_sidecars:
                self.n_rows += 1
                return
            if self._rows_fh is None:
                self._rows_fh = open(self.rows_path, "a", buffering=1)
            self._rows_fh.write(_dump(row) + "\n")
            self._rows_fh.flush()
            self.n_rows += 1

    def emit_trace(self, record):
        """Append one per-ROUND trace record (only when --trace-rounds).

        This is where the training trace belongs. `Selected Subsets` was
        carrying thousands of subsets inside a single Excel cell -- a
        column that no spreadsheet can display, no groupby can aggregate,
        and excel_utils has a width hack specifically to survive. Here each
        round is its own record with the state that makes it interpretable
        (lambda, cost, remaining budget, reward), so a lambda trajectory is
        a one-line plot instead of a parsing exercise.
        """
        if not self.trace_rounds:
            return
        with self._lock:
            if self._trace_fh is None:
                self._trace_fh = open(self.trace_path, "a", buffering=1)
            self._trace_fh.write(_dump(record) + "\n")

    def note_failure(self, cell, message, tb=None):
        self.failures.append({"cell": cell, "error": message, "traceback": tb})
        bump("n_guard_failures")

    # -- manifest --------------------------------------------------------
    def write_manifest(self, status="running", extra=None):
        m = dict(self.manifest)
        m.update({
            "run_id": self.run_id,
            "status": status,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                        time.localtime(self.t_start)),
            "elapsed_sec": round(time.time() - self.t_start, 3),
            "n_rows": self.n_rows,
            "n_failures": len(self.failures),
            "failures": self.failures[:50],
            "rows_jsonl": str(self.rows_path) if self.json_sidecars else None,
            "trace_jsonl": str(self.trace_path) if self.trace_rounds else None,
            "log_file": str(self.log_path) if self.file_logging else None,
        })
        if status != "running":
            m["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if extra:
            m.update(extra)
        if self.json_sidecars:
            self.manifest_path.write_text(
                json.dumps(m, indent=2, default=_json_default))
        return m

    def finalize(self, status="ok", extra=None):
        if self._finalized:
            return self.manifest
        m = self.write_manifest(status=status, extra=extra)
        with self._lock:
            for fh in (self._rows_fh, self._trace_fh):
                if fh is not None:
                    fh.flush()
                    fh.close()
            self._rows_fh = self._trace_fh = None
        self._finalized = True
        self.manifest = m
        return m

    # -- Excel "Run Info" sheet -----------------------------------------
    def info_rows(self):
        """Flatten the manifest into (key, value) pairs for the workbook.

        The sidecar JSON is the authoritative copy; this is the version you
        can read without leaving Excel, which is where these files are
        actually opened.
        """
        m = self.write_manifest(status=self.manifest.get("status", "running"))
        rows = []

        def walk(prefix, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(obj, (list, tuple)):
                if not obj:
                    rows.append((prefix, ""))
                elif all(not isinstance(v, (dict, list, tuple)) for v in obj):
                    rows.append((prefix, ", ".join(str(v) for v in obj)))
                else:
                    for i, v in enumerate(obj):
                        walk(f"{prefix}[{i}]", v)
            else:
                rows.append((prefix, "" if obj is None else obj))

        walk("", m)
        return rows


def get_run():
    """The active RunContext, or None when no entry point set one up."""
    return _ACTIVE


def get_logger(name="afa"):
    return logging.getLogger(name)


def _install_handlers(log_path, console_level, file_level, file_logging=True):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)

    # Console: terse -- this is what lands in the PBS .o file and what a
    # human tails. No timestamps on the console would make a stalled job
    # unreadable, so keep a short clock but drop the logger name.
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(getattr(logging, str(console_level).upper(), logging.INFO))
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s %(message)s",
                                      datefmt="%H:%M:%S"))
    root.addHandler(ch)

    # Optional separate Python file log. On batch systems stdout/stderr is
    # already captured by the scheduler, so CLI entry points can disable this
    # duplicate while retaining console output and every structured sidecar.
    fh = None
    if file_logging:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(getattr(logging, str(file_level).upper(), logging.DEBUG))
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s:%(lineno)d %(message)s"))
        root.addHandler(fh)

    # scipy/matplotlib/numba DEBUG is noise at this level of detail.
    for noisy in ("matplotlib", "numba", "PIL", "urllib3", "sklearn"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return fh


def setup_run(entry_point, args=None, argv=None, name_hint=None, run_id=None,
              results_dir="results", log_dir="logs",
              console_level="INFO", file_level="DEBUG",
              trace_rounds=False, timing=True, extra=None,
              file_logging=True, json_sidecars=True):
    """Create the RunContext, install logging, write the manifest.

    Parameters
    ----------
    entry_point : str
        Which script is running (e.g. "run_proposed_methods").
    args : argparse.Namespace or dict or None
        The RESOLVED arguments -- what the run actually used, after
        defaults and after 'all' -> None style normalisation. Recorded
        verbatim in the manifest. This, not argv, is what answers "what
        did this run do"; argv is kept alongside for the copy-pasteable
        reproduction command.
    name_hint : str
        Usually the output .xlsx stem. The run_id is built from it, so the
        log, the manifest, the JSONL checkpoint and the workbook all share
        one name and sort together in a directory listing.

    Returns the RunContext, which is also installed as the process-wide
    active run.
    """
    global _ACTIVE

    results_dir = Path(results_dir)
    log_dir = Path(log_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = (name_hint or entry_point).replace(".xlsx", "")
    base = "".join(c if (c.isalnum() or c in "._-") else "_" for c in base)
    run_id = run_id or f"{base}__{stamp}__{_job_tag()}"

    set_timing_enabled(timing)

    if args is None:
        args_dict = {}
    elif isinstance(args, dict):
        args_dict = dict(args)
    else:
        args_dict = {k: v for k, v in vars(args).items()}

    manifest = {
        "entry_point": entry_point,
        "argv": list(argv if argv is not None else sys.argv),
        "reproduce": " ".join([sys.executable] + list(argv if argv is not None else sys.argv)),
        "args": args_dict,
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": _package_versions(),
        "scheduler": _scheduler_env(),
        "timing_enabled": bool(timing),
        "trace_rounds": bool(trace_rounds),
        "file_logging": bool(file_logging),
        "json_sidecars": bool(json_sidecars),
    }
    manifest.update(_git_info())
    if extra:
        manifest.update(extra)

    ctx = RunContext(run_id, results_dir, results_dir, log_dir, manifest,
                     trace_rounds=trace_rounds, file_logging=file_logging,
                     json_sidecars=json_sidecars)
    _install_handlers(ctx.log_path, console_level, file_level,
                      file_logging=file_logging)
    ctx.write_manifest(status="running")
    _ACTIVE = ctx

    log = get_logger("afa.run")
    log.info("run_id  %s", run_id)
    if file_logging:
        log.info("log     %s", ctx.log_path)
    else:
        log.info("file log disabled")
    if json_sidecars:
        log.info("rows    %s", ctx.rows_path)
    else:
        log.info("JSON sidecars disabled")
    git = manifest.get("git_commit_short")
    if git:
        log.info("commit  %s%s (%s)", git,
                 "  *** DIRTY WORKING TREE ***" if manifest.get("git_dirty") else "",
                 manifest.get("git_branch"))
        if manifest.get("git_dirty"):
            log.warning("uncommitted changes present -- this run is NOT reproducible "
                        "from the commit alone; see the manifest's git_dirty_files")
    else:
        log.warning("no git metadata (not a repository?) -- provenance is limited "
                    "to argv and package versions")
    if manifest["scheduler"]:
        log.info("job     %s", manifest["scheduler"])
    return ctx


# ─────────────────────────────────────────────────────────────────────────
# Progress heartbeat
# ─────────────────────────────────────────────────────────────────────────
def fmt_hms(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "--:--:--"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class Progress:
    """Cell-level heartbeat with an ETA.

    A sweep is len(seeds) x len(budget_fractions) x len(init_fractions)
    cells and the per-seed print only fires once every few hundred of them.
    This reports done/total, elapsed, mean cell time and projected finish,
    so an over-ambitious walltime is visible in the first minutes rather
    than at the kill.
    """

    def __init__(self, total, label="cells", logger=None, log_every=None,
                 min_interval=0.0):
        self.total = int(total)
        self.label = label
        self.log = logger or get_logger("afa.progress")
        self.done = 0
        self.t0 = time.time()
        self._last = 0.0
        self.min_interval = float(min_interval)
        self.log_every = int(log_every) if log_every else max(1, self.total // 50)

    def step(self, note="", force=False):
        self.done += 1
        now = time.time()
        due = force or self.done == self.total or (self.done % self.log_every == 0)
        if not due or (now - self._last) < self.min_interval:
            return
        self._last = now
        elapsed = now - self.t0
        per = elapsed / max(1, self.done)
        eta = per * (self.total - self.done)
        pct = 100.0 * self.done / max(1, self.total)
        self.log.info(
            "progress %d/%d %s (%.1f%%) | elapsed %s | %.2fs/%s | eta %s | finish ~%s%s",
            self.done, self.total, self.label, pct, fmt_hms(elapsed), per,
            self.label.rstrip("s"), fmt_hms(eta),
            time.strftime("%H:%M:%S", time.localtime(now + eta)),
            f" | {note}" if note else "")


# ─────────────────────────────────────────────────────────────────────────
# Per-cell failure isolation
# ─────────────────────────────────────────────────────────────────────────
class CellOutcome:
    """Result of one guarded cell: `ok`, plus the message if it failed."""

    __slots__ = ("ok", "error", "traceback", "cell")

    def __init__(self, cell):
        self.ok = True
        self.error = ""
        self.traceback = ""
        self.cell = cell

    @property
    def status(self):
        return "ok" if self.ok else "error"


@contextmanager
def guard(cell, logger=None, reraise=False):
    """Run one sweep cell; on exception log it, record it, and continue.

    Rationale: a sweep is hundreds of independent cells, and the failures
    that actually occur here are local -- a degenerate LP at
    budget_fraction 0, a split whose test fold is missing a class, an arm
    table that blows the view cap at one --max-modalities. Losing the other
    299 cells to one of those is pure waste, and worse, it hides the
    pattern: with the sweep completed you can SEE that every failure is at
    one budget fraction.

    KeyboardInterrupt and SystemExit are deliberately NOT caught -- Ctrl-C
    and a PBS walltime signal must still stop the job immediately.
    """
    log = logger or get_logger("afa.cell")
    out = CellOutcome(cell)
    try:
        yield out
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:                            # noqa: BLE001
        out.ok = False
        out.error = f"{type(exc).__name__}: {exc}"
        out.traceback = traceback.format_exc()
        log.error("CELL FAILED %s -- %s", cell, out.error)
        log.debug("traceback for %s:\n%s", cell, out.traceback)
        run = get_run()
        if run is not None:
            run.note_failure(cell, out.error, out.traceback)
        if reraise:
            raise


# ─────────────────────────────────────────────────────────────────────────
# Reading back a checkpoint
# ─────────────────────────────────────────────────────────────────────────
def read_rows_jsonl(path):
    """Load a .rows.jsonl checkpoint back into a list of dicts.

    Tolerates a truncated final line -- the file is flushed per row, but a
    process killed mid-write can still leave a partial one, and refusing to
    read 8 completed seeds because of half a trailing line would defeat the
    purpose.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                get_logger("afa.rebuild").warning(
                    "%s: ignoring unparseable line %d (truncated write?)", path, i)
    return rows


def read_manifest(path):
    return json.loads(Path(path).read_text())
