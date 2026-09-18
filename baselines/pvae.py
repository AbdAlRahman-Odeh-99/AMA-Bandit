"""Partial VAE component used by online EDDI."""

import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal


class PVAE(nn.Module):
    """Variational imputer updated causally by the online EDDI adapter."""

    def __init__(
        self,
        encoder,
        decoder,
        mask_layer,
        num_samples=128,
        decoder_distribution="gaussian",
        deterministic_kl=True,
    ):
        super().__init__()
        if decoder_distribution not in ("gaussian", "bernoulli"):
            raise ValueError("decoder_distribution must be gaussian or bernoulli")
        self.encoder = encoder
        self.decoder = decoder
        self.mask_layer = mask_layer
        self.num_samples = num_samples
        self.deterministic_kl = deterministic_kl
        self.decoder_distribution = decoder_distribution

    def forward(self, x, mask):
        x_masked = self.mask_layer(x, mask)
        latent = self.encoder(x_masked)
        dims = latent.shape[1] // 2
        mean = latent[:, :dims]
        std = torch.exp(latent[:, dims:])
        eps = torch.randn(
            mean.shape[0], self.num_samples, mean.shape[1], device=mean.device
        )
        z = mean.unsqueeze(1) + eps * std.unsqueeze(1)
        return latent, z, self.decoder(z)

    def loss(self, x, mask):
        latent, z, reconstruction = self.forward(x, mask)
        latent_dims = latent.shape[1] // 2
        latent_mean = latent[:, :latent_dims]
        latent_std = torch.exp(latent[:, latent_dims:])
        if self.deterministic_kl:
            kl = torch.distributions.kl_divergence(
                Normal(latent_mean, latent_std), Normal(0.0, 1.0)
            ).sum(1).unsqueeze(1)
        else:
            prior = Normal(0.0, 1.0)
            posterior = Normal(latent_mean, latent_std)
            log_p = prior.log_prob(z)
            log_q = posterior.log_prob(z.permute(1, 0, 2)).permute(1, 0, 2)
            kl = (log_q - log_p).sum(dim=2)

        if self.decoder_distribution == "gaussian":
            distribution = Normal(reconstruction, torch.ones_like(reconstruction))
        else:
            distribution = Bernoulli(reconstruction.sigmoid())

        log_prob = distribution.log_prob(x.unsqueeze(1))
        mask_values = (
            mask @ self.mask_layer.group_matrix
            if hasattr(self.mask_layer, "group_matrix")
            else mask
        )
        log_prob = (log_prob * mask_values.unsqueeze(1)).sum(dim=2)
        return kl - log_prob

    def impute(self, x, mask):
        """Impute missing values from a partial input."""
        _, _, reconstruction = self.forward(x, mask)
        return self.output_sample(reconstruction)

    def generate(self, num_samples):
        """Generate samples from the decoder's latent prior."""
        latent_dim = list(self.decoder.parameters())[0].shape[1]
        device = next(self.decoder.parameters()).device
        z = torch.randn(num_samples, latent_dim, device=device)
        return self.output_sample(self.decoder(z))

    def output_sample(self, parameters):
        """Convert decoder parameters to an imputed/generated value."""
        if self.decoder_distribution == "gaussian":
            return parameters
        return parameters.sigmoid()
