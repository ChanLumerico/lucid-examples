import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim

import lucid.datasets as datasets

from tqdm import tqdm
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
        h = F.relu(self.linear(x))
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)

        sigma = lucid.exp(0.5 * logvar)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(
        self, latent_dim: int, hidden_dim: int, out_dim: int, use_sigmoid: bool = False
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(latent_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)
        self.use_sigmoid = use_sigmoid

    def forward(self, z: lucid.Tensor) -> lucid.Tensor:
        h = F.relu(self.linear_1(z))
        h = self.linear_2(h)

        return F.sigmoid(h) if self.use_sigmoid else h


def reparameterize(mu: lucid.Tensor, sigma: lucid.Tensor) -> lucid.Tensor:
    eps = lucid.random.randn(*sigma.shape)
    return mu + eps * sigma


class HierarchicalVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        use_bce: bool = False,
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "Number of layers must be >= 1"
        self.num_layers = num_layers
        self.use_bce = use_bce

        dims = [input_dim] + [latent_dim] * (num_layers - 1)
        self.encoders = nn.ModuleList(
            [Encoder(dims[i], hidden_dim, latent_dim) for i in range(num_layers)]
        )
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.decoders.append(
                    Decoder(latent_dim, hidden_dim, input_dim, use_sigmoid=True)
                )
            else:
                self.decoders.append(Decoder(latent_dim, hidden_dim, latent_dim))

    def get_loss(self, x: lucid.Tensor) -> lucid.Tensor:
        batch_size = x.shape[0]

        mus, sigmas, zs = [], [], []
        h = x
        for enc in self.encoders:
            mu, sigma = enc(h)
            z = reparameterize(mu, sigma)

            mus.append(mu)
            sigmas.append(sigma)
            zs.append(z)
            h = z

        x_hat = self.decoders[0](zs[0])
        z_hats = [None] * (self.num_layers - 1)
        for level in range(self.num_layers, 1, -1):
            idx = level - 1
            z_hats[idx - 1] = self.decoders[idx](zs[idx])

        if self.use_bce:
            L_recon = F.binary_cross_entropy(x_hat, x, reduction="sum")
        else:
            L_recon = F.mse_loss(x_hat, x, reduction="sum")

        mu_T, sigma_T = mus[-1], sigmas[-1]
        L_kl = -lucid.sum(1 + lucid.log(sigma_T**2) - mu_T**2 - sigma_T**2)

        for i in range(self.num_layers - 1):
            mu_i, sigma_i = mus[i], sigmas[i]
            z_hat_i = z_hats[i]
            L_kl += -lucid.sum(1 + sigma_i**2 - (mu_i - z_hat_i) ** 2 - sigma_i**2)

        return (L_recon + L_kl) / batch_size


input_dim = 784
hidden_dim = 100
latent_dim = 20

num_layers = 2
epochs = 2
learning_rate = 1e-3
batch_size = 64


# TODO: Continue from loading dataset
