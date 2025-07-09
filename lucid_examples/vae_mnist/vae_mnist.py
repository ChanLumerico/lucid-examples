import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim

import lucid.datasets as datasets
from lucid.data import DataLoader

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

        logvar = self.linear_logvar(h).clip(-5.0, 5.0)
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
        eps = 1e-6
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
        mu_T = lucid.clip(mu_T, -5.0, 5.0)

        sigma_sq_T = lucid.clip(sigma_T**2, min_value=eps)
        log_sigma_sq_T = lucid.log(sigma_sq_T)

        L_kl = -lucid.sum(1 + log_sigma_sq_T - mu_T**2 - sigma_sq_T)

        for i in range(self.num_layers - 1):
            mu_i, sigma_i = mus[i], sigmas[i]
            z_hat_i = z_hats[i]

            mu_i = lucid.clip(mu_i, -5.0, 5.0)
            sigma_sq_i = lucid.clip(sigma_i**2, min_value=eps)
            log_sigma_sq_i = lucid.log(sigma_sq_i)

            L_kl += -lucid.sum(1 + log_sigma_sq_i - (mu_i - z_hat_i) ** 2 - sigma_sq_i)

        return (L_recon + L_kl) / batch_size


input_dim = 784
hidden_dim = 200
latent_dim = 20

num_layers = 1
epochs = 30
learning_rate = 3e-4
batch_size = 32


dataset = datasets.MNIST(root="data/mnist/", train=True, download=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = HierarchicalVAE(input_dim, hidden_dim, latent_dim, num_layers, use_bce=True)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


def normalize_img(img: lucid.Tensor) -> lucid.Tensor:
    return (img.astype(lucid.Float32) / 255.0).reshape(img.shape[0], -1)


def train_model():
    losses = []
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0
        desc = f"Epoch {epoch + 1}/{epochs}"

        with tqdm(dataloader, desc=desc, unit="batch") as pbar:
            for x, _ in pbar:
                x_norm = normalize_img(x)

                optimizer.zero_grad()
                loss = model.get_loss(x_norm).eval()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                loss_sum += loss.item()
                cnt += 1
                pbar.set_postfix(avg_loss=loss_sum / cnt)

        # scheduler.step()

    plt.plot(losses, label="ELBO Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"{num_layers}-Hierarchy VAE on MNIST")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_loss.png")


def generated_image():
    model.eval()
    z = lucid.random.randn(64, latent_dim)

    with lucid.no_grad():
        x_gen = model.decoders[0](z)
    x_gen = x_gen.reshape(64, 28, 28).data

    _, axes = plt.subplots(8, 8, figsize=(8, 8))
    for ax, img in zip(axes.flatten(), x_gen):
        ax.imshow(img, cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("generated_img.png")


if __name__ == "__main__":
    print(model)
    print(f"\nTotal Parameters: {model.parameter_size:,}\n")

    train_model()
    generated_image()
