import torch
import torch.nn as nn
import torch.optim as optim


def generate_synthetic_data(num_samples=1000):
    """Generate synthetic data x and true latent variables z."""
    z = torch.randn(num_samples, 2)  # True latent variable (Gaussian)
    x = z + 0.1 * torch.randn_like(z)  # Observed data with noise
    return x, z


class VariationalPosterior(nn.Module):
    """Approximate posterior q_phi(z|x)."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # Mean and log-variance
        )

    def forward(self, x):
        params = self.encoder(x)
        mean, log_var = params[:, :2], params[:, 2:]
        std = torch.exp(0.5 * log_var)
        return mean, std

    def sample(self, x):
        mean, std = self.forward(x)
        eps = torch.randn_like(mean)
        return mean + eps * std


class Discriminator(nn.Module):
    """Discriminator T_psi(x, z)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, z):
        return self.net(torch.cat([x, z], dim=-1))


class AdaptiveContrast(nn.Module):
    """Adaptive Contrast Module."""

    def __init__(self):
        super().__init__()

    def forward(self, z_fake, z_real):
        """
        Computes adaptive contrast loss.

        z_fake: Samples from the approximate posterior.
        z_real: Samples from the true posterior.

        Returns:
            contrast_loss: Adaptive contrast loss.
        """
        z_fake_mean = z_fake.mean(dim=0)
        z_real_mean = z_real.mean(dim=0)
        contrast_loss = torch.norm(z_fake_mean - z_real_mean, p=2)
        return contrast_loss


# Generate synthetic data
x_data, z_true = generate_synthetic_data()

# Models
q_phi = VariationalPosterior()
T_psi = Discriminator()
adaptive_contrast = AdaptiveContrast()

# Optimizers
optimizer_q = optim.Adam(q_phi.parameters(), lr=1e-3)
optimizer_t = optim.Adam(T_psi.parameters(), lr=1e-3)

# Training loop
epochs = 500
batch_size = 64
for epoch in range(epochs):
    for i in range(0, x_data.size(0), batch_size):
        x_batch = x_data[i : i + batch_size]

        # --- Train Discriminator ---
        z_fake = q_phi.sample(x_batch)  # Samples from approximate posterior
        z_real = z_true[i : i + batch_size]  # Samples from true posterior

        logits_real = T_psi(x_batch, z_real)
        logits_fake = T_psi(x_batch, z_fake)

        loss_t = -torch.mean(
            torch.log(logits_real + 1e-8) + torch.log(1 - logits_fake + 1e-8)
        )

        optimizer_t.zero_grad()
        loss_t.backward()
        optimizer_t.step()

        # --- Train Generator ---
        z_fake = q_phi.sample(x_batch)
        logits_fake = T_psi(x_batch, z_fake)

        loss_q = -torch.mean(torch.log(logits_fake + 1e-8))

        # Add Adaptive Contrast Loss
        contrast_loss = adaptive_contrast(z_fake, z_real)
        total_loss_q = loss_q + 0.1 * contrast_loss  # Weighted combination

        optimizer_q.zero_grad()
        total_loss_q.backward()
        optimizer_q.step()

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch}: Discriminator Loss = {loss_t.item():.4f}, Generator Loss = {loss_q.item():.4f}, Contrast Loss = {contrast_loss.item():.4f}"
        )

# After training, q_phi approximates the true posterior.
