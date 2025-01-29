import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

# |%%--%%| <Cvvp9QOJww|3cKq5Gg53b>


# Hyperparameters
input_dim = 784  # For MNIST
hidden_dim = 500
latent_dim = 10
n_clusters = 10  # Number of clusters (e.g., 10 for MNIST digits)
batch_size = 100
epochs = 50
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


# Load MNIST dataset
def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = datasets.MNIST(
        root="~/Documents/data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="~/Documents/data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# |%%--%%| <3cKq5Gg53b|nY0ooJNnyg>


# VaDE Model
class VADE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_clusters):
        super(VADE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc4 = nn.Linear(latent_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

        # GMM Parameters
        self.pi = nn.Parameter(
            torch.ones(n_clusters) / n_clusters
        )  # Mixing coefficients
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim))  # Cluster means
        self.var_c = nn.Parameter(
            torch.ones(n_clusters, latent_dim)
        )  # Cluster variances

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        return torch.sigmoid(self.fc_out(h))

    def responsibility(self, z):
        # Compute responsibilities using current GMM parameters
        z = z.unsqueeze(1).expand(-1, self.n_clusters, -1)
        mu_c = self.mu_c.unsqueeze(0).expand(z.size(0), -1, -1)
        var_c = self.var_c.unsqueeze(0).expand(z.size(0), -1, -1)

        # Compute log Gaussian likelihood
        log_gaussian = -0.5 * torch.sum(
            torch.log(2 * math.pi * var_c) + (z - mu_c) ** 2 / var_c, dim=2
        )

        # Exponentiate to get Gaussian likelihood
        gaussian = torch.exp(log_gaussian)

        # Compute responsibilities
        pi = F.softmax(self.pi, dim=0)
        res = pi * gaussian
        res = res / (torch.sum(res, dim=1, keepdim=True) + 1e-10)
        return res

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_dim))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z


# |%%--%%| <nY0ooJNnyg|QKnVZyzjOI>


# Loss Function
def loss_function(recon_x, x, mu, logvar, z, pi, mu_c, var_c):
    # Reconstruction loss
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction="mean")

    # KL divergence between q(z|x) and p(z|c)
    z = z.unsqueeze(1).expand(-1, n_clusters, -1)
    mu_c = mu_c.unsqueeze(0).expand(z.size(0), -1, -1)
    var_c = var_c.unsqueeze(0).expand(z.size(0), -1, -1)

    log_p_c_z = -0.5 * torch.sum(
        torch.log(2 * math.pi * var_c) + (z - mu_c) ** 2 / var_c, dim=2
    )
    log_pi = torch.log(pi + 1e-10)
    log_p_c_z = log_p_c_z + log_pi.unsqueeze(0)
    log_q_z_x = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    # Compute KL divergence
    KL = torch.sum(log_q_z_x.unsqueeze(1) - log_p_c_z, dim=1).mean()

    # Total loss
    loss = BCE + KL
    return loss


# |%%--%%| <QKnVZyzjOI|yySsIy3DD8>


# Initialize GMM Parameters
def initialize_gmm_parameters(model, train_loader):
    model.eval()
    mu_list = []
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, input_dim))
            mu_list.append(mu.cpu())
    mu_all = torch.cat(mu_list, dim=0)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(mu_all.numpy())
    model.mu_c.data = torch.tensor(kmeans.cluster_centers_).float().to(device)


# Training Loop
def train(model, train_loader, optimizer):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar, z = model(data)
            pi = F.softmax(model.pi, dim=0)
            loss = loss_function(
                recon_x, data, mu, logvar, z, pi, model.mu_c, model.var_c
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader.dataset)}")


# Evaluation
def evaluate(model, test_loader):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, input_dim))
            res = model.responsibility(mu)
            preds.append(torch.argmax(res, dim=1).cpu().numpy())
            targets.append(target.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    nmi = normalized_mutual_info_score(targets, preds)
    ari = adjusted_rand_score(targets, preds)
    print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}")


# |%%--%%| <yySsIy3DD8|7jsOQNWUQT>


# Main Function
train_loader, test_loader = load_data()
model = VADE(input_dim, hidden_dim, latent_dim, n_clusters).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Initialize GMM parameters
initialize_gmm_parameters(model, train_loader)

# Train the model
train(model, train_loader, optimizer)

# Evaluate the model
evaluate(model, test_loader)
