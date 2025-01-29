# # Adversarial Variational Bayes toy example
#
# Based on:
#
# - https://github.com/LMescheder/AdversarialVariationalBayes/blob/master/notebooks/AVB_toy_example.ipynb
# - https://gist.github.com/poolio/b71eb943d6537d01f46e7b20e9225149
# - https://github.com/wiseodd/generative-models/blob/master/VAE/adversarial_vb/avb_pytorch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import defaultdict
import time

# Hyperparameters
args = {
    "batch_size": 512,
    "latent_dim": 2,  # Dimensionality of latent space
    "eps_dim": 8,  # Dimensionality of epsilon, used in inference net, z_phi(x, eps)
    "input_dim": 4,  # Dimensionality of input (also the number of unique datapoints)
    "n_hidden": 64,  # Number of hidden units in encoder/decoder networks
    "n_hidden_disc": 64,  # Number of hidden units in discriminator
    "lr_primal": 1e-4,  # Learning rate for encoder/decoder
    "lr_dual": 1e-3,  # Learning rate for discriminator (needs to be faster than encoder/decoder)
    "device": (
        "mps" if torch.backends.mps.is_available() else "cpu"
    ),  # Use MPS if available
}

# Generate synthetic data
n = args["batch_size"] // args["input_dim"]
x_real = np.eye(args["input_dim"]).repeat(n, axis=0)
labels = x_real.argmax(axis=1)
x_real = torch.FloatTensor(x_real).to(args["device"])


# Define models
class Encoder(nn.Module):
    def __init__(self, X_dim, eps_dim, h_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(X_dim + eps_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, z_dim),
        )

    def forward(self, x, eps):
        x = torch.cat([x, eps], dim=1)
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, X_dim, h_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, X_dim),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, X_dim, h_dim, z_dim):
        super().__init__()
        self.fc0 = nn.Linear(X_dim + z_dim, h_dim)
        self.mid = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim)
                )
                for _ in range(2)
            ]
        )
        self.fc1 = nn.Linear(h_dim, 1)

    def forward(self, x, z):
        x = torch.cat([x, z], dim=1)
        x = F.elu(self.fc0(x))

        for layer in self.mid:
            x = x + layer(x)
            x = F.elu(x)

        x = self.fc1(x).view(-1)
        x = x + torch.sum(z**2, dim=1)  # Add regularization term
        return x


# Build AVB model
def build_avb(X_dim, eps_dim, h_dim, z_dim, device):
    Q = Encoder(X_dim, eps_dim, h_dim, z_dim).to(device)
    P = Decoder(X_dim, h_dim, z_dim).to(device)
    T = Discriminator(X_dim, h_dim, z_dim).to(device)
    return Q, P, T


Q, P, T = build_avb(
    args["input_dim"],
    args["eps_dim"],
    args["n_hidden"],
    args["latent_dim"],
    args["device"],
)

# Optimizers
opt_primal = optim.Adam(
    list(Q.parameters()) + list(P.parameters()), lr=args["lr_primal"]
)
opt_dual = optim.Adam(T.parameters(), lr=args["lr_dual"])

# Loss tracking
losses = defaultdict(list)


# Training loop
try:
    start_time = time.time()
    for it in range(4000):
        # Sample noise and latent variables
        eps = torch.randn(args["batch_size"], args["eps_dim"], device=args["device"])
        z_sampled = torch.randn(
            args["batch_size"], args["latent_dim"], device=args["device"]
        )

        # --- Primal Optimization (Encoder/Decoder) ---
        z_inferred = Q(x_real, eps)
        x_reconstr = P(z_inferred)
        T_joint = T(x_real, z_inferred)

        reconstr_err = F.binary_cross_entropy_with_logits(
            x_reconstr, x_real, reduction="none"
        ).sum(dim=1)
        loss_primal = torch.mean(reconstr_err + T_joint)

        opt_primal.zero_grad()
        loss_primal.backward(retain_graph=True)
        opt_primal.step()

        # --- Dual Optimization (Discriminator) ---
        T_separate = T(x_real, z_sampled)
        loss_dual = -torch.mean(F.logsigmoid(T_joint) + F.logsigmoid(-T_separate))

        opt_dual.zero_grad()
        loss_dual.backward()
        opt_dual.step()

        # Log losses
        losses["loss_primal"].append(loss_primal.item())
        losses["loss_dual"].append(loss_dual.item())
        losses["reconstr_err"].append(torch.mean(reconstr_err).item())
        losses["T_joint"].append(torch.mean(T_joint).item())

        # Visualization
        if it % 100 == 0:
            results = []
            n_viz = 4
            for _ in range(n_viz):
                eps = torch.randn(
                    args["batch_size"], args["eps_dim"], device=args["device"]
                )
                z_sample = Q(x_real, eps)
                results.append(z_sample.detach().cpu().numpy())
            z = np.vstack(results)

            plt.figure(figsize=(14, 6), facecolor="white")
            plt.title("AVB Latent Space")
            plt.subplot(121, aspect="equal")
            plt.scatter(
                z[:, 0], z[:, 1], c=np.tile(labels, n_viz), edgecolor="none", alpha=0.5
            )
            plt.xlim(-3, 3)
            plt.ylim(-3.5, 3.5)

            plt.subplot(122)
            for key in losses:
                plt.plot(losses[key], label=key)
            plt.legend(loc="upper right")

            clear_output(wait=True)
            plt.show()
            speed = (it + 1) / (time.time() - start_time)
            print(f"Iteration {it}: {speed:.2f} iterations/sec")

except KeyboardInterrupt:
    pass
# |%%--%%| <c1X0j8gG39|pgp1GTDsPM>


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import defaultdict
import time

# Hyperparameters
args = {
    "batch_size": 512,
    "latent_dim": 2,  # Dimensionality of latent space
    "eps_dim": 8,  # Dimensionality of epsilon, used in inference net, z_phi(x, eps)
    "input_dim": 4,  # Dimensionality of input (also the number of unique datapoints)
    "n_hidden": 64,  # Number of hidden units in encoder/decoder networks
    "n_hidden_disc": 64,  # Number of hidden units in discriminator
    "lr_primal": 1e-4,  # Learning rate for encoder/decoder
    "lr_dual": 1e-3,  # Learning rate for discriminator (needs to be faster than encoder/decoder)
    "device": (
        "mps" if torch.backends.mps.is_available() else "cpu"
    ),  # Use MPS if available
    "iterations": 4000,  # Number of iterations
    "viz_interval": 100,  # Visualization interval
}

# Generate synthetic data
n = args["batch_size"] // args["input_dim"]
x_real = np.eye(args["input_dim"]).repeat(n, axis=0)
labels = x_real.argmax(axis=1)
x_real = torch.FloatTensor(x_real).to(args["device"])


# Define models
class Encoder(nn.Module):
    def __init__(self, X_dim, eps_dim, h_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(X_dim + eps_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, z_dim),
        )

    def forward(self, x, eps):
        x = torch.cat([x, eps], dim=1)
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, X_dim, h_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, X_dim),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, X_dim, h_dim, z_dim):
        super().__init__()
        self.fc0 = nn.Linear(X_dim + z_dim, h_dim)
        self.mid = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim)
                )
                for _ in range(2)
            ]
        )
        self.fc1 = nn.Linear(h_dim, 1)

    def forward(self, x, z):
        x = torch.cat([x, z], dim=1)
        x = F.elu(self.fc0(x))

        for layer in self.mid:
            x = x + layer(x).clone()  # Use `.clone()` to avoid in-place modification
            x = F.elu(x)

        x = self.fc1(x).view(-1)
        x = x + torch.sum(z**2, dim=1)  # Add regularization term
        return x


# Build AVB model
def build_avb(X_dim, eps_dim, h_dim, z_dim, device):
    Q = Encoder(X_dim, eps_dim, h_dim, z_dim).to(device)
    P = Decoder(X_dim, h_dim, z_dim).to(device)
    T = Discriminator(X_dim, h_dim, z_dim).to(device)
    return Q, P, T


Q, P, T = build_avb(
    args["input_dim"],
    args["eps_dim"],
    args["n_hidden"],
    args["latent_dim"],
    args["device"],
)

# Optimizers
opt_primal = optim.Adam(
    list(Q.parameters()) + list(P.parameters()), lr=args["lr_primal"]
)
opt_dual = optim.Adam(T.parameters(), lr=args["lr_dual"])

# Loss tracking
losses = defaultdict(list)

torch.autograd.set_detect_anomaly(True)
# Training loop
try:
    start_time = time.time()
    for it in range(args["iterations"]):
        # Sample noise and latent variables
        eps = torch.randn(args["batch_size"], args["eps_dim"], device=args["device"])
        z_sampled = torch.randn(
            args["batch_size"], args["latent_dim"], device=args["device"]
        )

        # --- Primal Optimization (Encoder/Decoder) ---
        z_inferred = Q(x_real, eps)
        x_reconstr = P(z_inferred)
        T_joint = T(x_real, z_inferred)

        reconstr_err = F.binary_cross_entropy_with_logits(
            x_reconstr, x_real, reduction="none"
        ).sum(dim=1)
        loss_primal = torch.mean(reconstr_err + T_joint)

        opt_primal.zero_grad()
        loss_primal.backward(retain_graph=True)
        opt_primal.step()

        # --- Dual Optimization (Discriminator) ---
        T_separate = T(x_real, z_sampled)
        loss_dual = -torch.mean(F.logsigmoid(T_joint) + F.logsigmoid(-T_separate))

        opt_dual.zero_grad()
        loss_dual.backward()
        opt_dual.step()

        # Log losses
        losses["loss_primal"].append(loss_primal.item())
        losses["loss_dual"].append(loss_dual.item())
        losses["reconstr_err"].append(torch.mean(reconstr_err).item())
        losses["T_joint"].append(torch.mean(T_joint).item())

        # Visualization
        if it % args["viz_interval"] == 0:
            results = []
            n_viz = 4
            for _ in range(n_viz):
                eps = torch.randn(
                    args["batch_size"], args["eps_dim"], device=args["device"]
                )
                z_sample = Q(x_real, eps)
                results.append(z_sample.detach().cpu().numpy())
            z = np.vstack(results)

            plt.figure(figsize=(14, 6), facecolor="white")
            plt.title("AVB Latent Space")
            plt.subplot(121, aspect="equal")
            plt.scatter(
                z[:, 0], z[:, 1], c=np.tile(labels, n_viz), edgecolor="none", alpha=0.5
            )
            plt.xlim(-3, 3)
            plt.ylim(-3.5, 3.5)

            plt.subplot(122)
            for key in losses:
                plt.plot(losses[key], label=key)
            plt.legend(loc="upper right")

            clear_output(wait=True)
            plt.show()

            speed = (it + 1) / (time.time() - start_time)
            print(f"Iteration {it}: {speed:.2f} iterations/sec")

except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    torch.save(
        {
            "encoder": Q.state_dict(),
            "decoder": P.state_dict(),
            "discriminator": T.state_dict(),
        },
        "avb_checkpoint.pth",
    )
    print("Model saved as 'avb_checkpoint.pth'.")
#|%%--%%| <pgp1GTDsPM|TFKUTkQC36>
r"""°°°
from 
https://chrisorm.github.io/AVB-pyt.html
°°°"""
# |%%--%%| <TFKUTkQC36|wtduIwOxGT>

import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

representation_size = 2
input_size = 4
n_samples = 2000
batch_size = 500
gen_hidden_size = 200
enc_hidden_size = 200
disc_hidden_size = 200
# |%%--%%| <wtduIwOxGT|YRcX94Huby>

n_samples_per_batch = n_samples // input_size

y = np.array([i for i in range(input_size) for _ in range(n_samples_per_batch)])

d = np.identity(input_size)
x = np.array([d[i] for i in y], dtype=np.float32)

# |%%--%%| <YRcX94Huby|12gUIkMdSB>


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.gen_l1 = torch.nn.Linear(representation_size, gen_hidden_size)
        self.gen_l2 = torch.nn.Linear(gen_hidden_size, input_size)

        self.enc_l1 = torch.nn.Linear(input_size + representation_size, enc_hidden_size)
        self.enc_l2 = torch.nn.Linear(enc_hidden_size, representation_size)

        self.disc_l1 = torch.nn.Linear(
            input_size + representation_size, disc_hidden_size
        )
        self.disc_l2 = torch.nn.Linear(disc_hidden_size, 1)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def sample_prior(self, s):
        if self.training:
            m = torch.zeros((s.data.shape[0], representation_size))
            std = torch.ones((s.data.shape[0], representation_size))
            d = Variable(torch.normal(m, std))
        else:
            d = Variable(torch.zeros((s.data.shape[0], representation_size)))

        return d

    def discriminator(self, x, z):
        i = torch.cat((x, z), dim=1)
        h = self.relu(self.disc_l1(i))
        return self.disc_l2(h)

    def sample_posterior(self, x):
        i = torch.cat((x, self.sample_prior(x)), dim=1)
        h = self.relu(self.enc_l1(i))
        return self.enc_l2(h)

    def decoder(self, z):
        i = self.relu(self.gen_l1(z))
        h = self.sigmoid(self.gen_l2(i))
        return h

    def forward(self, x):
        z_p = self.sample_prior(x)

        z_q = self.sample_posterior(x)
        log_d_prior = self.discriminator(x, z_p)
        log_d_posterior = self.discriminator(x, z_q)
        disc_loss = torch.mean(
            torch.nn.functional.binary_cross_entropy_with_logits(
                log_d_posterior, torch.ones_like(log_d_posterior)
            )
            + torch.nn.functional.binary_cross_entropy_with_logits(
                log_d_prior, torch.zeros_like(log_d_prior)
            )
        )

        x_recon = self.decoder(z_q)
        recon_liklihood = (
            -torch.nn.functional.binary_cross_entropy(x_recon, x) * x.data.shape[0]
        )

        gen_loss = torch.mean(log_d_posterior) - torch.mean(recon_liklihood)

        return disc_loss, gen_loss


# |%%--%%| <12gUIkMdSB|CVC9ORzJLf>

model = VAE()
disc_params = []
gen_params = []
for name, param in model.named_parameters():

    if "disc" in name:

        disc_params.append(param)
    else:
        gen_params.append(param)
disc_optimizer = torch.optim.Adam(disc_params, lr=1e-3)
gen_optimizer = torch.optim.Adam(gen_params, lr=1e-3)

#|%%--%%| <CVC9ORzJLf|7ipHMUhCt3>

def train(epoch, batches_per_epoch = 501, log_interval=500):
    model.train()
    ind = np.arange(x.shape[0])
    for i in range(batches_per_epoch):
        data = torch.from_numpy(x[np.random.choice(ind, size=batch_size)])
        data = Variable(data, requires_grad=False)
        discrim_loss, gen_loss= model(data)
        gen_optimizer.zero_grad()
        gen_loss.backward(retain_graph=True)
        gen_optimizer.step()
        disc_optimizer.zero_grad()
        discrim_loss.backward(retain_graph=True)
        disc_optimizer.step()
        if (i % log_interval == 0) and (epoch % 1 ==0):
            #Print progress
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tLoss: {:.6f}'.format(
                epoch, i * batch_size, batch_size*batches_per_epoch,
                discrim_loss.data[0] / len(data), gen_loss.data[0] / len(data)))

    print('====> Epoch: {} done!'.format(
          epoch))
for epoch in range(1, 15):
    train(epoch)

#|%%--%%| <7ipHMUhCt3|aVVAd2xygv>

data = Variable(torch.from_numpy(x), requires_grad=False)

model.train()
zs = model.sample_posterior(data).data.numpy()
#|%%--%%| <aVVAd2xygv|yqKTF7OsVb>

import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(zs[:,0], zs[:, 1], c=y)



#|%%--%%| <yqKTF7OsVb|LkjxLsgIvT>

data = Variable(torch.from_numpy(x), requires_grad=False)
model.eval()
zs = model.sample_posterior(data).data.numpy()

plt.scatter(zs[:,0], zs[:, 1], c=y)
Out[13]:


#|%%--%%| <LkjxLsgIvT|iPVOv6fQla>

test_point = np.array([0.5, 0.6], dtype=np.float32).reshape(1,-1)
test_point = Variable(torch.from_numpy(test_point), requires_grad=False)
s = model.decoder(test_point)
s.data

#|%%--%%| <iPVOv6fQla|C4AocqhhJf>

test_point = np.array([0., 0.], dtype=np.float32).reshape(1,-1)
test_point = Variable(torch.from_numpy(test_point), requires_grad=False)
s = model.decoder(test_point)
s.data

