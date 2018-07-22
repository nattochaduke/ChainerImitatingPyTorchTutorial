import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer import configuration
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.dataset import convert
from chainer.datasets import get_mnist
import matplotlib.pyplot as plt

# Device configuration
device = 'gpu' if chainer.cuda.available else 'cpu'
device_id = 0 if chainer.cuda.available else -1

# Hyper-parameters
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 15
batch_size = 128
learning_rate = 1e-3

# ======================Data Preparation ====================================

# MNIST dataset and iterators
train, test = get_mnist(ndim=1) # (784)
train_iter = iterators.SerialIterator(train, batch_size, repeat=True, shuffle=False)

# ========================== Model Definition ===============================

class VAE(Chain):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(image_size, h_dim)
            self.fc2 = L.Linear(h_dim, z_dim)
            self.fc3 = L.Linear(h_dim, z_dim)
            self.fc4 = L.Linear(z_dim, h_dim)
            self.fc5 = L.Linear(h_dim, image_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h) # mu, log_var

    def reparameterize(self, mu, log_var):
        z = F.gaussian(mu, log_var) # F.gaussian takes a mean and logarithm of variance log(sigma**2) as inputs
                                    # and outputs a sample drawn from a gaussian N(mu, sigma)
        return z

    def decode(self, z):
        h = F.relu(self.fc4(z))
        # return F.sigmoid(self.fc5(h))
        # Since F.sigmoid_cross_entropy takes pre-sigmoid activation as input,
        # in contrast to PyTorch's F.binary_cross_entorpy, we have to omit sigmoid.

        return self.fc5(h)

    def __call__(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

# ========================= Training =====================================

def vae_loss(x_reconst, t):
    reconstruct_loss = F.sigmoid_cross_entropy(x_reconst, t, normalize=False)
    kl_div = -0.5 * F.sum(1 + log_var - )

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
