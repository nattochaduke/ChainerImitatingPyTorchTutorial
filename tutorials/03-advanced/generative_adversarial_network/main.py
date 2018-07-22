import numpy as np
import time


import chainer
from chainer.backends import cuda
from chainer import Function, reporter, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer import configuration
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer.datasets import get_mnist
import matplotlib.pyplot as plt


# Device configuration
device = 'gpu' if chainer.cuda.available else 'cpu'
device_id = 0 if chainer.cuda.available else -1

# Hyper-parameters
latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 100
sample_dir = 'samples'
learning_rate = 0.0002



# ======================Data Preparation ====================================

# MNIST dataset and iterators
train, test = get_mnist(ndim=1) # (784)
train_iter = iterators.SerialIterator(train, batch_size, repeat=True, shuffle=False)

# ========================== Model Definition ===============================

class Discriminator(Chain):
    def __init__(self):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(image_size, hidden_size)
            self.l2 = L.Linear(hidden_size, hidden_size)
            self.l3 = L.Linear(hidden_size, 1)

    def __call__(self, x):
        out = self.l1(x)
        out = F.leaky_relu(out, 0.2)
        out = self.l2(out)
        out = F.leaky_relu(out, 0.2)
        out = self.l3(out)
        return out


class Generator(Chain):
    def __init__(self):
        super(Generator, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(latent_size, hidden_size)
            self.l2 = L.Linear(hidden_size, hidden_size)
            self.l3 = L.Linear(hidden_size, image_size)

    def __call__(self, x):
        out = self.l1(x)
        out = F.relu(out)
        out = self.l2(out)
        out = F.relu(out)
        out = self.l3(out)
        out = F.tanh(out)
        return out

# =================================== Updater definition ====================

class GANUpdater(training.StandardUpdater):
    def __init__(self, iterator, discriminator, generator, dis_optimizer, gen_optimizer, latent_size,
                 converter=convert.concat_examples, device=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main':iterator}
        self.latent_size = latent_size
        self._iterators = iterator
        self.D = discriminator
        self.G = generator

        self._optimizers = {"d_optimizer": dis_optimizer, "g_optimizer": gen_optimizer}

        self.converter = converter
        self.device = device
        self.iteration = 0

    def reset_grad(self):
        for optimizer in self._optimizers.values():
            optimizer.target.cleargrads()

    def update_core(self):
        batch = self._iterators['main'].next()
        images, _ = self.converter(batch, self.device)
        images = Variable(2*images-1)
        batch_size = images.shape[0]

        real_labels = Variable(np.ones((batch_size, 1)).astype(np.int))
        real_labels.to_gpu(self.device)
        fake_labels = Variable(np.zeros((batch_size, 1)).astype(np.int))
        fake_labels.to_gpu(self.device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1

        outputs = self.D(images)
        d_loss_real = F.sigmoid_cross_entropy(outputs, real_labels)
        real_score = outputs

        z = Variable(np.random.randn(batch_size, self.latent_size).astype(np.float32))
        z.to_gpu(self.device)
        fake_images = self.G(z)
        outputs = self.D(fake_images)
        d_loss_fake = F.sigmoid_cross_entropy(outputs, fake_labels)
        fake_score = outputs
        d_loss = d_loss_real + d_loss_fake
        self.reset_grad()
        d_loss.backward()
        self._optimizers["d_optimizer"].update()

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        z = Variable(np.random.randn(batch_size, self.latent_size).astype(np.float32))
        z.to_gpu(self.device)

        fake_images = self.G(z)
        outputs = self.D(fake_images)

        g_loss = F.sigmoid_cross_entropy(outputs, real_labels)
        self.reset_grad()
        g_loss.backward()

        self._optimizers["g_optimizer"].update()

        reporter.report({'d_loss': d_loss, 'g_loss': g_loss})

# ======================== Training setup ================================

D = Discriminator()
G = Generator()
D.to_gpu(0)
G.to_gpu(0)

d_optimizer = chainer.optimizers.Adam(alpha=0.00002)
d_optimizer.setup(D)
g_optimizer = chainer.optimizers.Adam(alpha=0.00002)
g_optimizer.setup(G)

updater = GANUpdater(train_iter, D, G, d_optimizer, g_optimizer, latent_size,  device=device_id)
trainer = training.Trainer(updater, stop_trigger=(num_epochs, 'epoch'), out='mnist_result')

trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'd_loss', 'g_loss', 'elapsed_time']))
trainer.run()

z = Variable(np.random.randn(10, 64).astype(np.float32))
z.to_gpu(0)
fake_images = G(z)
fake_images.to_cpu()

for i in range(10):
    plt.imshow(fake_images.data[i].reshape([28, 28]))
    plt.show()