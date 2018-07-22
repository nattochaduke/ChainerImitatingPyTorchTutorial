import chainer
from chainer.backends import cuda
from chainer import Function, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.dataset import convert
from chainer import configuration
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainercv import transforms
import numpy as np

# Device configuration
device = 'gpu' if chainer.cuda.available else 'cpu'
device_id = 0 if chainer.cuda.available else -1
print(device)

# Hyper-parameters
num_epochs = 80
learning_rate = 0.001
batch_size = 100


#============================= Data Preparation =======================================
# CIFAR-10 dataset
train, test = datasets.get_cifar10()


# Image preprocessing modules
# chainercv.transforms contains functions those are useful for data augmentation.
# You can implement your own augmentation functions.

def pad(image, pad=0):
    """
    zero padding to image
    """
    image_type = image.dtype
    # the image has shape (color_channels, height, width)
    canvas = np.zeros([image.shape[0], image.shape[1] + 2 * pad, image.shape[2] + 2 * pad], dtype=image_type)
    canvas[:, pad: image.shape[1] + pad, pad: image.shape[2] + pad] += image
    return canvas

def transform(inputs):
    img, label = inputs
    img = img.copy()
    img = pad(img, pad=4)
    img = transforms.random_flip(img, x_random=True)
    img = transforms.random_crop(img, (32, 32))
    return img, label


# To enable data augmentation, wrap the raw dataset and transform function with TransformDataset,
train = chainer.datasets.TransformDataset(train, transform)

# data iterators
train_iter = iterators.SerialIterator(train, batch_size, repeat=True, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

#==================================== Model Definition================================================

# Residual block
class ResidualBlock(Chain):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels, out_channels, ksize=3, stride=stride, pad=1, nobias=True)
            self.bn1 = L.BatchNormalization(out_channels)
            self.conv2 = L.Convolution2D(out_channels, out_channels, ksize=3, stride=1, pad=1, nobias=True)
            self.bn2 = L.BatchNormalization(out_channels)
            self.downsample = downsample

    def __call__(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out

# ResNet
class ResNet(Chain):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        with self.init_scope():
            self.in_channels = 16
            self.conv = L.Convolution2D(3, 16, ksize=3, stride=1, pad=1)
            self.bn = L.BatchNormalization(16)
            self.layer1 = self.make_layer(block, 16, layers[0])
            self.layer2 = self.make_layer(block, 32, layers[0], 2)
            self.layer3 = self.make_layer(block, 64, layers[1], 2)
            self.fc = L.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = chainer.Sequential(
                L.Convolution2D(self.in_channels, out_channels, ksize=3, stride=stride, pad=1, nobias=True),
                L.BatchNormalization(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return chainer.Sequential(*layers)

    def __call__(self, x):
        out = self.conv(x)  # (N, 3, 32, 32) -> (N, 16, 32, 32)
        out = self.bn(out)
        out = F.relu(out)
        out = self.layer1(out)  # (N, 16, 32, 32) -> (N, 16, 32, 32)
        out = self.layer2(out)  # (N, 16, 32, 32) -> (N, 32, 16, 16)
        out = self.layer3(out)  # (N, 32, 16, 16) -> (N, 64, 8, 8)
        out = F.average_pooling_2d(out, ksize=8)  # (N, 64, 8, 8) -> (N, 64, 1, 1)
        out = out.reshape(out.shape[0], -1)  # (N, 64, 1, 1) -> (N, 64)
        out = self.fc(out)
        return out

model = ResNet(ResidualBlock, [2, 2, 2, 2])

# ================================= trainer setting ===========================================

# loss
model = L.Classifier(model)
if device == 'gpu':
    model.to_gpu()

# optimizer
optimizer = chainer.optimizers.Adam(alpha=learning_rate)
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=device_id)
trainer = training.Trainer(updater, stop_trigger=(80, 'epoch'), out='cifar10_result')

trainer.extend(extensions.LogReport(keys=["main/loss", "validation/main/accuracy", "lr"],
                                    trigger=training.triggers.IntervalTrigger(100, 'iteration')))
                # logging statistics during calculation once a 100 iterations.

trainer.extend(extensions.ExponentialShift('alpha', 1/3),
               trigger=training.triggers.IntervalTrigger(20, 'epoch'))
                # reduce learning rate 1/3 once a 20 epochs
                # Since Adam's leraning rate in its original paper is "alpha",
                # chainer's Adam has attribute "alpha" instead of "lr".

trainer.extend(extensions.Evaluator(test_iter, model, device=device_id),
               trigger=training.triggers.IntervalTrigger(3, 'epoch'))#
                # conduct evaluation at the end of training only
trainer.extend(extensions.observe_lr(), trigger=training.triggers.IntervalTrigger(100, 'iteration')) # log the learning rate

trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time', 'lr']),
               trigger=training.triggers.IntervalTrigger(100, 'iteration')) # print statistics once a 100 iterations
trainer.run() # Train the model

