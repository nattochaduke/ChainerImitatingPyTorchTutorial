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
import numpy as np
import cupy as cp

# Device configuration
device = 'gpu' if chainer.cuda.available else 'cpu'
device_id = 0 if chainer.cuda.available else -1
print("device: ", device)

# Hyper-parameters
num_epochs = 80
learning_rate = 0.001
batch_size = 100

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train, test = chainer.datasets.get_mnist(ndim=3)

# Data loader (input pipeline)
# Setting repeat=False, the iterator raise stopiteration when the iterator runs out of
# data i.e. the end of the epoch.
train_iter = iterators.SerialIterator(dataset=train, batch_size=batch_size, repeat=False)
test_iter = iterators.SerialIterator(dataset=test, batch_size=batch_size, repeat=False)


# In PyTorch they implemented layers by nn.Sequential class.
# Chainer v5 has similar class but it is experimental we avoid using it.
class ConvLayer(Chain):
    def __init__(self, input_size, output_size):
        super(ConvLayer, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(input_size, output_size, ksize=5, stride=1, pad=2)
            self.bn = L.BatchNormalization(output_size)

    def __call__(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = F.max_pooling_2d(out, ksize=2)
        return out

class ConvNet(Chain):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ConvNet, self).__init__()
        # In chainer, learnable layers are defined in init_scope context.
        with self.init_scope():
            self.fc1 = ConvLayer(1, 16)
            self.fc2 = ConvLayer(16, 32)
            self.fc = L.Linear(7*7*32, num_classes)

    def __call__(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = F.reshape(out, (-1, 7*7*32))
        out = self.fc(out)
        return out

# ========================PyTorch style training.=============================
model = ConvNet(input_size, hidden_size, num_classes)
if device=="gpu":
    model.to_gpu(device_id)

# Loss and optimizer
criterion = F.softmax_cross_entropy

optimizer = optimizers.SGD(lr=learning_rate)
optimizer.setup(model)

# Chainer iterator has no method len()
total_step = sum(1 for _ in train_iter)
# Train the model
for epoch in range(num_epochs):
    # Since at the end of each epoch iterator raise stopiteration
    train_iter.reset()
    for i, batch in enumerate(train_iter):
        images, labels = convert.concat_examples(batch)
        images, labels = Variable(images), Variable(labels)
        if device == "gpu":
            images.to_gpu(device_id)
            labels.to_gpu(device_id)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        model.cleargrads()
        loss.backward()
        optimizer.update()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, cuda.to_cpu(loss.data)))
            # loss.data is a cupy array (on GPU). chainer.cuda.to_cpu function transports
            # cupy array to numpy array (cpu)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# and switch off the stochastical elements of the network e.g. batch normalization or dropout.
with chainer.using_config('enable_backprop', False): # disable backprop
    with configuration.using_config('train', False): # evaluation mode - affects batchnorm or dropout
        correct = 0
        total = 0
        for batch in test_iter:
            images, labels = convert.concat_examples(batch)
            images, labels = Variable(images), Variable(labels)
            if device == "gpu":
                images.to_gpu(device_id)
                labels.to_gpu(device_id)
            outputs = model(images)
            total += labels.size
            if device == "gpu":
                correct += (cuda.to_cpu(cp.argmax(outputs.data, 1)) == cuda.to_cpu(labels.data)).sum()
            else:
                correct += (np.argmax(outputs.data, 1) == labels.data).sum()


        print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
# To save parameters, model should be back to cpu
if device == "gpu":
    model.to_cpu()
serializers.save_npz("mymodel.npz", model)


# ========================Chainer style training.=============================
# Chainer provides a standard implementation of the training loops under the chainer.training module.
# By using trainer and its extensions, we can abstract training loops. We will view the example of trainer.

train_iter = iterators.SerialIterator(dataset=train, batch_size=batch_size, repeat=True)
test_iter = iterators.SerialIterator(dataset=test, batch_size=batch_size, repeat=False)

model = ConvNet(input_size, hidden_size, num_classes)
model = L.Classifier(model) # L.Classifier abstracts softmax and crossentropy loss.
if device == 'gpu':
    model.to_gpu(device_id)
optimizer = chainer.optimizers.SGD(lr=learning_rate)
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=device_id) # training data is transported to device.
trainer = training.Trainer(updater, stop_trigger=(num_epochs, 'epoch'), out='mnist_result')

trainer.extend(extensions.LogReport()) # log reports. reports are given by Reporter's instances in trainer.
trainer.extend(extensions.Evaluator(test_iter, model, device=device_id), trigger=(num_epochs, 'epoch'))
  # Evaluate the model using test_iter as validation data at the end of training
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy']))
  # print statistics in evely epoch.
# trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
# trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
# trainer.extend(extensions.dump_graph('main/loss'))
trainer.run()