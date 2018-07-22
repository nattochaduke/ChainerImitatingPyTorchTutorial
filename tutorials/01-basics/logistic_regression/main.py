import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.dataset import convert
from chainer import configuration
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import numpy as np

# Hyper-parameters
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train, test = chainer.datasets.get_mnist()

# Data loader (input pipeline)
# Setting repeat=False, the iterator raise stopiteration when the iterator runs out of
# data i.e. the end of the epoch.
train_iter = iterators.SerialIterator(dataset=train, batch_size=batch_size, repeat=False)

test_iter = iterators.SerialIterator(dataset=test, batch_size=batch_size, repeat=False)

# Logistic regression model
model = L.Linear(input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
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
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        model.cleargrads()
        loss.backward()
        optimizer.update()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.data))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with chainer.using_config('enable_backprop', False): # disable backprop
    correct = 0
    total = 0
    for batch in test_iter:
        images, labels = convert.concat_examples(batch)
        images = images.reshape(-1, 28 * 28)
        outputs = model(images)
        predicted = np.argmax(outputs.data, 1)
        total += labels.size
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
serializers.save_npz("mymodel.npz", model)