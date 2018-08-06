import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import numpy as np



# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic autograd example 1               (Line 31 to 45)
# 2. Basic autograd example 2               (Line 53 to 91)
# 3. Loading data from numpy                (Line 97 to 106)
# 4. Input pipline                          (Line 112 to 141)
# 5. Input pipline for custom dataset       (Line 148 to 169)
# 6. Pretrained model                       (Line 176 to 206)
# 7. Save and load model                    (Line 213 to 215)


# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

# Create variables. In chainer, Variables wraps numpy or cupy arrays.
x = Variable(np.array([1], dtype=np.float32))
w = Variable(np.array([2], dtype=np.float32))
b = Variable(np.array([3], dtype=np.float32))

# Build a computational graph.
y = w * x + b    # y = 2 * x + 3

# Compute gradients.
y.backward()

# Print out the gradients.
print(x.grad)    # x.grad = 2
print(w.grad)    # w.grad = 1
print(b.grad)    # b.grad = 1


# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #

# Create tensors of shape (10, 3) and (10, 2).
x = Variable(np.random.randn(10, 3).astype('f'))
y = Variable(np.random.randn(10, 2).astype('f'))

# Build a fully connected layer.
linear = L.Linear(3, 2)
linear.cleargrads()
print ('w: ', linear.W)
print ('b: ', linear.b)

# Build loss function and optimizer.
criterion = F.mean_squared_error
optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(linear)

# Forward pass.
pred = linear(x)

# Compute loss.
loss = criterion(pred, y)
print('loss: ', loss.data)

# Backward pass.
loss.backward()

# Print out the gradients.
print ('dL/dw: ', linear.W.grad)
print ('dL/db: ', linear.b.grad)

# 1-step gradient descent.
optimizer.update()

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.data)

# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #

# Chainer.Variable wraps numpy array or cupy array.

# Create a numpy array
x = np.array([[1, 2], [3, 4]])

# Convert the numpy array to a chainer.Variable
y = Variable(x)

# Convert the Variable to a numpy array.
z = y.data

# ================================================================== #
#                         4. Input pipline                           #
# ================================================================== #

# Download and construct CIFAR-10 dataset.
train, test = chainer.datasets.get_cifar10()

# Fetch one data pair (read data from disk).
image, label = train[0]
print (image.shape)
# Variable.shape returns a tuple of dimensions of each axis of the variable.
# Variable.size returns a integer that is product of Variable.shape
print (label)


# Data iterator (this provides queues and threads in a very simple way).
# iterators have Serial/Multiprocess/mMultithread variants.
train_iter = iterators.SerialIterator(dataset=train, batch_size=64)

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_iter)

# Mini-batch images and labels.
# data_iter.next() yields a list that has the shape
# [(image0, label0), (image1, label1), ..., (image63, label63)]
# convert.concat_examples transforms this list into
# (array([image0, image1, ..., image63]), array([label0, ..., label63]))
images, labels = convert.concat_examples(data_iter.next())

# Actual usage of the data loader is as below.
for batch in train_iter:
    images, labels = convert.concat_examples(batch)
    # Training code should be written here.
    break


# ================================================================== #
#                5. Input pipline for custom dataset                 #
# ================================================================== #

# You should your build your custom dataset as below.
class CustomDataset(chainer.dataset.DatasetMixin):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names.
        pass
    def get_example(self, index):
        # DatasetMixin.get_example method is called by DatasetMixin.__getitem__
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0

# You can then use the prebuilt data loader.
custom_dataset = CustomDataset()
train_iter = iterators.SerialIterator(dataset=custom_dataset,
                                           batch_size=64,
                                           shuffle=True)


# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Load the pretrained ResNet-18.
# Beforehand, have to download pretrained model manually.
resnet = chainer.links.ResNet50Layers(pretrained_model='auto')

# I have no idea how to concisely write fine-tuning model.
# In spite the verbosity, I would define a network class that contains
# freezed pretrained model and learnable fully-connected layer.
class ResNetFineTune(chainer.Chain):
    def __init__(self, out_size):
        super(ResNetFineTune, self).__init__()
        self.base = chainer.links.ResNet50Layers(pretrained_model='auto')
        # base will not be updated because it is defined out of init_scope context.

        with self.init_scope():
            # Layers defined in init_scope context are updated.
            self.fc = L.Linear(None, out_size)
            # By setting the dimension of input None, the number is determined
            # when the instance of this class has first input.

    def __call__(self, x):
        h = self.base.extract(x)["pool5"]
        # PretrainedModel.extract method gives the output of each layer in the pretrained model
        y = self.fc(h)
        return y

model = ResNetFineTune(100)
# Forward pass.
images = np.random.randn(64, 3, 224, 224)
outputs = model(images)   # Inputting numpy array, chainer automatically
                           # wraps it in Variable.
print(outputs.shape)     # (64, 100)


# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# Save and load only the model parameters.
serializers.save_npz("mymodel.npz", model)
serializers.load_npz('mymodel.npz', model)
