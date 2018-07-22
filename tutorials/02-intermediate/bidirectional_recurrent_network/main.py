import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer import configuration
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.dataset import convert
from chainer.datasets import get_mnist

# Device configuration
device = 'gpu' if chainer.cuda.available else 'cpu'
device_id = 0 if chainer.cuda.available else -1
print(device)

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# ============================ Data Preparation =======================
# MNIST dataset and iterators
train, test = get_mnist(ndim=2)  # (28, 28)

train_iter = iterators.SerialIterator(train, batch_size, repeat=True, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)


# ============================ Model Definition ==========================

class BiRNN(Chain):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        with self.init_scope():
            self.nsteplstm = L.NStepBiLSTM(num_layers, input_size, hidden_size,
                                           dropout=0)  # chainer.Links.NStepLSTM and torch.nn.LSTM are alike.
            self.fc = L.Linear(2 * hidden_size, num_classes)

    def __call__(self, x):
        # print(x)
        # x = list(x)
        # print(len(x))
        # global test
        # test = x
        # print(type(x[0].data))
        # print(len(x[0].data))

        x = list(map(Variable, x))
        hy, cy, ys = self.nsteplstm(hx=None, cx=None, xs=x)
        out = F.stack(ys)  # list of batches to a variable
        out = out[:, -1, :]  # The outputs of last timestep of samples in the batch (batch, hidden_size)
        out = self.fc(out)
        return out


# =====================Trainer setup ========================================

model = BiRNN(input_size ,hidden_size, num_layers, num_classes)
model = L.Classifier(model)
if device=='gpu':
    model.to_gpu()

optimizer = chainer.optimizers.Adam(learning_rate)
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=device_id)
trainer = training.Trainer(updater, stop_trigger=(num_epochs, 'epoch'), out='mnist_result')

trainer.extend(extensions.LogReport())
trainer.extend(extensions.Evaluator(test_iter, model, device=device_id))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy',
                                       'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
trainer.run()