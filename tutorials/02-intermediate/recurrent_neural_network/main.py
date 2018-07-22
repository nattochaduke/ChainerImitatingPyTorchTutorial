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
train, test = get_mnist(ndim=2) # (28, 28)

train_iter = iterators.SerialIterator(train, batch_size, repeat=True, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

# ============================ Model Definition ==========================

class RNN(Chain):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        with self.init_scope():
            self.nsteplstm = L.NStepLSTM(num_layers, input_size, hidden_size,
                                         dropout=0)  # chainer.Links.NStepLSTM and torch.nn.LSTM are alike.
            self.fc = L.Linear(hidden_size, num_classes)

    def __call__(self, x):
        x = list(map(Variable, x))
        hy, cy, ys = self.nsteplstm(hx=None, cx=None, xs=x)
        """
        L.NStepLSTM.__call__()
        parameters:
            hx is a Variable (or None) of initila hidden states. If None is specified zero-vector is used.
            cx is a Variable (or None) of initila cell states. If None is specified zero-vector is use.
            xs is a list of Variable s, or list of input sequences.
        returns:
            hy is an updated hidden states wose shape is the same as hx
            cy is an updated cell states whose shape is the same as cx
            ys is a list of variable. Each element ys[t] holds hidden states of the last layer
              correspoding to an input xs[t].
        """
        out = F.stack(ys)  # list of batches to a variable
        out = out[:, -1, :]  # The outputs of last timestep of samples in the batch (batch, hidden_size)
        out = self.fc(out)
        return out

# =====================Trainer setup ========================================

model = RNN(input_size, hidden_size, num_layers, num_classes)
model = L.Classifier(model)
if device == 'gpu':
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


"""
# ============================ PyTorch Style ===============================
# initilization of model
model = RNN(input_size ,hidden_size, num_layers, num_classes)
model = L.Classifier(model)
if device == 'gpu':
    model.to_gpu(device_id)
optimizer = chainer.optimizers.Adam(alpha=learning_rate)
optimizer.setup(model)

train_iter.reset()
test_iter.reset()

sum_accuracy = 0
sum_loss = 0
train_count = len(train)
test_count = len(test)

print('epoch \t main/loss \t main/accuracy \t validation/main/loss \t validation/main/accuracy \t elapsed time')
start = time.time()
while train_iter.epoch < num_epochs:
    batch = train_iter.next() # [(data0, label0), (data1, label1), ...]
    x_array, t_array = convert.concat_examples(batch, device_id) # x_array = [data0, data1, ..,]
                                                                 # t_array = [label0, label1, ...]
                                                                 # and they are transported to device

    x, t = x_array, t_array
    optimizer.update(model, x, t)
    sum_loss += float(model.loss.data) * len(t)
    sum_accuracy += float(model.accuracy.data) * len(t)
    if train_iter.is_new_epoch:
        train_loss = sum_loss / train_count
        train_accuracy = sum_accuracy / train_count
        sum_loss, sum_accuracy = 0, 0
        with configuration.using_config('train', False): # evaluation mode
            with chainer.using_config('enable_backprop', False):
                for batch in test_iter:
                    x_array, t_array = convert.concat_examples(batch, device_id)
                    x, t = x_array, t_array
                    loss = model(x, t)
                    sum_loss += float(loss.data) * len(t)
                    sum_accuracy += float(model.accuracy.data) * len(t)
        test_iter.reset()
        test_loss = sum_loss / test_count
        test_accuracy = sum_accuracy / test_count
                    
        print(f'{train_iter.epoch} \t {train_loss:.4f} \t {train_accuracy:.4f} \t {test_loss:.4f}\
                    \t\t {test_accuracy:.4f} \t\t{time.time()-start}')
        sum_accuracy = 0
        sum_loss = 0

"""