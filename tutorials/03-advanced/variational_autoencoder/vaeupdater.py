from chainer import Function, reporter, training, utils, Variable
from chainer.dataset import iterator as iterator_module
from chainer.dataset import convert


class VAEUpdater(training.StandardUpdater):
    def __init__(self, iterator, vae, optimizer, converter=convert.concat_examples, device=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main':iterator}
        self._iterators = iterator
        self._optimizers = {"main": optimizer}
        self.converter = converter
        self.device = device
        self.iteration = 0

    def update_core(self):
        batch = self._iterators['main'].next()
        images, _ = self.converter(batch, self.device)
        batch_size = images.shape[0]

        # ================================================================== #
        #                      Calculation of Loss                           #
        # ================================================================== #



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
