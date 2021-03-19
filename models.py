import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = (-1,) + shape

    def extra_repr(self):
        return "shape={}".format(self.shape[1:])

    def forward(self, inp):
        return inp.view(self.shape)


def get_discriminator():
    # TODO: expects input tensor of shape (batch, 1, 32, 32)
    return nn.Sequential(
        nn.Conv2d(1, 16, (5,5)),
        nn.LeakyReLU(),
        nn.Conv2d(16, 32, (5,5)),
        nn.MaxPool2d(2),
        nn.LeakyReLU(),
        nn.Conv2d(32, 64, (5,5)),
        nn.MaxPool2d(2),
        nn.LeakyReLU(),
        nn.Flatten(),
        nn.Linear(1024, 1)
    )


def get_generator(embedding_size=256):
    return nn.Sequential(
        nn.Linear(embedding_size, 32*8*8),
        nn.LeakyReLU(),
        Reshape((32, 8, 8)),
        nn.ConvTranspose2d(32, 64, (5,5)),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(64, 128, (5,5)),
        nn.LeakyReLU(),
        nn.Upsample(scale_factor=2),
        nn.ConvTranspose2d(128, 64, (3,3)),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(64, 64, (3,3)),
        nn.LeakyReLU(),
        nn.Conv2d(64, 1, (5,5))
    )

