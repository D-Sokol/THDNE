#!/usr/bin/env python3

import itertools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import get_CJK, get_noise
from models import get_discriminator, get_generator

def train(disc, gen, real_dataset, noise_dataset, n_iter=100, k=5, batch_size=100):
    optim_d = torch.optim.Adam(disc.parameters())
    optim_g = torch.optim.Adam(gen.parameters())
    noise_loader = DataLoader(noise_dataset, batch_size=batch_size)
    data_loader = DataLoader(real_dataset, batch_size=batch_size)

    disc_target = torch.cat((torch.zeros(batch_size), torch.ones(batch_size)))
    gen_target = torch.ones(batch_size)

    for i in range(n_iter):
        for noise_batch, real_batch in itertools.islice(zip(noise_loader, data_loader), k):
            with torch.no_grad():
                fake_batch = gen(noise_batch)
                batch = torch.cat((fake_batch, real_batch))
            logits = disc(batch).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, target=disc_target)
            loss.backward()
            optim_d.step()
            optim_d.zero_grad()

        for noise_batch in itertools.islice(noise_loader, 1):
            fake_batch = gen(noise_batch)
            logits = disc(fake_batch).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, target=gen_target)
            loss.backward()
            optim_g.step()
            optim_g.zero_grad()

        # Some callbacks here


if __name__ == '__main__':
    EMBEDDING_SIZE = 256

    # stupid but fast-to-implement behavior: call train many times and sample image after each one
    import matplotlib.pyplot as plt
    from tqdm import trange
    disc = get_discriminator()
    gen = get_generator()

    real_dataset = get_CJK(size=32)
    noise_dataset = get_noise((EMBEDDING_SIZE,))

    for i in trange(100):
        with torch.no_grad():
            sample = gen(next(iter(noise_dataset))).numpy().clip(0, 1).squeeze()
            plt.imsave(f'generated/sample-{i:02d}.png', sample, cmap='gray')
        train(disc, gen, real_dataset, noise_dataset, n_iter=1)

