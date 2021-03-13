#!/usr/bin/env python3

import itertools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import get_CJK, get_noise
from models import get_discriminator, get_generator
from parser import get_parser, check_sanity
from callbacks import *


def _get_range(*args, verbose=False):
    return (trange if verbose else range)(*args)


def train(disc, gen, real_dataset, noise_dataset,
          n_iter=100, k=5, batch_size=100, verbose=False,
          histogram_dir=None, histogram_freq=1,
          images=1, sample_dir=None, sample_freq=1):
    optim_d = torch.optim.Adam(disc.parameters())
    optim_g = torch.optim.Adam(gen.parameters())
    noise_loader = DataLoader(noise_dataset, batch_size=batch_size)
    data_loader = DataLoader(real_dataset, batch_size=batch_size)

    disc_target = torch.cat((torch.zeros(batch_size), torch.ones(batch_size)))
    gen_target = torch.ones(batch_size)

    for i in _get_range(n_iter, verbose=verbose):
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

        if histogram_dir is not None and i % histogram_freq == 0:
            render_histograms(disc, gen, real_dataset, noise_dataset, iteration=i+1, directory=histogram_dir)

        if sample_dir is not None and i % sample_freq == 0:
            sample_images(gen, noise_dataset, count=images, iteration=i+1, directory=sample_dir)


if __name__ == '__main__':
    EMBEDDING_SIZE = 256

    parser = get_parser()
    config = parser.parse_args()

    if not config.force_mode:
        problems = check_sanity(config)
        if problems:
            print("Discrepancies in command line arguments detected:")
            for problem in problems:
                print(" *", problem)
            print("Use flag --force if you are really want to run script with these parameters.")
            exit(2)

    if config.mode == 'train':
        disc = get_discriminator()
        gen = get_generator()
        real_dataset = get_CJK(size=32)               # TODO: adjust this
        noise_dataset = get_noise((EMBEDDING_SIZE,))  # TODO (optional): read this parameter from config
        if config.load_dir is not None:
            load_models(disc, gen, directory=config.load_dir, label=config.load_label)

        train(disc, gen, real_dataset, noise_dataset,
              n_iter=config.iter, k=config.disc_steps, batch_size=config.batch, verbose=config.verbose,
              histogram_dir=config.histogram_dir, histogram_freq=config.histogram_freq,
              images=config.images, sample_dir=config.sample_dir, sample_freq=config.sample_freq)

        if config.save_dir is not None:
            save_models(disc, gen, directory=config.save_dir, label=config.save_label)

    elif config.mode == 'sampling':
        gen = get_generator()
        noise_dataset = get_noise((EMBEDDING_SIZE,))
        if config.load_dir is not None:
            load_models(None, gen, directory=config.load_dir, label=config.load_label)
        sample_images(gen, noise_dataset, count=config.images, directory=config.sample_dir)

