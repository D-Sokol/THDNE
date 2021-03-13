import itertools
import matplotlib.pyplot as plt
import pathlib
import torch
from torch.utils.data import DataLoader

def _filename(directory, label=None, for_disc: bool=True):
    path = pathlib.Path(directory)
    assert path.is_dir()
    filename = "{}{}.pth".format(
        "discriminator" if for_disc else "generator",
        f"-{label}" if label is not None else ""
    )
    path /= filename
    return str(path)


def save_models(disc, gen, directory=pathlib.Path(), label=None):
    torch.save(disc.state_dict(), _filename(directory, label, True))
    torch.save(gen.state_dict(), _filename(directory, label, False))


# TODO: refactor: swap first 2 arguments and add '=None'
def load_models(disc, gen, directory=pathlib.Path(), label=None):
    if disc is not None:
        disc.load_state_dict(torch.load(_filename(directory, label, True)))
    gen.load_state_dict(torch.load(_filename(directory, label, False)))


def _generate_sample_name(directory, name, index=None, iteration=None, ext='png'):
    return str(directory / "{name}{iteration_part}{index_part}.{ext}".format(
        name=name,
        iteration_part=-iteration if iteration is not None else "",
        index_part=-index if index is not None else "",
        ext=ext
    ))


def sample_images(gen, noise_dataset, count=1, iteration=None,
                  directory=pathlib.Path(), name='sample', ext='png'):
    directory = pathlib.Path(directory)
    assert directory.is_dir()

    with torch.no_grad():
        noise_it = itertools.islice(noise_dataset, count)
        noise = torch.stack(tuple(noise_it))
        image_stack = gen(noise).numpy().clip(0, 1).squeeze(1)

    for i, image in enumerate(image_stack, 1):
        filename = _generate_sample_name(directory, name, index=i, iteration=iteration, ext=ext)
        plt.imsave(filename, image, cmap='gray')


def render_histograms(disc, gen, real_dataset, noise_dataset, iteration, count=512,
                      directory=pathlib.Path()):
    directory = pathlib.Path(directory)
    assert directory.is_dir()

    noise_loader = DataLoader(noise_dataset, batch_size=count)
    data_loader = DataLoader(real_dataset, batch_size=count)

    real_batch = next(iter(data_loader))
    fake_batch = gen(next(iter(noise_loader)))

    with torch.no_grad():
        probas_real_is_real = torch.sigmoid(disc(real_batch).squeeze(-1)).cpu().numpy()
        probas_fake_is_real = torch.sigmoid(disc(fake_batch).squeeze(-1)).cpu().numpy()

    plt.subplot(1, 2, 1)
    plt.hist(probas_real_is_real, bins=count//20, range=(0, 1))
    plt.xlabel('Estimated probability')
    plt.title('Real samples')

    plt.subplot(1, 2, 2)
    plt.hist(probas_fake_is_real, bins=count//20, range=(0, 1))
    plt.xlabel('Estimated probability')
    plt.title('Fake samples')

    plt.savefig(_generate_sample_name(directory, name='histograms', iteration=iteration))
    plt.clf()

