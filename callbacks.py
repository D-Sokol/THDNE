import itertools
import matplotlib.pyplot as plt
import pathlib
import torch


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


def load_models(disc, gen, directory=pathlib.Path(), label=None):
    disc.load_state_dict(torch.load(_filename(directory, label, True)))
    gen.load_state_dict(torch.load(_filename(directory, label, False)))


def _generate_sample_name(directory, name, count, total_count, ext):
    return str(directory / f"{name}{-count if total_count != 1 else ''}.{ext}")


def sample_images(gen, noise_dataset, count=1,
                  directory=pathlib.Path(), name='sample', ext='png'):
    directory = pathlib.Path(directory)
    assert directory.is_dir()

    with torch.no_grad():
        noise_it = itertools.islice(noise_dataset, count)
        noise = torch.stack(tuple(noise_it))
        image_stack = gen(noise).numpy().clip(0, 1).squeeze(1)

    for i, image in enumerate(image_stack):
        filename = _generate_sample_name(directory, name, i, count, ext)
        plt.imsave(filename, image, cmap='gray')

