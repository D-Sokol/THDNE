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


def _generate_sample_name(directory, name, count, total_count=1, ext='png'):
    return str(directory / f"{name}{-count if total_count != 1 else ''}.{ext}")


def sample_images(gen, noise_dataset, count=1,
                  directory=pathlib.Path(), name='sample', ext='png'):
    directory = pathlib.Path(directory)
    assert directory.is_dir()

    with torch.no_grad():
        noise_it = itertools.islice(noise_dataset, count)
        noise = torch.stack(tuple(noise_it))
        image_stack = gen(noise).numpy().clip(0, 1).squeeze(1)

    for i, image in enumerate(image_stack, 1):
        filename = _generate_sample_name(directory, name, i, count, ext)
        plt.imsave(filename, image, cmap='gray')


def render_histograms(disc, gen, real_dataset, noise_dataset, count=512,
                      directory=pathlib.Path()):
    noise_loader = DataLoader(noise_dataset, batch_size=count)
    data_loader = DataLoader(real_dataset, batch_size=count)

    real_batch = next(iter(data_loader)).unsqueeze(1)
    fake_batch = gen(next(iter(noise_loader)))

    with torch.no_grad():
        probas_real_is_real = torch.sigmoid(disc(real_batch).squeeze(-1))
        probas_fake_is_real = torch.sigmoid(disc(fake_batch).squeeze(-1))

    plt.subplot(1, 2, 1)
    plt.hist(probas_real_is_real)
    plt.xlabel('Estimated probability')
    plt.title('Real samples')

    plt.subplot(1, 2, 2)
    plt.hist(probas_fake_is_real)
    plt.xlabel('Estimated probability')
    plt.title('Real samples')

    plt.savefig(_generate_sample_name(pathlib.Path(directory), 'histograms', 1, 1))
    plt.clf()

