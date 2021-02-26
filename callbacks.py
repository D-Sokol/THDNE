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

