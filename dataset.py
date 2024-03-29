import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, IterableDataset


def render_char(char: str, size=128, font_path="SimSun.ttf") -> Image.Image:
    img = Image.new("L", (size, size), 'white')
    drawer = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size)
    drawer.text((0, 0), char, font=font)
    return img


class UnicodeRange(Dataset):
    def __init__(self, begin, end, size=128):
        super().__init__()
        assert end >= begin
        self._begin = begin
        self._end = end
        self._size = size

    def __len__(self):
        return self._end - self._begin

    def __getitem__(self, ix):
        if not 0 <= ix < self._end:
            raise IndexError
        symbol = chr(self._begin + ix)
        image = np.array(render_char(symbol, size=self._size), dtype=np.float32)
        image /= 255.
        # Returns one-channel image of shape (1, size, size)
        return torch.as_tensor(image)[None]


class GaussNoise(IterableDataset):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __iter__(self):
        return self

    def __next__(self):
        return torch.randn(*self.shape)


def get_CJK(size) -> UnicodeRange:
    return UnicodeRange(0x4e00, 0x9fd6, size=size)

def get_noise(shape) -> GaussNoise:
    return GaussNoise(shape)

