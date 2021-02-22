import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset


def render_char(char: str, size=128, font_path="SimSun.ttf") -> Image.Image:
    img = Image.new("L", (size, size), 'white')
    drawer = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size)
    drawer.text((0, 0), char, font=font)
    return img


class UnicodeRange(Dataset):
    def __init__(self, begin, end):
        assert end >= begin
        self._begin = begin
        self._end = end

    def __len__(self):
        return self._end - self._begin

    def __getitem__(self, ix):
        if not 0 <= ix < self._end:
            raise IndexError
        symbol = chr(self._begin + ix)
        image = np.array(render_char(symbol), dtype=np.float32)
        image /= 255.
        return torch.as_tensor(image)


def get_CJK() -> UnicodeRange:
    return UnicodeRange(0x4e00, 0x9fd6)

