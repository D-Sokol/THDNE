from torch.utils.data import Dataset


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
        # TODO: some transformation
        return symbol


def get_CJK() -> UnicodeRange:
    return UnicodeRange(0x4e00, 0x9fd6)

