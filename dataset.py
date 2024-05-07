from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import IterableDataset
import tiktoken


# Iterable dataset for Tiny Stories
class TinyStoriesDataset(IterableDataset):
    def __init__(self, data_folder: Path, mode: str = "train", context_length: int = 2):

        if mode not in ["train", "test"]:
            raise ValueError("mode must be one of 'train', 'valid', or 'test'")

        self.data_path = data_folder / f"{mode}.bin"
        self.context_length = context_length
        # self.tokenizer = tiktoken.get_encoding("gpt2")

        self.data = self.load_data()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab

    def load_data(self) -> List[List[int]]:

        data = torch.from_numpy(
            np.memmap(self.data_path, dtype=np.uint16, mode="r").astype(np.int64)
        )
        return data

    def __iter__(self):
        while True:
            idx = torch.randint(len(self.data) - self.context_length, (1,)).item()
            yield self.data[idx : idx + self.context_length], self.data[
                idx + 1 : idx + self.context_length + 1
            ]

    def __len__(self):
        return len(self.data) - self.context_length
