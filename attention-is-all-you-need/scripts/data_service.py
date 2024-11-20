from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchtext
from torchtext.vocab import build_vocab_from_iterator
import torch
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from collections import Counter
from typing import List, Tuple


class DataService:
    def __init__(self, src_language: str = "de", tgt_language: str = "en", batch_size: int = 32):
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.batch_size = batch_size

        self._src_tokenizer = get_tokenizer(
            "spacy", language="de_core_news_sm")
        self._tgt_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

        self.train_iter, self.valid_iter = self._collect_data()

        self.src_vocab, self.tgt_vocab, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = self._prepare_vocab()

        self.train_loader, self.valid_loader = self._prepare_data_loaders()

    def _collect_data(self) -> Tuple:
        """Load training and validation datasets."""
        train_iter = Multi30k(split="train", language_pair=(
            self.src_language, self.tgt_language))
        valid_iter = Multi30k(split="valid", language_pair=(
            self.src_language, self.tgt_language))
        return train_iter, valid_iter

    def _build_vocab(self, data_iter: List[str], tokenizer) -> torchtext.vocab.Vocab:
        """Build a vocabulary from the data iterator."""
        counter = Counter()
        for text in data_iter:
            counter.update(tokenizer(text))
        return build_vocab_from_iterator([counter.keys()], specials=["<unk>", "<pad>", "<bos>", "<eos>"])

    def _prepare_vocab(self) -> Tuple[torchtext.vocab.Vocab, torchtext.vocab.Vocab, int, int, int]:
        """Prepare vocabularies for source and target languages."""
        src_vocab = self._build_vocab(
            (src for src, _ in self.train_iter), self._src_tokenizer)
        tgt_vocab = self._build_vocab(
            (tgt for _, tgt in self.train_iter), self._tgt_tokenizer)

        # Special token indices
        src_vocab.set_default_index(src_vocab["<unk>"])
        tgt_vocab.set_default_index(tgt_vocab["<unk>"])
        pad_idx = src_vocab["<pad>"]
        bos_idx = src_vocab["<bos>"]
        eos_idx = src_vocab["<eos>"]

        return src_vocab, tgt_vocab, pad_idx, bos_idx, eos_idx

    def _data_process(self, raw_data_iter) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Process raw data into tensors."""
        data = []
        for src, tgt in raw_data_iter:
            src_tensor = torch.tensor(
                [self.src_vocab[token] for token in self._src_tokenizer(src)], dtype=torch.long)
            tgt_tensor = torch.tensor(
                [self.tgt_vocab[token] for token in self._tgt_tokenizer(tgt)], dtype=torch.long)
            data.append((
                torch.cat([torch.tensor([self.BOS_IDX]), src_tensor,
                          torch.tensor([self.EOS_IDX])]),
                torch.cat([torch.tensor([self.BOS_IDX]),
                          tgt_tensor, torch.tensor([self.EOS_IDX])])
            ))
        return data

    def _collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate function for batching."""
        src_batch, tgt_batch = zip(*batch)
        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD_IDX)
        return src_batch, tgt_batch

    def _prepare_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare DataLoaders for training and validation data."""
        train_data = self._data_process(self.train_iter)
        valid_data = self._data_process(self.valid_iter)

        train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn)
        valid_loader = DataLoader(
            valid_data, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn)

        return train_loader, valid_loader

    def get_train_loader(self) -> DataLoader:
        """Public method to access the training data loader."""
        return self.train_loader

    def get_valid_loader(self) -> DataLoader:
        """Public method to access the validation data loader."""
        return self.valid_loader

    def get_vocabularies(self) -> Tuple[torchtext.vocab.Vocab, torchtext.vocab.Vocab]:
        """Public method to access vocabularies."""
        return self.src_vocab, self.tgt_vocab
