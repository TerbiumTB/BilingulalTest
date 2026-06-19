"""Char-level CEFR classifier. Predicts a level for any word, including
generated (non-dictionary) ones, by learning character-pattern signals."""

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset

from dataset import LEVELS, level2idx
from .func import get_model_device


class WordComplexityClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 dim_feedforward, num_levels, max_positions, pad_idx, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_positions, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, num_levels)

        self.pad_idx = pad_idx

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        tok_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        embed = tok_emb + pos_emb

        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        out = self.transformer_encoder(embed, src_key_padding_mask=src_key_padding_mask)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = out.mean(dim=1)

        return self.head(pooled)


class ComplexityDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lemma, level = self.data.iloc[idx]
        lemma = self.tokenizer.bos_token + lemma + self.tokenizer.eos_token

        encoded = self.tokenizer(
            lemma,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "level_idx": level2idx[str(level)],
        }


def _encode_words(words, tokenizer, max_len, device):
    bos, eos = tokenizer.bos_token, tokenizer.eos_token
    full = [bos + w + eos for w in words]
    encoded = tokenizer(
        full,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


@torch.no_grad()
def predict_level(model, word, tokenizer, device=None, max_len=32):
    model.eval()
    device = device or get_model_device(model)
    input_ids, attention_mask = _encode_words([word], tokenizer, max_len, device)
    logits = model(input_ids, attention_mask)
    return LEVELS[int(logits.argmax(dim=-1).item())]


@torch.no_grad()
def predict_levels(model, words, tokenizer, device=None, max_len=32):
    model.eval()
    device = device or get_model_device(model)
    if not words:
        return []
    input_ids, attention_mask = _encode_words(words, tokenizer, max_len, device)
    logits = model(input_ids, attention_mask)
    idxs = logits.argmax(dim=-1).cpu().tolist()
    return [LEVELS[i] for i in idxs]


@torch.no_grad()
def predict_proba(model, words, tokenizer, device=None, max_len=32):
    """Returns full softmax over levels: shape [len(words), num_levels]."""
    model.eval()
    device = device or get_model_device(model)
    if not words:
        return torch.empty(0, len(LEVELS))
    input_ids, attention_mask = _encode_words(words, tokenizer, max_len, device)
    logits = model(input_ids, attention_mask)
    return torch.softmax(logits, dim=-1).cpu()
