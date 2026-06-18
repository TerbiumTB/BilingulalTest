import math
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy
import torch
import torch.nn as nn

class TokenCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index


    def forward(self, preds, target_ids):
        return cross_entropy(preds.flatten(end_dim=-2),
                             target_ids.flatten(),
                             ignore_index=self.ignore_index,)

class TokenAccuracy(nn.Module):
    def __init__(self, num_labels, ignore_index, top_k=3):
        super().__init__()
        self.accuracy_score = Accuracy(task="multiclass",
                                       num_classes=num_labels,
                                       ignore_index=ignore_index,
                                       top_k=top_k)

    def forward(self, preds, target_ids):
       return self.accuracy_score(preds.flatten(end_dim=-2), target_ids.flatten())


class Perplexity(nn.Module):
    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, preds, target_ids):
        ce = cross_entropy(preds.flatten(end_dim=-2),
                           target_ids.flatten(),
                           ignore_index=self.ignore_index)
        return torch.exp(ce)


class BitsPerChar(nn.Module):
    def __init__(self, tokenizer, ignore_index):
        super().__init__()
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def forward(self, preds, target_ids):
        ce_sum = cross_entropy(preds.flatten(end_dim=-2),
                               target_ids.flatten(),
                               ignore_index=self.ignore_index,
                               reduction="sum")

        n_chars = 0
        for seq in target_ids:
            text = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
            n_chars += len(text.replace(" ", ""))

        if n_chars == 0:
            return torch.tensor(float("inf"))
        return ce_sum / math.log(2) / n_chars


class SequenceExactMatch(nn.Module):
    def __init__(self, pad_idx, eos_idx=None):
        super().__init__()
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx

    def forward(self, preds, target_ids):
        pred_ids = preds.argmax(dim=-1)
        total = pred_ids.size(0)
        if total == 0:
            return torch.tensor(0.0)

        matches = 0
        for p, t in zip(pred_ids, target_ids):
            if torch.equal(self._trim(p), self._trim(t)):
                matches += 1
        return torch.tensor(matches / total)

    def _trim(self, seq):
        seq = seq[seq != self.pad_idx]
        if self.eos_idx is not None:
            eos_positions = (seq == self.eos_idx).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                seq = seq[:eos_positions[0]]
        return seq
