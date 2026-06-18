from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy
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
