import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy


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



    
      

    

def n_params(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())



def get_model_device(model: nn.Module):
    return next(model.parameters()).device

def show_scores(metrics, scores):
  from tabulate import tabulate

  data = [[str(metric), score] for metric, score in zip(metrics, scores)]

  print(tabulate(data, headers=['metric', 'score'], tablefmt='fancy_grid'))


def show_train_loss(epoch_losses):
  data = []
  step = 1
  for i, losses in enumerate(epoch_losses):
    for loss in losses:
      data.append([i+1, loss])

  df = pd.DataFrame(data, columns=['epoch', 'loss'])

  sns.lineplot(df, x=df.index, y='loss', color='gray', linewidth=0.5, zorder=0, legend=False)
  sns.lineplot(df, x=df.index, y='loss', hue='epoch', linewidth=1.5, zorder=1, legend='full')

  plt.xlabel('step')
  plt.legend(title='epoch')
  plt.title('Loss during training')
  plt.show()

def get_batch(dataloader):
    return next(iter(dataloader))
