import os
import torch
import torch.nn as nn

class CheckPointer():
  def __init__(self, model: nn.Module, chkpt_path: str):
    self.model = model

    self.chkpt_path= chkpt_path
    os.makedirs(self.chkpt_path, exist_ok=True)

    version=0
    while (os.path.isdir(self.chkpt_path + f'/version_{version}')):
      version += 1

    self.chkpt_path = self.chkpt_path + f'/version_{version}'
    os.mkdir(self.chkpt_path)

    self.checkpoint_count = 1

  def _get_chkpt_pathname(self, chkpt: int):
    return self.chkpt_path + f"/checkpoint_{chkpt}.pt"

  def on_chkpt(self) -> None:
    self.model.eval()
    chkpt_pathname = self._get_chkpt_pathname(self.checkpoint_count)
    torch.save(self.model.state_dict(), chkpt_pathname)
    self.checkpoint_count += 1

  def from_chkpt_path(self, chkpt_path: str) -> nn.Module:
    self.model.load_state_dict(torch.load(chkpt_path, weights_only=True))
    return self.model

  def from_chkpt(self, chkpt: int) -> nn.Module:
    chkpt_pathname = self._get_chkpt_pathname(chkpt)
    self.model.load_state_dict(torch.load(chkpt_pathname, weights_only=True))
    return self.model
