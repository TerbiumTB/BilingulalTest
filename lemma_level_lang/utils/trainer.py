import torch
import torch.nn as nn
from .logger import BaseLogger
from .func import get_model_device


class Trainer():
    def __init__(self, model: nn.Module, logger: BaseLogger, device = None, chkpter = None):
        self.model = model
        self.logger = logger
        self.chkpter = chkpter
        self.device = device or get_model_device(model)

    def _predict_batch(self, batch):
        features, target = batch[:-1], batch[-1]
        features = [feature.to(self.device) for feature in features]
        target = target.float().to(self.device)

        # opt.zero_grad()

        pred = self.model(*features)
        return pred, target

    @torch.inference_mode()
    def _predict_epoch(self, data_loader, transform=None):
        self.model.eval()

        preds = []
        targets = []
        for batch in data_loader:
            pred, target = self._predict_batch(batch)
            targets.append(target.detach().cpu())

            if transform is None:
                preds.append(pred.detach().cpu())
            else:
                preds.append(transform(pred).detach().cpu())

        return torch.cat(preds), torch.cat(targets)


    @torch.inference_mode()
    def _val_epoch(self, val_loader, metrics: dict[str, nn.Module], transform=None):
        preds, targets = self._predict_epoch(val_loader, transform)

        val_metrics = {name: m(preds, targets).item() for name, m in metrics.items()}
        self.logger.on_val_epoch(val_metrics)
        return val_metrics


    def _train_epoch(self, train_loader, criterion, opt):
        self.model.train()
        for step, batch in enumerate(train_loader):
            opt.zero_grad()

            pred, target = self._predict_batch(batch)

            loss = criterion(pred, target)

            loss.backward()

            opt.step()

            self.logger.on_epoch_step(loss.item())

        return self.model


    def fit(self, train_loader, train_params,
            val_loader=None, val_params=None,
            epochs=4):
        self.logger.on_train_start(len(train_loader))

        for epoch in range(epochs):
            self.logger.on_epoch_start()

            self.model = self._train_epoch(train_loader, **train_params)

            if self.chkpter is not None:
                self.chkpter.on_chkpt()

            if val_loader is not None:
                self._val_epoch(val_loader, **val_params)



            self.logger.on_epoch_end()

        return self.model

    @torch.inference_mode()
    def predict(self, data_loader, transform=None):
        preds, _ = self._predict_epoch(data_loader, transform)

        return preds

    @torch.inference_mode()
    def test(self, test_loader, metrics: dict[str, nn.Module], transform=None):
        preds, targets = self._predict_epoch(test_loader, transform)

        test_metrics = {name: m(preds, targets).item() for name, m in metrics.items()}
        return test_metrics

    def load(self, chkpt_num: int|None = None, ckpt_path: str|None = None, ):
        '''loads model from checkpoint. checkpoint is either a number or a full path'''
        if chkpt_num is not None:
            self.model = self.chkpter.from_chkpt(chkpt_num)
        elif ckpt_path is not None:
            self.model = self.chkpter.from_chkpt_path(ckpt_path)

        return self.model



class LTrainer(Trainer):
    def _predict_batch(self, batch):    
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        # level_idx = batch["level_idx"].to(self.device)
        # lang_idx = batch["lang_idx"].to(self.device)

        pred_ids = self.model(input_ids, attention_mask)

        return pred_ids, target_ids
    
class LLTrainer(Trainer):
    def _predict_batch(self, batch):    
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        level_idx = batch["level_idx"].to(self.device)
        # lang_idx = batch["lang_idx"].to(self.device)

        pred_ids = self.model(input_ids, level_idx, attention_mask)

        return pred_ids, target_ids
    
class LLLTrainer(Trainer):
    def _predict_batch(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        level_idx = batch["level_idx"].to(self.device)
        lang_idx = batch["lang_idx"].to(self.device)

        pred_ids = self.model(input_ids, level_idx, lang_idx, attention_mask)

        return pred_ids, target_ids


class ClassifierTrainer(Trainer):
    def _predict_batch(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        level_idx = batch["level_idx"].to(self.device)

        logits = self.model(input_ids, attention_mask)

        return logits, level_idx
