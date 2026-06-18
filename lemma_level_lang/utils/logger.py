import os
import numpy as np
import torch
import torch.nn as nn
# from tqdm.notebook import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from abc import ABC, abstractmethod

class BaseLogger(ABC):
	def __init__(self, log_every_n_step=100):
		...

	@abstractmethod
	def on_train_start(self, total):
		...

	@abstractmethod
	def on_epoch_step(self, batch_loss):
		...

	@abstractmethod
	def on_epoch_end(self):
		...

	@abstractmethod
	def on_val_epoch(self, val_metric):
		...

	@abstractmethod
	def on_epoch_start(self):
		...

	@abstractmethod
	def get_best_val_score(self):
		...

class TQDMLogger(BaseLogger):
	def __init__(self, log_every_n_step=100):
		self.log = None
		self.total = None
		self.log_every_n_step = log_every_n_step

		self.epoch_losses = []
		self._batch_losses = []
		self._n_batch_losses = []
		self.val_metrics = []
		self._postfix = {}

	def on_train_start(self, total):
		self.epoch_losses = []
		self._batch_losses = []
		self._n_batch_losses = []
		self.val_metrics = []
		self.total = total

	def on_epoch_step(self, batch_loss):
		self._batch_losses.append(batch_loss)

		if  len(self._batch_losses) % self.log_every_n_step == self.log_every_n_step - 1:
			self._n_batch_losses.append(np.mean(self._batch_losses[-self.log_every_n_step:]))
			self.log.update(self.log_every_n_step)
			self._postfix['average batch loss'] = self._n_batch_losses[-1]
			self.log.set_postfix(self._postfix)

	def on_epoch_end(self):
		self._n_batch_losses.append(np.mean(self._batch_losses[-self.log_every_n_step:]))
		self.log.update(self.total % self.log_every_n_step)
		self._postfix['average batch loss'] = self._n_batch_losses[-1]
		self.log.set_postfix(self._postfix)
		self.log.close()

		self.epoch_losses.append(self._n_batch_losses)

	def on_val_epoch(self, val_metric):
		self._postfix['validation metric'] = val_metric

	def on_epoch_start(self):
		self.log = tqdm(unit="batch",
					total=self.total,
					unit_scale=1,
			desc=f"EPOCH {len(self.epoch_losses) + 1}",
					miniters=10
		)
		self._batch_losses = []
		self._n_batch_losses = []
		self._postfix = {}

	def get_best_val_score(self):
		best_epoch = np.argmin(self.val_metrics)
		return best_epoch, self.val_metrics[best_epoch]
  


class PlotLogger(BaseLogger):
	def __init__(self, log_every_n_step=100, figsize=(12, 4)):
		self.log_every_n_step = log_every_n_step
		self.figsize = figsize

		self.epoch_losses = []
		self.val_metrics = []
		self._batch_losses = []
		self._n_batch_losses = []
		self._total = None

	def on_train_start(self, total):
		self.epoch_losses = []
		self._batch_losses = []
		self._n_batch_losses = []
		self.val_metrics = []
		self._total = total

	def on_epoch_start(self):
		self._batch_losses = []
		self._n_batch_losses = []

	def on_epoch_step(self, batch_loss):
		self._batch_losses.append(batch_loss)

		if len(self._batch_losses) % self.log_every_n_step == self.log_every_n_step - 1:
			self._n_batch_losses.append(
				np.mean(self._batch_losses[-self.log_every_n_step:])
			)
			self._redraw()

	def on_epoch_end(self):
		remainder = len(self._batch_losses) % self.log_every_n_step
		if remainder:
			self._n_batch_losses.append(np.mean(self._batch_losses[-remainder:]))

		self.epoch_losses.append(self._n_batch_losses)
		self._redraw()

	def on_val_epoch(self, val_metric):
		self.val_metrics.append(val_metric)

	def _build_series(self):
		all_points = []
		for ep in self.epoch_losses:
			all_points.extend(ep)
		all_points.extend(self._n_batch_losses)
		return all_points

	def _epoch_boundaries(self):
		boundaries = []
		cursor = 0
		for ep in self.epoch_losses:
			boundaries.append(cursor)
			cursor += len(ep)
		if self._n_batch_losses:
			boundaries.append(cursor)
		return boundaries
	
	def draw_val(self, ax_val):
		# ax_val = axes[1]
		val_xs = list(range(1, len(self.val_metrics) + 1))
		ax_val.plot(val_xs, self.val_metrics, color="#D85A30", linewidth=1.5,
					marker="o", markersize=4, label="val metric")

		vmin = min(self.val_metrics)
		vmax = max(self.val_metrics)
		vmargin = (vmax - vmin) * 0.1 or 0.1
		ax_val.set_ylim(vmin - vmargin, vmax + vmargin)

		# ax_val.set_xticks(val_xs)
		# ax_val.set_xticklabels(val_xs, fontsize=9)
		# ax_val.set
		ax_val.set_xlabel("epoch")
		ax_val.set_ylabel("val metric")
		ax_val.set_title("Validation metric")
		# ax_val.legend(fontsize=9, loc="upper right")
		ax_val.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.4)

	def _build_fig(self):
		series = self._build_series()
		if not series:
			return

		has_val = len(self.val_metrics) > 0

		if has_val:
			fig, (ax, val_ax) = plt.subplots(
				1, 2,
				figsize=self.figsize,
				gridspec_kw={"width_ratios": [3, 1]}
			)
		else: 
			fig, ax = plt.subplots(
				1, 1,
				figsize=self.figsize,
				# gridspec_kw={"width_ratios": [3, 1]}

			)


		xs = list(range(len(series)))
		ax.plot(xs, series, color="#378ADD", linewidth=1.5, label="avg batch loss")
		ax.fill_between(xs, series, alpha=0.07, color="#378ADD")

		ymin = min(series)
		ymax = max(series)
		margin = (ymax - ymin) * 0.1 or 0.1
		ax.set_ylim(ymin - margin, ymax + margin)

		boundaries = self._epoch_boundaries()
		epoch_positions = [b for b in boundaries if b < len(series)]

		ax.set_xticks(epoch_positions, minor=False)
		ax.set_xticklabels(
			[i for i in range(len(epoch_positions))],
			minor=False,
			fontsize=9,
		)

		for pos in epoch_positions[1:]:
			ax.axvline(pos, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

		ax.set_xticks(xs, minor=True)
		ax.tick_params(axis="x", which="minor", length=3, color="gray", labelsize=0)
		ax.tick_params(axis="x", which="major", length=7)

		ax.set_xlabel("log step (epochs)")
		ax.set_ylabel("avg batch loss")
		ax.set_title("Training loss")
		# ax.legend(fontsize=9, loc="upper right")
		ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.4)
		ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.25)

		if has_val:
			self.draw_val(val_ax)

		fig.tight_layout()

		return fig

	def _redraw(self):
		fig = self._build_fig()
		if fig is None:
			return

		clear_output(wait=True)
		plt.show()
		plt.close(fig)

	def get_best_val_score(self):
		if not self.val_metrics:
			return None, None
		best_epoch = int(np.argmin(self.val_metrics))
		return best_epoch, self.val_metrics[best_epoch]

	def save(self, path: str):
		fig = self._build_fig()
		if fig is None:
			return

		fig.savefig(os.path.join(path, "logplot.png"), dpi=150, bbox_inches="tight")
		plt.close(fig)
