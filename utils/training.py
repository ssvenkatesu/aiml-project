from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _iterate_minibatches(X, A, M, Y, batch_size: int, shuffle: bool = True):
	N = X.shape[0]
	indices = torch.randperm(N) if shuffle else torch.arange(N)
	for start in range(0, N, batch_size):
		idx = indices[start : start + batch_size]
		xb, ab, mb = X[idx], A[idx], M[idx]
		yb = Y[idx] if Y is not None else None
		yield xb, ab, mb, yb


def pretrain_task_targets(M: torch.Tensor) -> torch.Tensor:
	# Predict graph size (number of nodes used)
	# M: [B, N]
	sizes = M.sum(dim=1, keepdim=True)  # [B,1]
	return sizes


def train_with_transfer_learning(
	model: nn.Module,
	train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
	val_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
	learning_rate: float,
	batch_size: int,
	pretrain_epochs: int,
	finetune_epochs: int,
	device: torch.device,
) -> Tuple[List[float], List[float]]:
	Xtr, Atr, Mtr, Ytr = train_data
	Xva, Ava, Mva, Yva = val_data

	Xtr = Xtr.to(device); Atr = Atr.to(device); Mtr = Mtr.to(device)
	Xva = Xva.to(device); Ava = Ava.to(device); Mva = Mva.to(device)
	if Ytr is not None: Ytr = Ytr.to(device)
	if Yva is not None: Yva = Yva.to(device)

	# PRETRAIN: predict graph size (self-supervised)
	pre_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	mse = nn.MSELoss()
	pre_losses: List[float] = []
	model.train()
	for epoch in range(pretrain_epochs):
		epoch_loss = 0.0
		for xb, ab, mb, _ in _iterate_minibatches(Xtr, Atr, Mtr, Ytr, batch_size, shuffle=True):
			pre_optimizer.zero_grad()
			preds = model(xb, ab, mb)  # [B,1]
			targets = pretrain_task_targets(mb)  # [B,1]
			loss = mse(preds, targets)
			loss.backward()
			pre_optimizer.step()
			epoch_loss += loss.item() * xb.size(0)
		pre_losses.append(epoch_loss / Xtr.size(0))

	# FINETUNE: predict bioactivity/effectiveness
	fin_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	fin_losses: List[float] = []
	model.train()
	for epoch in range(finetune_epochs):
		epoch_loss = 0.0
		for xb, ab, mb, yb in _iterate_minibatches(Xtr, Atr, Mtr, Ytr, batch_size, shuffle=True):
			fin_optimizer.zero_grad()
			preds = model(xb, ab, mb)
			loss = mse(preds, yb)
			loss.backward()
			fin_optimizer.step()
			epoch_loss += loss.item() * xb.size(0)
		fin_losses.append(epoch_loss / Xtr.size(0))

	return pre_losses, fin_losses


@torch.no_grad()
def evaluate_model(
	model: nn.Module,
	data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
	device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray]:
	X, A, M, Y = data
	X = X.to(device); A = A.to(device); M = M.to(device)
	model.eval()
	preds = model(X, A, M).cpu().numpy().reshape(-1)
	metrics = {"mse": float("nan"), "mae": float("nan")}
	if Y is not None:
		y_true = Y.cpu().numpy().reshape(-1)
		metrics = {
			"mse": float(mean_squared_error(y_true, preds)),
			"mae": float(mean_absolute_error(y_true, preds)),
		}
	return metrics, preds
