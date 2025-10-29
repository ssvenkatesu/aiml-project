import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
	def __init__(self, in_dim: int, out_dim: int):
		super().__init__()
		self.linear_neighbor = nn.Linear(in_dim, out_dim)
		self.linear_self = nn.Linear(in_dim, out_dim)
		self.bn = nn.BatchNorm1d(out_dim)

	def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		# x: [B, N, F], adj: [B, N, N], mask: [B, N]
		neigh = torch.matmul(adj, x)  # [B, N, F]
		h = self.linear_neighbor(neigh) + self.linear_self(x)  # [B, N, out]
		h = F.relu(h)
		# batch norm over nodes (flatten B*N)
		B, N, C = h.shape
		h = h.reshape(B * N, C)
		h = self.bn(h)
		h = h.reshape(B, N, C)
		# zero-out padding nodes
		h = h * mask.unsqueeze(-1)
		return h


class GlobalReadout(nn.Module):
	def __init__(self, mode: str = "mean"):
		super().__init__()
		assert mode in {"mean", "sum", "max"}
		self.mode = mode

	def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		# x: [B, N, F], mask: [B, N]
		if self.mode == "mean":
			sum_pool = (x * mask.unsqueeze(-1)).sum(dim=1)
			counts = mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
			return sum_pool / counts
		if self.mode == "sum":
			return (x * mask.unsqueeze(-1)).sum(dim=1)
		# max
		masked = x + (mask.unsqueeze(-1) - 1.0) * 1e9  # push padded to -inf
		return masked.max(dim=1).values


class SimpleGraphRegressor(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3, readout: str = "mean"):
		super().__init__()
		layers = []
		dims = [input_dim] + [hidden_dim] * num_layers
		for i in range(num_layers):
			layers.append(GraphConvLayer(dims[i], dims[i + 1]))
		self.layers = nn.ModuleList(layers)
		self.readout = GlobalReadout(readout)
		self.mlp = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
		)

	def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		h = x
		for layer in self.layers:
			h = layer(h, adj, mask)
		g = self.readout(h, mask)
		out = self.mlp(g)
		return out
