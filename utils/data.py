from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch


ALPHABET = list("CNOFPSclBrI[]=#()")  # toy set of tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def build_tokenizer(smiles_list: List[str]) -> Dict[str, int]:
	tokens = {PAD_TOKEN: 0, UNK_TOKEN: 1}
	idx = 2
	for s in smiles_list:
		for ch in s:
			if ch not in tokens:
				tokens[ch] = idx
				idx += 1
	# ensure alphabet present
	for ch in ALPHABET:
		if ch not in tokens:
			tokens[ch] = idx
			idx += 1
	return tokens


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Ensure common column names regardless of capitalization or synonyms."""
	renamed = {}
	for col in df.columns:
		lower = col.strip().lower()
		if lower == "smiles":
			renamed[col] = "smiles"
		elif lower in {"label", "target", "bioactivity", "toxicity"}:
			renamed[col] = "label"
		elif lower in {"compound", "compound_id", "molecule", "molid", "name"}:
			renamed[col] = "compound"
	if renamed:
		df = df.rename(columns=renamed)
	return df


def tokenize(smiles: str, tokenizer: Dict[str, int], max_len: int) -> List[int]:
	ids = [tokenizer.get(ch, tokenizer[UNK_TOKEN]) for ch in smiles[:max_len]]
	if len(ids) < max_len:
		ids = ids + [tokenizer[PAD_TOKEN]] * (max_len - len(ids))
	return ids


def ids_to_onehot(ids: List[int], vocab_size: int) -> np.ndarray:
	arr = np.zeros((len(ids), vocab_size), dtype=np.float32)
	for i, t in enumerate(ids):
		arr[i, t] = 1.0
	return arr


def linear_chain_adjacency(length: int, max_len: int) -> np.ndarray:
	A = np.zeros((max_len, max_len), dtype=np.float32)
	for i in range(length - 1):
		A[i, i + 1] = 1.0
		A[i + 1, i] = 1.0
	# self loops
	for i in range(length):
		A[i, i] = 1.0
	return A


def generate_dummy_smiles_dataset(
	n_samples: int,
	max_len: int,
	base_df: Optional[pd.DataFrame] = None,
	seed: int = 42,
) -> pd.DataFrame:
	rng = np.random.default_rng(seed)
	if base_df is None:
		smiles = []
		compounds = []
		for idx in range(n_samples):
			length = int(rng.integers(low=max_len // 2, high=max_len))
			chars = rng.choice(ALPHABET, size=length, replace=True)
			s = "".join(chars)
			smiles.append(s)
			compounds.append(f"Mol{idx + 1:03d}")
		labels = _synthetic_labels(smiles, rng)
		return pd.DataFrame({"compound": compounds, "smiles": smiles, "label": labels})

	# When base dataframe provided, standardise columns and honour provided labels if present.
	df = standardize_columns(base_df.copy())
	if "smiles" not in df.columns:
		raise ValueError("Expected a column named 'smiles' (case-insensitive) in the dataset.")
	if len(df) > n_samples:
		df = df.sample(n_samples, random_state=seed).reset_index(drop=True)
	elif len(df) < n_samples:
		extra = generate_dummy_smiles_dataset(n_samples - len(df), max_len, base_df=None, seed=seed + 1)
		df = pd.concat([df.reset_index(drop=True), extra], ignore_index=True)
	if "compound" not in df.columns:
		df["compound"] = [f"Mol{idx + 1:03d}" for idx in range(len(df))]
	df["smiles"] = df["smiles"].astype(str)
	if "label" not in df.columns:
		df["label"] = _synthetic_labels(df["smiles"].tolist(), rng)
	else:
		df["label"] = pd.to_numeric(df["label"], errors="coerce")
		if df["label"].isnull().any():
			df.loc[df["label"].isnull(), "label"] = _synthetic_labels(
				df.loc[df["label"].isnull(), "smiles"].tolist(), rng
			)
	return df.reset_index(drop=True)


def _synthetic_labels(smiles: List[str], rng: np.random.Generator) -> List[float]:
	labels = []
	for s in smiles:
		c = s.count("C")
		n = s.count("N")
		o = s.count("O")
		aromatic = s.count("=") + s.count("#")
		length = len(s)
		y = 0.8 * c + 1.2 * n + 0.5 * o + 0.3 * aromatic + 0.02 * length
		y += rng.normal(0, 0.5)
		labels.append(float(y))
	return labels


def smiles_to_graph_batch(
	smiles_list: List[str],
	labels: Optional[np.ndarray],
	max_len: int,
	tokenizer: Optional[Dict[str, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, int]]:
	if tokenizer is None:
		tokenizer = build_tokenizer(smiles_list)
	vocab_size = len(tokenizer)

	X = []
	A = []
	M = []
	for s in smiles_list:
		ids = tokenize(s, tokenizer, max_len)
		onehot = ids_to_onehot(ids, vocab_size)
		length = min(len(s), max_len)
		adj = linear_chain_adjacency(length, max_len)
		mask = np.zeros((max_len,), dtype=np.float32)
		mask[:length] = 1.0

		X.append(onehot)
		A.append(adj)
		M.append(mask)

	X = torch.tensor(np.stack(X), dtype=torch.float32)
	A = torch.tensor(np.stack(A), dtype=torch.float32)
	M = torch.tensor(np.stack(M), dtype=torch.float32)

	Y = None
	if labels is not None:
		Y = torch.tensor(labels.reshape(-1, 1), dtype=torch.float32)

	return X, A, M, Y, tokenizer


def split_dataset(df: pd.DataFrame, train_split: float, val_split: float, seed: int = 42):
	assert 0 < train_split < 1
	assert 0 < val_split < 1
	df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
	n = len(df)
	n_train = int(n * train_split)
	remaining = n - n_train
	n_val = int(remaining * val_split)
	train_df = df.iloc[:n_train]
	val_df = df.iloc[n_train : n_train + n_val]
	test_df = df.iloc[n_train + n_val :]
	return train_df, val_df, test_df
