import io
import json
import time
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path

# Local imports
from models.gnn import SimpleGraphRegressor
from utils.data import (
    generate_dummy_smiles_dataset,
    smiles_to_graph_batch,
    split_dataset,
    standardize_columns,
)
from utils.training import train_with_transfer_learning, evaluate_model


st.set_page_config(page_title="AI-Assisted Drug Discovery (Demo)", layout="wide")
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "dummy_smiles.csv"


@st.cache_resource(show_spinner=False)
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_figure(loss_history_pre: list, loss_history_fine: list):
    fig, ax = plt.subplots(figsize=(6, 3))
    if loss_history_pre:
        ax.plot(loss_history_pre, label="pretrain loss")
    if loss_history_fine:
        ax.plot(loss_history_fine, label="finetune loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _build_token_freq(smiles_list: list) -> dict:
    freq = {}
    total = 0
    for s in smiles_list:
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1
            total += 1
    # smooth to avoid zero division
    for k in list(freq.keys()):
        freq[k] = freq[k] / max(total, 1)
    if total == 0:
        # fallback uniform
        return {"<any>": 1.0}
    return freq


def _novelty_score(smiles: str, token_freq: dict) -> float:
    if not smiles:
        return 0.0
    invs = []
    for ch in smiles:
        p = token_freq.get(ch, 1e-6)
        invs.append(1.0 / max(p, 1e-6))
    return float(sum(invs) / len(invs))


def main():
    st.title("AI-Assisted Drug Discovery (GNN Demo)")
    st.markdown(
        "This demo trains a small GNN on dummy SMILES-like data, uses transfer learning, and ranks candidates by predicted effectiveness."
    )

    with st.sidebar:
        st.header("Configuration")
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset_size = st.slider("Dataset size", 200, 5000, 1000, 100)
        max_len = st.slider("Max SMILES length", 8, 64, 24, 2)
        batch_size = st.slider("Batch size", 8, 128, 64, 8)
        pretrain_epochs = st.slider("Pretrain epochs", 1, 100, 20, 1)
        finetune_epochs = st.slider("Finetune epochs", 1, 100, 25, 1)
        learning_rate = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3], value=1e-3)
        hidden_dim = st.select_slider("Hidden dim", options=[32, 64, 128, 256], value=128)
        gnn_layers = st.select_slider("GNN layers", options=[2, 3, 4, 5], value=3)
        train_split = st.slider("Train split", 0.5, 0.9, 0.8, 0.05)
        val_split = st.slider("Validation split (of remaining)", 0.05, 0.45, 0.1, 0.05)

        top_k = st.slider("Top-K ranking", 5, 50, 20, 5)
        novelty_weight = st.slider("Novelty weight (0=ignore,1=only novelty)", 0.0, 1.0, 0.2, 0.05)
        device = get_device()
        st.caption(f"Device: {device}")

    st.subheader("Data")
    st.write("Use built-in dummy dataset (recommended) or upload a CSV with a column 'smiles'. Optional 'label' for effectiveness.")

    uploaded = st.file_uploader("Upload CSV (columns: smiles[, label])", type=["csv"]) 

    if uploaded is not None:
        df_all = pd.read_csv(uploaded)
        df_all = standardize_columns(df_all)
        if "smiles" not in df_all.columns:
            st.error("CSV must contain a 'smiles' column.")
            return
        has_label = "label" in df_all.columns
        if not has_label:
            st.info("No 'label' column found. Will generate synthetic labels (dummy effectiveness).")
            df_all = generate_dummy_smiles_dataset(
                n_samples=len(df_all),
                max_len=max_len,
                base_df=df_all,
                seed=seed,
            )
    else:
        if DEFAULT_DATA_PATH.exists():
            base_df = pd.read_csv(DEFAULT_DATA_PATH)
            df_all = generate_dummy_smiles_dataset(
                n_samples=dataset_size,
                max_len=max_len,
                base_df=base_df,
                seed=seed,
            )
        else:
            df_all = generate_dummy_smiles_dataset(n_samples=dataset_size, max_len=max_len, seed=seed)
        has_label = "label" in df_all.columns

    st.dataframe(df_all.head())

    st.subheader("Train")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        start_btn = st.button("Train Model (Pretrain + Finetune)")
    with col_b:
        infer_btn = st.button("Predict/Rank without training (use cached model)")

    model_state_key = "gnn_model_state"
    tokenizer_state_key = "tokenizer"

    if start_btn:
        with st.spinner("Preparing data and model..."):
            train_df, val_df, test_df = split_dataset(df_all, train_split, val_split)

            # Convert to graph batches
            x_train, a_train, m_train, y_train, tokenizer = smiles_to_graph_batch(
                train_df["smiles"].tolist(), train_df["label"].values if has_label else None, max_len=max_len
            )
            x_val, a_val, m_val, y_val, _ = smiles_to_graph_batch(
                val_df["smiles"].tolist(), val_df["label"].values if has_label else None, max_len=max_len, tokenizer=tokenizer
            )
            x_test, a_test, m_test, y_test, _ = smiles_to_graph_batch(
                test_df["smiles"].tolist(), test_df["label"].values if has_label else None, max_len=max_len, tokenizer=tokenizer
            )

            input_dim = x_train.shape[-1]
            model = SimpleGraphRegressor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=gnn_layers,
            ).to(device)

        loss_history_pre, loss_history_fine = train_with_transfer_learning(
            model,
            (x_train, a_train, m_train, y_train),
            (x_val, a_val, m_val, y_val),
            learning_rate=learning_rate,
            batch_size=batch_size,
            pretrain_epochs=pretrain_epochs,
            finetune_epochs=finetune_epochs,
            device=device,
        )

        st.pyplot(make_figure(loss_history_pre, loss_history_fine))

        st.session_state[model_state_key] = model.state_dict()
        st.session_state[tokenizer_state_key] = tokenizer

        with st.spinner("Evaluating on test set..."):
            test_metrics, preds = evaluate_model(model, (x_test, a_test, m_test, y_test), device=device)
        st.success(f"Test MSE: {test_metrics['mse']:.4f} | MAE: {test_metrics['mae']:.4f}")

        # Ranking on full dataset
        with st.spinner("Scoring and ranking candidates..."):
            x_all, a_all, m_all, y_all, _ = smiles_to_graph_batch(
                df_all["smiles"].tolist(), df_all["label"].values if has_label else None, max_len=max_len, tokenizer=tokenizer
            )
            with torch.no_grad():
                model.eval()
                preds_all = model(x_all.to(device), a_all.to(device), m_all.to(device)).cpu().numpy().reshape(-1)
        df_ranked = df_all.copy()
        df_ranked["predicted_effectiveness"] = preds_all
        # novelty scores based on token rarity from training split
        token_freq = _build_token_freq(train_df["smiles"].astype(str).tolist())
        df_ranked["novelty_score"] = [
            _novelty_score(s, token_freq) for s in df_ranked["smiles"].astype(str).tolist()
        ]
        # combine with simple min-max normalization
        pe = df_ranked["predicted_effectiveness"].values
        nv = df_ranked["novelty_score"].values
        pe_min, pe_max = float(pe.min()), float(pe.max())
        nv_min, nv_max = float(nv.min()), float(nv.max())
        pe_norm = (pe - pe_min) / (pe_max - pe_min + 1e-8)
        nv_norm = (nv - nv_min) / (nv_max - nv_min + 1e-8)
        combined = (1.0 - novelty_weight) * pe_norm + novelty_weight * nv_norm
        df_ranked["combined_score"] = combined
        df_ranked = df_ranked.sort_values("combined_score", ascending=False).reset_index(drop=True)

        st.subheader("Top Candidates")
        st.dataframe(df_ranked.head(top_k))

        csv_buf = io.StringIO()
        df_ranked.to_csv(csv_buf, index=False)
        st.download_button("Download Ranked Candidates", data=csv_buf.getvalue(), file_name="ranked_candidates.csv", mime="text/csv")

    if infer_btn:
        if model_state_key not in st.session_state or tokenizer_state_key not in st.session_state:
            st.warning("No trained model found in session. Please train first.")
            return
        with st.spinner("Preparing data and loading model..."):
            tokenizer = st.session_state[tokenizer_state_key]
            x_all, a_all, m_all, y_all, _ = smiles_to_graph_batch(
                df_all["smiles"].tolist(), df_all["label"].values if "label" in df_all.columns else None,
                max_len=max_len, tokenizer=tokenizer
            )
            input_dim = x_all.shape[-1]
            device = get_device()
            model = SimpleGraphRegressor(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=gnn_layers).to(device)
            model.load_state_dict(st.session_state[model_state_key])
            model.eval()
            with torch.no_grad():
                preds_all = model(x_all.to(device), a_all.to(device), m_all.to(device)).cpu().numpy().reshape(-1)
        df_ranked = df_all.copy()
        df_ranked["predicted_effectiveness"] = preds_all
        # novelty based on all visible data (no train split in quick infer)
        token_freq = _build_token_freq(df_all["smiles"].astype(str).tolist())
        df_ranked["novelty_score"] = [
            _novelty_score(s, token_freq) for s in df_ranked["smiles"].astype(str).tolist()
        ]
        pe = df_ranked["predicted_effectiveness"].values
        nv = df_ranked["novelty_score"].values
        pe_min, pe_max = float(pe.min()), float(pe.max())
        nv_min, nv_max = float(nv.min()), float(nv.max())
        pe_norm = (pe - pe_min) / (pe_max - pe_min + 1e-8)
        nv_norm = (nv - nv_min) / (nv_max - nv_min + 1e-8)
        combined = (1.0 - novelty_weight) * pe_norm + novelty_weight * nv_norm
        df_ranked["combined_score"] = combined
        df_ranked = df_ranked.sort_values("combined_score", ascending=False).reset_index(drop=True)
        st.subheader("Top Candidates")
        st.dataframe(df_ranked.head(top_k))
        csv_buf = io.StringIO()
        df_ranked.to_csv(csv_buf, index=False)
        st.download_button("Download Ranked Candidates", data=csv_buf.getvalue(), file_name="ranked_candidates.csv", mime="text/csv")


if __name__ == "__main__":
    main()
