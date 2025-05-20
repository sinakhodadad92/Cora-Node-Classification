import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data

import models                  # GCN / APPNP / GCN-II classes

DATA_DIR = Path(__file__).parent / "cora"

# CLI bands → (model_name, description)
BANDS = {
    "80": ("logreg", "Logistic Regression on TF-IDF word vectors (~80 %)."),
    "85": ("gcn",    "2-layer Graph Convolutional Network (~83 %)."),
    "87": ("appnp",  "APPNP – MLP + PPR propagation (~87 %)."),
    "89": ("gcn2",   "GCN-II – 32-layer residual GCN + DropEdge (~89 %)."),
}


# ──────────────────── Data loader ───────────────────────────
def read_cora():
    """
    Load Cora files from DATA_DIR, remap raw paper-IDs to 0…2707,
    build PyG Data object, and return:

        data        : torch_geometric.data.Data
        paper_ids   : original paper IDs (int) in dataset order
        label_str   : np.ndarray of class strings, index = int label
    """
    # ----- read cora.content -----
    content = np.loadtxt(DATA_DIR / "cora.content", dtype=str)
    raw_ids = content[:, 0].astype(int)                    # non-consecutive
    id2idx = {pid: i for i, pid in enumerate(raw_ids)}     # remap dict

    X_bin = content[:, 1:-1].astype(np.float32)            # 1433-dim BOW
    labels_str = content[:, -1]

    classes = np.sort(np.unique(labels_str))
    str2idx = {s: i for i, s in enumerate(classes)}
    y_int = np.array([str2idx[s] for s in labels_str], dtype=np.int64)

    # ----- read & remap cora.cites -----
    cites = np.loadtxt(DATA_DIR / "cora.cites", dtype=int)
    mask = np.isin(cites, raw_ids).all(axis=1)             # drop unknowns
    cites = cites[mask]
    src = [id2idx[s] for s in cites[:, 0]]
    dst = [id2idx[d] for d in cites[:, 1]]

    # make edges undirected
    edge_index = torch.tensor(
        np.vstack([np.r_[src, dst], np.r_[dst, src]]), dtype=torch.long
    )

    # ----- build PyG Data -----
    data = Data(
        x=torch.tensor(X_bin, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y_int)
    )
    return data, raw_ids, classes


# ──────────────────── Helper functions ─────────────────────
def choose_band() -> str:
    """Ask user for desired accuracy range; default = '80'."""
    print("Which accuracy range?")
    for k, (_, msg) in BANDS.items():
        print(f"  [{k}] {msg}")
    return input("Enter 80 / 85 / 87 / 89 (default 80): ").strip() or "80"


def run_logreg(x_np, y_int, train_idx):
    """
    TF-IDF + Logistic Regression baseline.
    Returns integer label predictions for *all* nodes.
    """
    tfidf = TfidfTransformer().fit_transform(x_np).toarray()  
    clf = LogisticRegression(max_iter=500, n_jobs=-1, C=4.0)
    clf.fit(tfidf[train_idx], y_int[train_idx])
    return clf.predict(tfidf).astype(np.int64)


def run_gnn(model_name, data, train_idx,
            epochs=200, lr=0.01, wd=5e-4, device="cpu"):
    """
    Train a PyG model on one fold and return integer label predictions.
    Only nodes in `train_idx` participate in loss; inference on all nodes.
    """
    net = models.get_model(model_name, data.num_features,
                           int(data.y.max()) + 1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    data = data.to(device)

    for _ in range(epochs):
        net.train(); opt.zero_grad()
        logits = net(data.x, data.edge_index)
        F.cross_entropy(logits[train_idx], data.y[train_idx]).backward()
        opt.step()

    net.eval()
    return net(data.x, data.edge_index).argmax(dim=-1).cpu().numpy()


def write_tsv(pred_int, paper_ids, label_strings,
              path="predictions.tsv"):
    """Save TSV file in the required  <paper_id> <class_label>  format."""
    with open(path, "w") as f:
        for pid, lab_idx in zip(paper_ids, pred_int):
            f.write(f"{pid}\t{label_strings[lab_idx]}\n")


# ──────────────────── Main entry ­point ────────────────────
def main():
    warnings.filterwarnings("ignore")

    # --- CLI args ---
    argp = argparse.ArgumentParser()
    argp.add_argument("--model", choices=[v[0] for v in BANDS.values()],
                      help="Skip prompt and pick a model directly.")
    argp.add_argument("--epochs", type=int, default=200)
    argp.add_argument("--lr",     type=float, default=0.01)
    argp.add_argument("--wd",     type=float, default=5e-4)
    args = argp.parse_args()

    # --- decide which model to run ---
    band = (next(k for k, v in BANDS.items() if v[0] == args.model)
            if args.model else choose_band())
    model_name, description = BANDS[band]
    print(f"\n Using **{model_name.upper()}** – {description}\n")

    # --- load data ---
    data, paper_ids, label_strings = read_cora()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 10-fold stratified cross-validation ---
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    preds_global = np.empty(data.num_nodes, dtype=np.int64)

    for fold, (train_idx, test_idx) in enumerate(skf.split(data.x, data.y)):
        print(f"⋯ fold {fold + 1}/10")
        train_idx = train_idx.astype(np.int64)

        # choose baseline or GNN
        if model_name == "logreg":
            preds_fold = run_logreg(data.x.numpy(), data.y.numpy(), train_idx)
        else:
            preds_fold = run_gnn(model_name, data.clone(),
                                 torch.tensor(train_idx),
                                 args.epochs, args.lr, args.wd, device)

        preds_global[test_idx] = preds_fold[test_idx]

    # --- evaluation & write-out ---
    acc = accuracy_score(data.y, preds_global)
    write_tsv(preds_global, paper_ids, label_strings)
    print(f"\n 10-fold accuracy = {acc:.2%}   (predictions.tsv written)")


if __name__ == "__main__":
    main()
