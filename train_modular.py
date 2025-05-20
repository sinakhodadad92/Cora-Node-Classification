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

import models  # GCN / APPNP / GCN-II classes

DATA_DIR = Path(__file__).parent / "cora"

# CLI key → (model_name, description)
BANDS = {
    "logreg": ("logreg", "Logistic Regression on TF-IDF word vectors."),
    "gcn":    ("gcn",    "2-layer Graph Convolutional Network."),
    "appnp":  ("appnp",  "APPNP – MLP + PPR propagation."),
    "gcn2":   ("gcn2",   "GCN-II – 32-layer residual GCN + DropEdge."),
}


# ────────────── Data loader ─────────────────────────────────────────
def read_cora():
    content = np.loadtxt(DATA_DIR / "cora.content", dtype=str)
    raw_ids = content[:, 0].astype(int)
    id2idx  = {pid: i for i, pid in enumerate(raw_ids)}

    X_bin      = content[:, 1:-1].astype(np.float32)
    label_str  = content[:, -1]
    classes    = np.sort(np.unique(label_str))
    str2idx    = {s: i for i, s in enumerate(classes)}
    y_int      = np.array([str2idx[s] for s in label_str], dtype=np.int64)

    cites = np.loadtxt(DATA_DIR / "cora.cites", dtype=int)
    mask  = np.isin(cites, raw_ids).all(axis=1)
    cites = cites[mask]
    src   = [id2idx[s] for s in cites[:, 0]]
    dst   = [id2idx[d] for d in cites[:, 1]]

    edge_index = torch.tensor(
        np.vstack([np.r_[src, dst], np.r_[dst, src]]), dtype=torch.long
    )

    data = Data(
        x=torch.tensor(X_bin, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y_int),
    )
    return data, raw_ids, classes


# ────────────── Helper functions ────────────────────────────────────
def choose_band() -> str:
    """Prompt user; fall back to 'logreg' if input is unknown."""
    print("Which model?")
    for k, (_, msg) in BANDS.items():
        print(f"  [{k}] {msg}")
    choice = (
        input("Enter logreg / gcn / appnp / gcn2 (default logreg): ")
        .strip()
        .lower()
    )
    if choice not in BANDS:
        print("Unrecognised option → using 'logreg'.")
        choice = "logreg"
    return choice


def run_logreg(x_np, y_int, train_idx):
    tfidf = TfidfTransformer().fit_transform(x_np).toarray()
    clf   = LogisticRegression(max_iter=500, n_jobs=-1, C=4.0)
    clf.fit(tfidf[train_idx], y_int[train_idx])
    return clf.predict(tfidf).astype(np.int64)


def run_gnn(model_name, data, train_idx, epochs, lr, wd, device):
    net = models.get_model(model_name, data.num_features, int(data.y.max()) + 1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    data = data.to(device)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        out = net(data.x, data.edge_index)
        F.cross_entropy(out[train_idx], data.y[train_idx]).backward()
        opt.step()
    net.eval()
    return net(data.x, data.edge_index).argmax(dim=-1).cpu().numpy()


def write_tsv(pred_int, paper_ids, label_strings, path="predictions.tsv"):
    with open(path, "w") as f:
        for pid, lab_idx in zip(paper_ids, pred_int):
            f.write(f"{pid}\t{label_strings[lab_idx]}\n")


# ────────────── Main ───────────────────────────────────────────────
def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=BANDS.keys(),
                        help="Skip prompt and pick a model directly.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=5e-4)
    args = parser.parse_args()

    chosen = args.model or choose_band()
    model_name, description = BANDS[chosen]
    print(f"\nUsing **{model_name.upper()}** – {description}\n")

    data, paper_ids, label_strings = read_cora()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    preds_global = np.empty(data.num_nodes, dtype=np.int64)

    for fold, (train_idx, test_idx) in enumerate(skf.split(data.x, data.y)):
        print(f"⋯ fold {fold + 1}/10")
        train_idx = train_idx.astype(np.int64)

        if model_name == "logreg":
            preds_fold = run_logreg(data.x.numpy(), data.y.numpy(), train_idx)
        else:
            preds_fold = run_gnn(
                model_name, data.clone(),
                torch.tensor(train_idx), args.epochs, args.lr, args.wd, device
            )

        preds_global[test_idx] = preds_fold[test_idx]

    acc = accuracy_score(data.y, preds_global)
    out_name = f"predictions_{model_name}.tsv"
    write_tsv(preds_global, paper_ids, label_strings, path=out_name)
    print(f"\n10-fold accuracy = {acc:.2%}   ({out_name} written)")
    

if __name__ == "__main__":
    main()
