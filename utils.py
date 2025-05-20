from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# ── TSV writer ───────────────────────────────────────────────
def write_predictions(pred_tensor: torch.Tensor, data: Data) -> None:
    """Save <paper_id> <class_label> lines to predictions.tsv."""
    labels = [data.y.unique(sorted=True)[i].item() for i in range(data.num_classes)]
    lines = [f"{int(data.paper_id[i])}\t{labels[pred_tensor[i]]}"
             for i in range(pred_tensor.size(0))]
    Path("predictions.tsv").write_text("\n".join(lines))

# ── Local Cora loader (no ID remap) ──────────────────────────
def read_cora_local(data_dir: str | Path = "cora"):
    """Load cora.content / cora.cites from disk and return Data, ids, labels."""
    data_dir = Path(data_dir)

    content = pd.read_csv(data_dir / "cora.content", sep="\t", header=None)
    cites   = pd.read_csv(data_dir / "cora.cites",   sep="\t", header=None)

    paper_ids = content.iloc[:, 0].to_numpy(dtype=int)
    X_bin     = content.iloc[:, 1:-1].to_numpy(dtype=np.float32)
    y_str     = content.iloc[:, -1].to_numpy()

    lbl2idx = {s: i for i, s in enumerate(sorted(np.unique(y_str)))}
    y_int   = np.array([lbl2idx[s] for s in y_str], dtype=np.int64)

    edge_index = torch.tensor(cites.to_numpy().T, dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)  # make undirected

    data = Data(x=torch.tensor(X_bin, dtype=torch.float),
                edge_index=edge_index,
                y=torch.tensor(y_int, dtype=torch.long))

    return data, paper_ids, y_str