import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, APPNP, GCN2Conv


# ───────────────────────────── MODELS ──────────────────────────────
class GCN(nn.Module):
    """Classic 2-layer Graph Convolutional Network."""
    def __init__(self, in_dim: int, hid: int = 16, num_classes: int = 7):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)          # first graph conv
        self.c2 = GCNConv(hid, num_classes)     # output layer

    def forward(self, x, edge_index):
        # ReLU + dropout after 1st layer, no act after 2nd (logits)
        x = F.dropout(F.relu(self.c1(x, edge_index)), 0.5, self.training)
        return self.c2(x, edge_index)


class APPNNet(nn.Module):
    """MLP feature encoder +  K-step personalised-PageRank propagation."""
    def __init__(self, in_dim: int, hid: int = 64, num_classes: int = 7,
                 K: int = 10, alpha: float = 0.1):
        super().__init__()
        self.lin = nn.Linear(in_dim, hid)          # MLP encoder (1 layer)
        self.prop = APPNP(K=K, alpha=alpha)        # fixed PPR operator
        self.out = nn.Linear(hid, num_classes)

    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.lin(x)), 0.5, self.training)
        x = self.prop(x, edge_index)               # propagate features
        return self.out(x)


class GCNII(nn.Module):
    """
    Identity-mapping deep GCN (GCN-II).

    * 32 residual graph convolutions by default
    * alpha, theta are hyper-parameters from the paper
    """
    def __init__(self, in_dim: int, hid: int = 64, layers: int = 32, num_classes: int = 7, alpha: float = 0.1, theta: float = 0.5):
        super().__init__()
        self.lin0 = nn.Linear(in_dim, hid)         # initial projection
        # stack of GCN2Conv layers (same hidden size)
        self.convs = nn.ModuleList(
            GCN2Conv(hid, alpha, theta, layer=i + 1)
            for i in range(layers)
        )
        self.out = nn.Linear(hid, num_classes)

    def forward(self, x, edge_index):
        h0 = F.relu(self.lin0(x))                  # keep initial features
        h = h0
        for conv in self.convs:                    # deep residual stack
            h = F.dropout(F.relu(conv(h, h0, edge_index)),
                           0.6, self.training)
        return self.out(h)


# ───────────────────────────── FACTORY ─────────────────────────────
def get_model(name: str, in_dim: int, num_classes: int):
    name = name.lower()
    if name == "gcn":
        return GCN(in_dim, 16, num_classes)
    if name == "appnp":
        return APPNNet(in_dim, 64, num_classes)
    if name in ("gcn2", "gcnii"):
        return GCNII(in_dim, 64, 32, num_classes)
    raise ValueError(f"Unknown model name: {name}")
