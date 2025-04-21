import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class VanillaMPNN(MessagePassing):
    """1‑hop message‑passing block with an edge MLP (optional)."""
    def __init__(self, dim, edge_dim=None, aggr="add"):
        super().__init__(aggr=aggr)
        self.msg_mlp = nn.Sequential(
            nn.Linear(2*dim + (edge_dim or 0), dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(dim + dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, edge_index, edge_attr=None):
        # add self‑loops so every node sees its own state
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_j: source, x_i: target
        if edge_attr is None:
            m = torch.cat([x_i, x_j], dim=-1)
        else:
            m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_mlp(m)

    def update(self, aggr_out, x):
        return self.upd_mlp(torch.cat([x, aggr_out], dim=-1))
