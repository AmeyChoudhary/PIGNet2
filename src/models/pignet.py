# from collections import defaultdict
# from typing import Dict, Optional, Tuple

# import torch
# import torch.nn.functional as F
# from omegaconf import DictConfig
# from torch.nn import Dropout, Module, ModuleList, Parameter, ReLU, Sigmoid, Tanh
# from torch.nn.parameter import UninitializedParameter
# from torch.optim import Adam
# from torch_geometric.data import Batch
# from torch_geometric.nn import Linear, Sequential
# from torch_scatter import scatter

# from . import physics
# from .layers import GatedGAT, InteractionNet


# class PIGNet(Module):
#     def __init__(
#         self,
#         config: DictConfig,
#         in_features: int = -1,
#         **kwargs,
#     ):
#         super().__init__()
#         self.reset_log()
#         self.config = config
#         n_gnn = config.model.n_gnn
#         dim_gnn = config.model.dim_gnn
#         dim_mlp = config.model.dim_mlp
#         dropout_rate = config.run.dropout_rate

#         self.embed = Linear(in_features, dim_gnn, bias=False)

#         self.intraconv = ModuleList()
#         for _ in range(n_gnn):
#             self.intraconv.append(
#                 Sequential(
#                     "x, edge_index",
#                     [
#                         (GatedGAT(dim_gnn, dim_gnn), "x, edge_index -> x"),
#                         (Dropout(dropout_rate), "x -> x"),
#                     ],
#                 )
#             )

#         self.interconv = ModuleList()
#         if config.model.interconv:
#             for _ in range(n_gnn):
#                 self.interconv.append(
#                     Sequential(
#                         "x, edge_index",
#                         [
#                             (InteractionNet(dim_gnn), "x, edge_index -> x"),
#                             (Dropout(dropout_rate), "x -> x"),
#                         ],
#                     )
#                 )

#         self.nn_vdw_epsilon = Sequential(
#             "x",
#             [
#                 (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
#                 ReLU(),
#                 Linear(dim_mlp, 1),
#                 Sigmoid(),
#             ],
#         )

#         self.nn_dvdw = Sequential(
#             "x",
#             [
#                 (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
#                 ReLU(),
#                 Linear(dim_mlp, 1),
#                 Tanh(),
#             ],
#         )

#         self.hbond_coeff = Parameter(torch.tensor([1.0]))
#         self.hydrophobic_coeff = Parameter(torch.tensor([0.5]))
#         self.rotor_coeff = Parameter(torch.tensor([0.5]))
#         if config.model.get("include_ionic", False):
#             self.ionic_coeff = Parameter(torch.tensor([1.0]))

#     @property
#     def size(self) -> Tuple[int, int]:
#         """Get the number of all learnable parameters.

#         Returns: (num_parameters, num_uninitialized_parameters)
#         """
#         num_params = 0
#         num_uninitialized = 0

#         for param in self.parameters():
#             if isinstance(param, UninitializedParameter):
#                 num_uninitialized += 1
#             elif param.requires_grad:
#                 num_params += param.numel()

#         return num_params, num_uninitialized

#     @property
#     def in_features(self) -> int:
#         """Get the number of input features."""
#         try:
#             return self.embed.in_channels
#         except AttributeError:
#             return self.embed.in_features

#     @property
#     def device(self) -> torch.device:
#         return next(self.parameters()).device

#     def conv(self, x, edge_index_1, edge_index_2):
#         for conv in self.intraconv:
#             x = conv(x, edge_index_1)

#         for conv in self.interconv:
#             x = conv(x, edge_index_2)
#         return x

#     def forward(self, sample: Batch):
#         cfg = self.config.model

#         # Initial embedding
#         x = self.embed(sample.x)

#         # Graph convolutions
#         x = self.conv(x, sample.edge_index, sample.edge_index_c)

#         # Ligand-to-target uni-directional edges
#         # to compute pairwise interactions: (2, pairs)
#         edge_index_i = physics.interaction_edges(sample.is_ligand, sample.batch)

#         # Pairwise distances: (pairs,)
#         D = physics.distances(sample.pos, edge_index_i)

#         # Limit the interaction distance.
#         _mask = (cfg.interaction_range[0] <= D) & (D <= cfg.interaction_range[1])
#         edge_index_i = edge_index_i[:, _mask]
#         D = D[_mask]

#         # Pairwise node features: (pairs, 2*features)
#         x_cat = torch.cat((x[edge_index_i[0]], x[edge_index_i[1]]), -1)

#         # Pairwise vdW-radii deviations: (pairs,)
#         dvdw_radii = self.nn_dvdw(x_cat).view(-1)
#         dvdw_radii = dvdw_radii * cfg.dev_vdw_radii_coeff

#         # Pairwise vdW radii: (pairs,)
#         R = (
#             sample.vdw_radii[edge_index_i[0]]
#             + sample.vdw_radii[edge_index_i[1]]
#             + dvdw_radii
#         )

#         # Prepare a pair-energies contrainer: (energy_types, pairs)
#         if cfg.get("include_ionic", False):
#             energies_pairs = torch.empty(5, D.numel()).to(self.device)
#         else:
#             energies_pairs = torch.empty(4, D.numel()).to(self.device)

#         # vdW energy minima (well depths): (pairs,)
#         vdw_epsilon = self.nn_vdw_epsilon(x_cat).view(-1)
#         # Scale the minima as done in AutoDock Vina.
#         vdw_epsilon = (
#             vdw_epsilon * (cfg.vdw_epsilon_scale[1] - cfg.vdw_epsilon_scale[0])
#             + cfg.vdw_epsilon_scale[0]
#         )
#         # vdW interaction
#         energies_pairs[0] = physics.lennard_jones_potential(
#             D, R, vdw_epsilon, cfg.vdw_N_short, cfg.vdw_N_long
#         )

#         # Hydrogen-bond, metal-ligand, hydrophobic interactions
#         minima_hbond = -(self.hbond_coeff**2)
#         minima_hydrophobic = -(self.hydrophobic_coeff**2)
#         energies_pairs[1] = physics.linear_potential(
#             D, R, minima_hbond, *cfg.hydrogen_bond_cutoffs
#         )
#         energies_pairs[2] = physics.linear_potential(
#             D, R, minima_hbond, *cfg.metal_ligand_cutoffs
#         )
#         energies_pairs[3] = physics.linear_potential(
#             D, R, minima_hydrophobic, *cfg.hydrophobic_cutoffs
#         )
#         # Include the ionic interaction if required.
#         if cfg.get("include_ionic", False):
#             # Note the sign of `minima_ionic`
#             minima_ionic = self.ionic_coeff**2 * (
#                 sample.atom_charges[edge_index_i[0]]
#                 * sample.atom_charges[edge_index_i[1]]
#             )
#             energies_pairs[4] = physics.linear_potential(
#                 D, R, minima_ionic, *cfg.ionic_cutoffs
#             )

#         # Interaction masks according to atom types: (energy_types, pairs)
#         masks = physics.interaction_masks(
#             sample.is_metal,
#             sample.is_h_donor,
#             sample.is_h_acceptor,
#             sample.is_hydrophobic,
#             edge_index_i,
#             include_ionic=cfg.get("include_ionic", False),
#         )
#         energies_pairs = energies_pairs * masks

#         # Per-graph sum -> (energy_types, batch)
#         energies = scatter(energies_pairs, sample.batch[edge_index_i[0]])
#         # Reshape -> (batch, energy_types)
#         energies = energies.t().contiguous()

#         # Rotor penalty
#         if cfg.rotor_penalty:
#             penalty = 1 + self.rotor_coeff**2 * sample.rotor
#             # -> (batch, 1)
#             energies = energies / penalty

#         return energies, dvdw_radii

#     def loss_dvdw(self, dvdw_radii: torch.Tensor):
#         loss = dvdw_radii.pow(2).mean()
#         return loss

#     def loss_regression(
#         self,
#         energies: torch.Tensor,
#         true: torch.Tensor,
#     ):
#         return F.mse_loss(energies.sum(-1, True), true)

#     def loss_augment(
#         self,
#         energies: torch.Tensor,
#         true: torch.Tensor,
#         min: Optional[float] = None,
#         max: Optional[float] = None,
#     ):
#         """Loss functions for docking, random & cross screening.

#         Args:
#             sample
#             task: 'docking' | 'random' | 'cross'
#         """
#         loss_energy = true - energies.sum(-1, True)
#         loss_energy = loss_energy.clamp(min, max)
#         loss_energy = loss_energy.mean()
#         return loss_energy

#     def training_step(self, batch: Dict[str, Batch]):
#         loss_total = torch.tensor(0.0, device=self.device)

#         for task, sample in batch.items():
#             task_config = self.config.data[task]

#             energies, dvdw_radii = self(sample)
#             loss_dvdw = self.loss_dvdw(dvdw_radii)
#             if task_config.objective == "regression":
#                 loss_energy = self.loss_regression(energies, sample.y)
#             elif task_config.objective == "augment":
#                 loss_energy = self.loss_augment(
#                     energies, sample.y, *task_config.loss_range
#                 )
#             else:
#                 raise NotImplementedError(
#                     "Current loss functions only support regression and augment."
#                 )

#             loss_total += loss_energy * task_config.loss_ratio
#             loss_total += loss_dvdw * self.config.run.loss_dvdw_ratio

#             # Update log
#             self.losses["energy"][task].append(loss_energy.item())
#             self.losses["dvdw"][task].append(loss_dvdw.item())
#             for key, pred, true in zip(sample.key, energies, sample.y):
#                 self.predictions[task][key] = pred.tolist()
#                 self.labels[task][key] = true.item()

#         return loss_total

#     def validation_step(self, batch: Dict[str, Batch]):
#         return self.training_step(batch)

#     def test_step(self, batch: Batch):
#         sample = batch
#         task = next(iter(self.config.data))

#         energies, dvdw_radii = self(sample)
#         loss_energy = self.loss_regression(energies, sample.y)
#         loss_dvdw = self.loss_dvdw(dvdw_radii)

#         # Update log
#         self.losses["energy"][task].append(loss_energy.item())
#         self.losses["dvdw"][task].append(loss_dvdw.item())
#         for key, pred, true in zip(sample.key, energies, sample.y):
#             self.predictions[task][key] = pred.tolist()
#             self.labels[task][key] = true.item()

#     def predict_step(self, batch: Batch):
#         sample = batch
#         task = next(iter(self.config.data))
#         energies, dvdw_radii = self(sample)
#         for key, pred in zip(sample.key, energies):
#             self.predictions[task][key] = pred.tolist()

#     def configure_optimizers(self):
#         return Adam(
#             self.parameters(),
#             lr=self.config.run.lr,
#             weight_decay=self.config.run.weight_decay,
#         )

#     def reset_log(self):
#         """Reset logs. Intended to be called every epoch.

#         Attributes:
#             losses: Dict[str, Dict[str, List[float]]]
#                 losses[loss_type][task] -> loss_values
#                 where
#                     loss_type: 'energy' | 'dvdw'
#                     task: 'scoring' | 'docking' | 'random' | 'cross' | ...
#                     loss_values: List[float] of shape (batches,)

#             predictions: Dict[str, Dict[str, Tuple[float, ...]]]
#                 predictions[task][key] -> energies
#                 where
#                     energies: List[float] of shape (4,)

#             labels: Dict[str, Dict[str, float]]
#                 labels[task][key] -> energy (float)
#         """
#         self.losses = defaultdict(lambda: defaultdict(list))
#         self.predictions = defaultdict(dict)
#         self.labels = defaultdict(dict)

"""PIGNet upgraded with ProtBERT + MPNN **and** automatic sequence inference.

* 2025‑04‑14 – verbose edition: every major stage prints a message so you can
  trace a single forward pass from raw `Batch` → energies.
* If you wish to silence the output, comment‑out the `print()` calls or guard
  them with an `if self.verbose:` flag.

Required *per‑atom* fields (same as original PIGNet):
    x, pos, vdw_radii, is_ligand, is_metal, is_h_donor, is_h_acceptor,
    is_hydrophobic, atom_charges, batch, edge_index, edge_index_c, rotor

Optional fields for automatic ProtBERT inference:
    residue_name (bytes/str 3‑letter), residue_id (int), chain_id (int)
If neither the optional fields **nor** the pre‑built `prot_seq` / `residue_index`
are present, the model falls back to a single dummy residue (zero vector).
"""

from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import Dropout, Module, ModuleList, Parameter, ReLU, Sigmoid, Tanh
from torch.nn.parameter import UninitializedParameter
from torch.optim import Adam
from torch_geometric.data import Batch
from torch_geometric.nn import Linear, Sequential
from torch_scatter import scatter
from transformers import AutoTokenizer, AutoModel

from . import physics
from .layers import InteractionNet
from .layers import VanillaMPNN as MPNN

# ──────────────────────────────────────────────────────────────────────────────
# 3‑letter → 1‑letter AA map (minimal, no BioPython dependency)
# ──────────────────────────────────────────────────────────────────────────────
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# loading prot_bert
prot_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
prot_bert = AutoModel.from_pretrained("Rostlab/prot_bert")
# print("[PIGNet] Loading ProtBERT… this may take a few seconds the first time")
for p in prot_bert.parameters():
    p.requires_grad = False
# print("[PIGNet] ProtBERT frozen (no fine‑tuning)")
prot_proj = Linear(prot_bert.config.hidden_size, 128, bias=False) # dim_gnn taken to be 128 always


class PIGNet(Module):
    """Protein–ligand graph network with ProtBERT embeddings and MPNN message passing.

    Verbose edition – prints every major step to stdout.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Init
    # ──────────────────────────────────────────────────────────────────────
    def __init__(self, config: DictConfig, in_features: int = -1, **kwargs):
        super().__init__()
        self.reset_log()
        self.config = config
        n_gnn = config.model.n_gnn
        dim_gnn = config.model.dim_gnn
        dim_mlp = config.model.dim_mlp
        dropout_rate = config.run.dropout_rate
        lig_in_features = config.model.ligand_in_features
        self.dim_gnn = dim_gnn

        # print("[PIGNet] Initialising…  dim_gnn =", dim_gnn)

        # Ligand atom embedder
        self.lig_embed = None
        # print("[PIGNet] Ligand embedder created ({} → {})".format(lig_in_features, dim_gnn))

        # ProtBERT encoder
        # print("[PIGNet] Loading ProtBERT… this may take a few seconds the first time")

        # Message‑passing layers
        self.intraconv = torch.nn.ModuleList([MPNN(dim_gnn, aggr="add") for _ in range(n_gnn)])

        # print(f"[PIGNet] Created {n_gnn} MPNN layers")

        # Optional inter‑molecular blocks (same as original)
        self.interconv = ModuleList()
        if config.model.interconv:
            for _ in range(n_gnn):
                self.interconv.append(Sequential("x, edge_index", [(InteractionNet(dim_gnn), "x, edge_index -> x"), (Dropout(dropout_rate), "x -> x")]))
            # print(f"[PIGNet] Added {n_gnn} InteractionNet blocks for cross‑graph attention")

        # Physics‑based head (unchanged)
        self.nn_vdw_epsilon = Sequential("x", [(Linear(dim_gnn * 2, dim_mlp), "x -> x"), ReLU(), Linear(dim_mlp, 1), Sigmoid()])
        self.nn_dvdw = Sequential("x", [(Linear(dim_gnn * 2, dim_mlp), "x -> x"), ReLU(), Linear(dim_mlp, 1), Tanh()])
        self.hbond_coeff = Parameter(torch.tensor([1.0]))
        self.hydrophobic_coeff = Parameter(torch.tensor([0.5]))
        self.rotor_coeff = Parameter(torch.tensor([0.5]))
        if config.model.get("include_ionic", False):
            self.ionic_coeff = Parameter(torch.tensor([1.0]))

        # print("[PIGNet] Initialisation complete")

    # ──────────────────────────────────────────────────────────────────────
    # Helpers – protein attributes
    # ──────────────────────────────────────────────────────────────────────
    def _ensure_protein_attributes(self, sample: Batch):
        if hasattr(sample, "prot_seq") and hasattr(sample, "residue_index"):
            # print("[PIGNet] Batch already contains prot_seq & residue_index → using them")
            return
        try:
            self._infer_protein_attributes(sample)
            # print("[PIGNet] Inferred prot_seq & residue_index from residue_name/id/chain_id")
        except Exception as e:
            # print("[PIGNet] WARNING: could not infer protein sequence (", e, ") → using dummy residue")
            prot_mask = ~sample.is_ligand
            sample.prot_seq = "X"
            sample.residue_index = torch.zeros(prot_mask.sum(), dtype=torch.long, device=sample.x.device)

    def _infer_protein_attributes(self, sample: Batch):
        assert hasattr(sample, "residue_name") and hasattr(sample, "residue_id") and hasattr(sample, "chain_id"), "Missing fields for inference"
        prot_mask = ~sample.is_ligand
        names = sample.residue_name[prot_mask]
        res_ids = sample.residue_id[prot_mask]
        chains = sample.chain_id[prot_mask]
        keys, seq_letters, idx_map = [], [], {}
        for nm, rid, ch in zip(names, res_ids, chains):
            key = (int(ch), int(rid))
            if key not in idx_map:
                idx_map[key] = len(seq_letters)
                three = nm.decode() if isinstance(nm, (bytes, bytearray)) else str(nm)
                seq_letters.append(THREE_TO_ONE.get(three.upper(), "X"))
            keys.append(idx_map[key])
        sample.prot_seq = "".join(seq_letters)
        sample.residue_index = torch.tensor(keys, dtype=torch.long, device=sample.x.device)

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────
    @property
    def size(self) -> Tuple[int, int]:
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad and not isinstance(p, UninitializedParameter))
        num_uninit = sum(1 for p in self.parameters() if isinstance(p, UninitializedParameter))
        return num_params, num_uninit

    @property
    def in_features(self) -> int:
        return self.lig_embed.in_channels

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ──────────────────────────────────────────────────────────────────────
    # Graph conv helper
    # ──────────────────────────────────────────────────────────────────────
    def conv(self, x, edge_index_1, edge_index_2):
        # print("[PIGNet]   » Intra‑graph message passing…")
        for conv in self.intraconv:
            x = conv(x, edge_index_1)
        for conv in self.interconv:
            x = conv(x, edge_index_2)
        return x

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────
    def forward(self, sample: Batch):
        cfg = self.config.model
        device = self.device

        # print("[PIGNet] Forward pass started – batch size:", sample.batch.max().item() + 1)

        # Ensure PLM attributes
        self._ensure_protein_attributes(sample)

        lig_mask = sample.is_ligand
        prot_mask = ~lig_mask

        # 1. Node embeddings
        # print("[PIGNet]   » Embedding ligand atoms…")
        if self.lig_embed is None:
            in_dim = sample.x.size(1)          # detect at runtime
            self.lig_embed = Linear(in_dim, self.dim_gnn, bias=False).to(self.device)
            # print(f"[PIGNet] Created ligand embedder on-the-fly ({in_dim} → {self.dim_gnn})")
        x_lig = self.lig_embed(sample.x[lig_mask])
        # print("[PIGNet]   » Encoding protein sequence of length", len(sample.prot_seq))
        global prot_proj
        prot_proj = prot_proj.to(device)
        residue_repr = prot_proj(self._encode_protein(sample.prot_seq, device))  # (L, dim_gnn)
        x_prot = residue_repr[sample.residue_index]

        x = torch.zeros(sample.x.size(0), residue_repr.size(1), device=device)
        x[lig_mask] = x_lig
        x[prot_mask] = x_prot
        # print("[PIGNet]   » Node feature matrix built – shape:", x.shape)

        # 2. Message passing
        x = self.conv(x, sample.edge_index, sample.edge_index_c)
        # print("[PIGNet]   » Message passing done")

        # 3. Physics head (identical to original)
        # print("[PIGNet]   » Computing physics‑based interaction energies…")
        edge_index_i = physics.interaction_edges(sample.is_ligand, sample.batch)
        D = physics.distances(sample.pos, edge_index_i)
        _mask = (cfg.interaction_range[0] <= D) & (D <= cfg.interaction_range[1])
        edge_index_i = edge_index_i[:, _mask]
        D = D[_mask]
        x_cat = torch.cat((x[edge_index_i[0]], x[edge_index_i[1]]), -1)
        dvdw_radii = self.nn_dvdw(x_cat).view(-1) * cfg.dev_vdw_radii_coeff
        R = sample.vdw_radii[edge_index_i[0]] + sample.vdw_radii[edge_index_i[1]] + dvdw_radii
        n_types = 5 if cfg.get("include_ionic", False) else 4
        energies_pairs = torch.empty(n_types, D.numel(), device=device)
        vdw_epsilon = self.nn_vdw_epsilon(x_cat).view(-1)
        vdw_epsilon = vdw_epsilon * (cfg.vdw_epsilon_scale[1] - cfg.vdw_epsilon_scale[0]) + cfg.vdw_epsilon_scale[0]
        energies_pairs[0] = physics.lennard_jones_potential(D, R, vdw_epsilon, cfg.vdw_N_short, cfg.vdw_N_long)
        minima_hbond = -(self.hbond_coeff ** 2)
        minima_hydrophobic = -(self.hydrophobic_coeff ** 2)
        energies_pairs[1] = physics.linear_potential(D, R, minima_hbond, *cfg.hydrogen_bond_cutoffs)
        energies_pairs[2] = physics.linear_potential(D, R, minima_hbond, *cfg.metal_ligand_cutoffs)
        energies_pairs[3] = physics.linear_potential(D, R, minima_hydrophobic, *cfg.hydrophobic_cutoffs)
        if cfg.get("include_ionic", False):
            minima_ionic = self.ionic_coeff ** 2 * (sample.atom_charges[edge_index_i[0]] * sample.atom_charges[edge_index_i[1]])
            energies_pairs[4] = physics.linear_potential(D, R, minima_ionic, *cfg.ionic_cutoffs)
        masks = physics.interaction_masks(sample.is_metal, sample.is_h_donor, sample.is_h_acceptor, sample.is_hydrophobic, edge_index_i, include_ionic=cfg.get("include_ionic", False))
        energies_pairs = energies_pairs * masks
        energies = scatter(energies_pairs, sample.batch[edge_index_i[0]])
        energies = energies.t().contiguous()
        if cfg.rotor_penalty:
            penalty = 1 + self.rotor_coeff ** 2 * sample.rotor
            energies = energies / penalty
        # print("[PIGNet]   » Energies computed – shape:", energies.shape)
        return energies, dvdw_radii

    # ──────────────────────────────────────────────────────────────────────
    # ProtBERT helper
    # ──────────────────────────────────────────────────────────────────────
    def _encode_protein(self, seq: str, device: torch.device):
        toks = prot_tokenizer(" ".join(list(seq)), return_tensors="pt").to(device)
        prot_bert.to(device)
        with torch.set_grad_enabled(prot_bert.training):
            out = prot_bert(**toks).last_hidden_state.squeeze(0).to(device)
        return out[1:-1].to(device)  # strip CLS/SEP
        
    # ──────────────────────────────────────────────────────────────────────
    # Losses (unchanged)
    # ──────────────────────────────────────────────────────────────────────
    def loss_dvdw(self, dvdw_radii: torch.Tensor):
        return dvdw_radii.pow(2).mean()

    def loss_regression(self, energies: torch.Tensor, true: torch.Tensor):
        return F.mse_loss(energies.sum(-1, True), true)

    def loss_augment(self, energies: torch.Tensor, true: torch.Tensor, min: Optional[float] = None, max: Optional[float] = None):
        loss_energy = true - energies.sum(-1, True)
        return loss_energy.clamp(min, max).mean()

    # ──────────────────────────────────────────────────────────────────────
    # Training / validation / test steps (prints epoch‑level summaries)
    # ──────────────────────────────────────────────────────────────────────
    def training_step(self, batch: Dict[str, Batch]):
        loss_total = torch.tensor(0.0, device=self.device)
        for task, sample in batch.items():
            task_cfg = self.config.data[task]
            energies, dvdw_radii = self(sample)
            loss_dvdw = self.loss_dvdw(dvdw_radii)
            if task_cfg.objective == "regression":
                loss_energy = self.loss_regression(energies, sample.y)
            elif task_cfg.objective == "augment":
                loss_energy = self.loss_augment(energies, sample.y, *task_cfg.loss_range)
            else:
                raise NotImplementedError
            loss_total += loss_energy * task_cfg.loss_ratio + loss_dvdw * self.config.run.loss_dvdw_ratio
            self.losses["energy"][task].append(loss_energy.item())
            self.losses["dvdw"][task].append(loss_dvdw.item())
            for key, pred, true in zip(sample.key, energies, sample.y):
                self.predictions[task][key] = pred.tolist()
                self.labels[task][key] = true.item()
        return loss_total

    def validation_step(self, batch: Dict[str, Batch]):
        return self.training_step(batch)

    def test_step(self, batch: Batch):
        sample = batch
        task = next(iter(self.config.data))
        energies, dvdw_radii = self(sample)
        loss_energy = self.loss_regression(energies, sample.y)
        loss_dvdw = self.loss_dvdw(dvdw_radii)
        self.losses["energy"][task].append(loss_energy.item())
        self.losses["dvdw"][task].append(loss_dvdw.item())
        for key, pred, true in zip(sample.key, energies, sample.y):
            self.predictions[task][key] = pred.tolist()
            self.labels[task][key] = true.item()

    def predict_step(self, batch: Batch):
        sample = batch
        task = next(iter(self.config.data))
        energies, _ = self(sample)
        for key, pred in zip(sample.key, energies):
            self.predictions[task][key] = pred.tolist()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.config.run.lr, weight_decay=self.config.run.weight_decay)

    def reset_log(self):
        self.losses = defaultdict(lambda: defaultdict(list))
        self.predictions = defaultdict(dict)
        self.labels = defaultdict(dict)
