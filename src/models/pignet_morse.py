# import torch
# from omegaconf import DictConfig
# from torch.nn import Parameter, ReLU, Sigmoid
# from torch_geometric.data import Batch
# from torch_geometric.nn import Linear, Sequential
# from torch_scatter import scatter

# from . import physics
# from .pignet import PIGNet


# class PIGNetMorse(PIGNet):
#     def __init__(
#         self,
#         config: DictConfig,
#         in_features: int = -1,
#         **kwargs,
#     ):
#         super().__init__(config=config)
#         self.reset_log()
#         self.config = config
#         dim_gnn = config.model.dim_gnn
#         dim_mlp = config.model.dim_mlp

#         self.embed = Linear(in_features, dim_gnn, bias=False)

#         self.nn_vdw_epsilon = Sequential(
#             "x",
#             [
#                 (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
#                 ReLU(),
#                 Linear(dim_mlp, 1),
#                 Sigmoid(),
#             ],
#         )
#         self.nn_vdw_width = Sequential(
#             "x",
#             [
#                 (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
#                 ReLU(),
#                 Linear(dim_mlp, 1),
#                 Sigmoid(),
#             ],
#         )
#         self.nn_vdw_radius = Sequential(
#             "x",
#             [
#                 (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
#                 ReLU(),
#                 Linear(dim_mlp, 1),
#                 ReLU(),
#             ],
#         )

#         self.hbond_coeff = Parameter(torch.tensor([0.714]))
#         self.metal_ligand_coeff = Parameter(torch.tensor([1.0]))
#         self.hydrophobic_coeff = Parameter(torch.tensor([0.216]))
#         self.rotor_coeff = Parameter(torch.tensor([0.102]))
#         self.ionic_coeff = Parameter(torch.tensor([1.0]))  # NOT USED

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
#         energies_pairs = torch.zeros(5, D.numel()).to(self.device)

#         # vdW energy minima (well depths): (pairs,)
#         vdw_epsilon = self.nn_vdw_epsilon(x_cat).squeeze(-1)

#         # Scale the minima as done in AutoDock Vina.
#         vdw_epsilon = (
#             vdw_epsilon * (cfg.vdw_epsilon_scale[1] - cfg.vdw_epsilon_scale[0])
#             + cfg.vdw_epsilon_scale[0]
#         )

#         vdw_width = self.nn_vdw_width(x_cat).squeeze(-1)
#         vdw_width = (
#             vdw_width * (cfg.vdw_width_scale[1] - cfg.vdw_width_scale[0])
#             + cfg.vdw_width_scale[0]
#         )
#         energies_pairs[0] = physics.morse_potential(
#             D,
#             R,
#             vdw_epsilon,
#             vdw_width,
#             cfg.short_range_A,
#         )

#         minima_hbond = -(self.hbond_coeff**2)
#         minima_metal_ligand = -(self.metal_ligand_coeff**2)
#         minima_hydrophobic = -(self.hydrophobic_coeff**2)
#         energies_pairs[1] = physics.linear_potential(
#             D, R, minima_hbond, *cfg.hydrogen_bond_cutoffs
#         )
#         energies_pairs[2] = physics.linear_potential(
#             D, R, minima_metal_ligand, *cfg.metal_ligand_cutoffs
#         )
#         energies_pairs[3] = physics.linear_potential(
#             D, R, minima_hydrophobic, *cfg.hydrophobic_cutoffs
#         )

#         # Interaction masks according to atom types: (energy_types, pairs)
#         masks = physics.interaction_masks(
#             sample.is_metal,
#             sample.is_h_donor,
#             sample.is_h_acceptor,
#             sample.is_hydrophobic,
#             edge_index_i,
#             True,
#         )

#         # ionic interaction
#         energies_pairs[4] = torch.zeros_like(energies_pairs[4])
#         if cfg.get("include_ionic", False):
#             # Note the sign of `minima_ionic`
#             minima_ionic = self.ionic_coeff**2 * (
#                 sample.atom_charges[edge_index_i[0]]
#                 * sample.atom_charges[edge_index_i[1]]
#             )
#             energies_pairs[4] = physics.linear_potential(
#                 D, R, minima_ionic, *cfg.ionic_cutoffs
#             )

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

"""PIGNetMorse – ProtBERT + MPNN backbone with a Morse‑potential vdW term.

This class re‑uses the upgraded PIGNet (ProtBERT residue embeddings, MPNN
message passing, automatic protein‑sequence inference) and simply swaps the
vdW interaction from Lennard‑Jones to a Morse potential with a learnable width.

Verbosity matches the parent class: every major step prints to stdout.
"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import Parameter, ReLU, Sigmoid, Tanh
from torch_geometric.data import Batch
from torch_geometric.nn import Linear, Sequential
from torch_scatter import scatter

from . import physics
from .pignet import PIGNet  # parent class with ProtBERT + MPNN


class PIGNetMorse(PIGNet):
    """PIGNet variant that uses a Morse potential for vdW interactions."""

    def __init__(self, config: DictConfig, in_features: int = -1, **kwargs):
        super().__init__(config=config)
        lig_in_features = config.model.ligand_in_features
        dim_gnn = config.model.dim_gnn
        dim_mlp = config.model.dim_mlp
        self.dim_gnn = dim_gnn

        print("[PIGNetMorse] Replacing vdW head with Morse‑potential parametrisation")

        # vdW parameter predictors (overwrite the parent nn_vdw_epsilon)
        self.nn_vdw_epsilon = Sequential(
            "x",
            [
                (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
                ReLU(),
                Linear(dim_mlp, 1),
                Sigmoid(),
            ],
        )
        self.nn_vdw_width = Sequential(
            "x",
            [
                (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
                ReLU(),
                Linear(dim_mlp, 1),
                Sigmoid(),
            ],
        )
        # Parent already defines nn_dvdw (radius deviation)

        # Re‑tuned coefficients (as in original Morse version)
        self.hbond_coeff = Parameter(torch.tensor([0.714]))
        self.metal_ligand_coeff = Parameter(torch.tensor([1.0]))
        self.hydrophobic_coeff = Parameter(torch.tensor([0.216]))
        self.rotor_coeff = Parameter(torch.tensor([0.102]))
        self.ionic_coeff = Parameter(torch.tensor([1.0]))

    # ------------------------------------------------------------------
    # Forward – identical to parent until vdW term, then Morse potential
    # ------------------------------------------------------------------
    def forward(self, sample: Batch):
        cfg = self.config.model
        device = self.device

        print("[PIGNetMorse] Forward pass started – Morse potential variant")

        # Ensure protein attributes & build embeddings (re‑use parent helpers)
        self._ensure_protein_attributes(sample)
        lig_mask, prot_mask = sample.is_ligand, ~sample.is_ligand
        if self.lig_embed is None:
            in_dim = sample.x.size(1)          # detect at runtime
            self.lig_embed = Linear(in_dim, self.dim_gnn, bias=False).to(self.device)
            print(f"[PIGNet] Created ligand embedder on-the-fly ({in_dim} → {self.dim_gnn})")
        x_lig = self.lig_embed(sample.x[lig_mask])
        residue_repr = self.prot_proj(self._encode_protein(sample.prot_seq, device))
        x_prot = residue_repr[sample.residue_index]
        x = torch.zeros(sample.x.size(0), residue_repr.size(1), device=device)
        x[lig_mask] = x_lig
        x[prot_mask] = x_prot

        # Message passing
        x = self.conv(x, sample.edge_index, sample.edge_index_c)

        # Interaction edges & distances
        edge_index_i = physics.interaction_edges(sample.is_ligand, sample.batch)
        D = physics.distances(sample.pos, edge_index_i)
        _mask = (cfg.interaction_range[0] <= D) & (D <= cfg.interaction_range[1])
        edge_index_i = edge_index_i[:, _mask]
        D = D[_mask]
        x_cat = torch.cat((x[edge_index_i[0]], x[edge_index_i[1]]), -1)

        # Predict vdW parameters
        dvdw_radii = self.nn_dvdw(x_cat).view(-1) * cfg.dev_vdw_radii_coeff
        R = sample.vdw_radii[edge_index_i[0]] + sample.vdw_radii[edge_index_i[1]] + dvdw_radii
        vdw_epsilon = self.nn_vdw_epsilon(x_cat).squeeze(-1)
        vdw_epsilon = vdw_epsilon * (cfg.vdw_epsilon_scale[1] - cfg.vdw_epsilon_scale[0]) + cfg.vdw_epsilon_scale[0]
        vdw_width = self.nn_vdw_width(x_cat).squeeze(-1)
        vdw_width = vdw_width * (cfg.vdw_width_scale[1] - cfg.vdw_width_scale[0]) + cfg.vdw_width_scale[0]

        # Prepare energy tensor (5 types to align with original order)
        energies_pairs = torch.zeros(5, D.numel(), device=device)
        energies_pairs[0] = physics.morse_potential(D, R, vdw_epsilon, vdw_width, cfg.short_range_A)

        # Other interaction types (same as parent but coeffs differ)
        minima_hbond = -(self.hbond_coeff ** 2)
        minima_metal = -(self.metal_ligand_coeff ** 2)
        minima_hydrophobic = -(self.hydrophobic_coeff ** 2)
        energies_pairs[1] = physics.linear_potential(D, R, minima_hbond, *cfg.hydrogen_bond_cutoffs)
        energies_pairs[2] = physics.linear_potential(D, R, minima_metal, *cfg.metal_ligand_cutoffs)
        energies_pairs[3] = physics.linear_potential(D, R, minima_hydrophobic, *cfg.hydrophobic_cutoffs)

        # Ionic (optional)
        energies_pairs[4] = torch.zeros_like(energies_pairs[4])
        if cfg.get("include_ionic", False):
            minima_ionic = self.ionic_coeff ** 2 * (sample.atom_charges[edge_index_i[0]] * sample.atom_charges[edge_index_i[1]])
            energies_pairs[4] = physics.linear_potential(D, R, minima_ionic, *cfg.ionic_cutoffs)

        # Masks & aggregation
        masks = physics.interaction_masks(sample.is_metal, sample.is_h_donor, sample.is_h_acceptor, sample.is_hydrophobic, edge_index_i, True)
        energies_pairs *= masks
        energies = scatter(energies_pairs, sample.batch[edge_index_i[0]])
        energies = energies.t().contiguous()
        if cfg.rotor_penalty:
            penalty = 1 + self.rotor_coeff ** 2 * sample.rotor
            energies = energies / penalty

        print("[PIGNetMorse] Energies computed – shape:", energies.shape)
        return energies, dvdw_radii

    # Loss functions, training_step, etc. inherit unchanged from PIGNet
