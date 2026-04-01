import torch
import torch.nn as nn

from src.models.building_blocks import EGNNLayer, EvoformerBlock, FeatureProjection



# =============================================================================
# FULL MODEL: Combining Everything
# =============================================================================

class ProteinBindingGNN(nn.Module):
    """
    Full model for protein binding site prediction.

    Architecture:
    1. Feature projection (ESM2 + biochemical features -> hidden dim)
    2. EGNN layers (equivariant message passing, updates coords)
    3. Evoformer blocks (attention + FFN, captures long-range dependencies)
    4. Output head (per-residue binding probability)

    Why this combination?
    - EGNN: handles 3D structure, maintains rotation equivariance
    - Evoformer: captures sequence-level patterns and long-range contacts
    - The EGNN processes local geometry, Evoformer processes global context
    """

    def __init__(
        self,
        node_input_dim: int = 1304,   # ESM2(1280) + one-hot(20) + biochem(4)
        edge_input_dim: int = 4,       # weight, type, contact, coulomb
        hidden_dim: int = 256,         # H
        num_egnn_layers: int = 3,      # Number of equivariant layers
        num_evoformer_blocks: int = 4, # Number of attention blocks
        num_heads: int = 8,            # Attention heads
        dropout: float = 0.1,
        update_coords: bool = True,    # Whether to update 3D positions
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # --- Feature Projections ---
        self.node_proj = FeatureProjection(node_input_dim, hidden_dim, dropout)
        self.edge_proj = FeatureProjection(edge_input_dim, hidden_dim, dropout)
        

        # --- EGNN Layers (Equivariant) ---
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=hidden_dim,
                edge_dim=hidden_dim,  # After projection
                update_coords=update_coords,
                dropout=dropout
            )
            for _ in range(num_egnn_layers)
        ])

        # --- Evoformer Blocks (Attention) ---
        self.evoformer_blocks = nn.ModuleList([
            EvoformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=hidden_dim * 4,
                dropout=dropout,
                use_edge_bias=True
            )
            for _ in range(num_evoformer_blocks)
        ])

        # --- Output Head ---
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # Single output per residue
        )

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: PyTorch Geometric Data object with:
                - x: (N, 1304) node features
                - pos: (N, 3) Cb coordinates
                - edge_index: (2, E) edges
                - edge_attr: (E, 4) edge features

        Returns:
            (N,) binding probability per residue
        """
        # Unpack data (like accessing dict in TensorFlow)
        h = data.x              # (N, 1304)
        pos = data.pos          # (N, 3)
        edge_index = data.edge_index  # (2, E)
        edge_attr = data.edge_attr    # (E, 4)

        # --- Project to hidden dimension ---
        h = self.node_proj(h)           # (N, H)
        edge_h = self.edge_proj(edge_attr)  # (E, H)

        # --- EGNN Layers ---
        # These update both node features AND coordinates
        for egnn in self.egnn_layers:
            h, pos = egnn(h, pos, edge_index, edge_h)

        # --- Evoformer Blocks ---
        # These only update node features (attention-based)
        for evoformer in self.evoformer_blocks:
            h = evoformer(h, edge_index, edge_h)

        # --- Output ---
        h = self.final_norm(h)
        logits = self.output_head(h).squeeze(-1)  # (N,)

        # Sigmoid for probability (use BCEWithLogitsLoss for training)
        # During training, return logits; for inference, apply sigmoid
        return logits

    def predict(self, data) -> torch.Tensor:
        """Inference mode: returns probabilities."""
        logits = self.forward(data)
        return torch.sigmoid(logits)