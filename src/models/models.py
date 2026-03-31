

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional, Tuple


# =============================================================================
# BUILDING BLOCK 1: Feature Projection (like Dense layers in TF)
# =============================================================================

class FeatureProjection(nn.Module):
    """
    Projects raw features to hidden dimension.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__() 

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'gelu':
            x = F.gelu(x)
        x = self.dropout(x)
        return x


# =============================================================================
# BUILDING BLOCK 2: E(3) Equivariant Graph Conv Layer
# =============================================================================

class EGNNLayer(MessagePassing):
    """
    E(n) Equivariant Graph Neural Network Layer.

    Key insight for equivariance:
    - Node features (h) are INVARIANT (don't change with rotation)
    - Positions (x) are EQUIVARIANT (rotate with the molecule)

    To maintain equivariance:
    - Messages use ||x_i - x_j|| (distance, invariant) NOT raw positions
    - Position updates use (x_i - x_j) * scalar (equivariant direction)

    In TensorFlow, you'd manually implement message passing.
    PyTorch Geometric's MessagePassing class handles the aggregation.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        update_coords: bool = True,
        dropout: float = 0.1
    ):
        # aggr='add' means: aggregate messages by summing (like tf.math.segment_sum)
        super().__init__(aggr='add')

        self.hidden_dim = hidden_dim
        self.update_coords = update_coords

        # Message MLP: takes [h_i, h_j, distance, edge_features]
        # Input: hidden_dim*2 + 1 + edge_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim),
            nn.SiLU(),  # Swish activation, popular in recent architectures
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Coordinate update MLP: outputs a scalar weight for direction vector
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1, bias=False),  # Output scalar
            )

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,           # (N, hidden_dim) node features
        pos: torch.Tensor,         # (N, 3) coordinates
        edge_index: torch.Tensor,  # (2, E) edges
        edge_attr: torch.Tensor,   # (E, edge_dim) edge features
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns updated (h', pos')
        """
        # Compute pairwise distances (INVARIANT - doesn't change with rotation)
        row, col = edge_index  # source, target indices

        # Relative position vector (EQUIVARIANT)
        rel_pos = pos[row] - pos[col]  # (E, 3)

        # Distance (INVARIANT) - safe for gradients
        dist = torch.norm(rel_pos, dim=-1, keepdim=True).clamp(min=1e-6)  # (E, 1)

        # propagate() calls message() then aggregate() - PyG handles this
        # It's like tf.math.unsorted_segment_sum but more convenient
        h_out, coord_update = self.propagate(
            edge_index,
            h=h,
            dist=dist,
            rel_pos=rel_pos,
            edge_attr=edge_attr,
        )

        # Residual connection (KEY for deep networks)
        h_out = h + self.dropout(h_out)
        h_out = self.norm(h_out)

        # Update coordinates (equivariant update)
        if self.update_coords and coord_update is not None:
            # Normalize by number of neighbors to stabilize
            # This maintains equivariance: rotating input rotates output
            pos = pos + coord_update

        return h_out, pos

    def message(
        self,
        h_i: torch.Tensor,      # (E, H) features of source nodes
        h_j: torch.Tensor,      # (E, H) features of target nodes
        dist: torch.Tensor,     # (E, 1) distances
        rel_pos: torch.Tensor,  # (E, 3) relative positions
        edge_attr: torch.Tensor # (E, edge_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute message from j to i.

        In TensorFlow you'd write:
            inputs = tf.concat([h_i, h_j, dist, edge_attr], axis=-1)
            messages = self.message_mlp(inputs)
        """
        # Concatenate all invariant features
        inputs = torch.cat([h_i, h_j, dist, edge_attr], dim=-1)

        # Message (invariant)
        m_ij = self.message_mlp(inputs)  # (E, H)

        # Coordinate message (equivariant)
        if self.update_coords:
            # Scalar weight from message, multiply by direction
            coord_weight = self.coord_mlp(m_ij)  # (E, 1)
            coord_msg = rel_pos * coord_weight   # (E, 3) - equivariant!
        else:
            coord_msg = torch.zeros_like(rel_pos)

        # Return both for aggregation
        # We store coord_msg in a way that propagate can handle
        self._coord_msg = coord_msg
        return m_ij

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, dim_size: int = None):
        """
        Aggregate messages to nodes.

        TensorFlow equivalent:
            tf.math.unsorted_segment_sum(inputs, index, num_segments)
        """
        # Aggregate node messages
        h_agg = super().aggregate(inputs, index, dim_size=dim_size)

        # Aggregate coordinate updates
        if hasattr(self, '_coord_msg'):
            coord_agg = super().aggregate(self._coord_msg, index, dim_size=dim_size)
        else:
            coord_agg = None

        return h_agg, coord_agg

    def update(self, agg_out, h):
        """
        Update node features with aggregated messages.

        agg_out: (h_agg, coord_agg) from aggregate()
        h: original node features
        """
        h_agg, coord_agg = agg_out

        # Combine original features with aggregated messages
        h_new = self.node_mlp(torch.cat([h, h_agg], dim=-1))

        return h_new, coord_agg


# =============================================================================
# BUILDING BLOCK 3: Evoformer-style Attention Block
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with edge bias.

    Like the attention in Evoformer/AlphaFold:
    - Q, K, V projections
    - Optional edge features as attention bias
    - Multiple heads for capturing different relationships

    TensorFlow equivalent: tf.keras.layers.MultiHeadAttention
    but we add edge bias capability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_edge_bias: bool = True
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k) for scaled dot-product

        # Q, K, V projections (like tf.keras.layers.Dense)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge feature projection for attention bias
        self.use_edge_bias = use_edge_bias
        if use_edge_bias:
            self.edge_proj = nn.Linear(hidden_dim, num_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,                    # (N, H) node features
        edge_index: torch.Tensor,           # (2, E) edge indices
        edge_attr: Optional[torch.Tensor] = None,  # (E, H) edge features
    ) -> torch.Tensor:
        """
        Self-attention over graph nodes with optional edge bias.

        For proteins: nodes=residues, edges=contacts/bonds
        Attention helps each residue "see" relevant context from other residues.
        """
        N = h.size(0)

        # Project to Q, K, V
        # Shape: (N, H) -> (N, num_heads, head_dim)
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(h).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(h).view(N, self.num_heads, self.head_dim)

        # Compute attention scores: Q @ K^T / sqrt(d)
        # Shape: (N, num_heads, head_dim) @ (N, head_dim, num_heads)
        # We use einsum for clarity (like tf.einsum)
        # 'nhd,mhd->nmh' means: for each head, compute dot product between all pairs
        attn_scores = torch.einsum('nhd,mhd->nmh', Q, K) * self.scale  # (N, N, num_heads)

        # Add edge bias if using (Evoformer-style)
        if self.use_edge_bias and edge_attr is not None:
            # Project edge features to per-head bias
            edge_bias = self.edge_proj(edge_attr)  # (E, num_heads)

            # Create sparse attention bias matrix
            row, col = edge_index
            # Only add bias where edges exist
            attn_scores[row, col] = attn_scores[row, col] + edge_bias

        # Softmax over source nodes (dim=1, the 'm' dimension)
        attn_weights = F.softmax(attn_scores, dim=1)  # (N, N, num_heads)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # 'nmh,mhd->nhd' means: weighted sum of V vectors
        out = torch.einsum('nmh,mhd->nhd', attn_weights, V)  # (N, num_heads, head_dim)

        # Reshape and project output
        out = out.reshape(N, self.hidden_dim)  # (N, H)
        out = self.out_proj(out)

        return out


class EvoformerBlock(nn.Module):
    """
    Single Evoformer-style block with:
    1. Multi-head self-attention + residual
    2. Feed-forward network + residual

    This is the "Transformer block" pattern used in:
    - Original Transformer (Vaswani et al.)
    - BERT, GPT
    - AlphaFold's Evoformer (with pair updates)

    The key insight: residual connections let gradients flow and
    prevent loss of information in deep networks.

    TensorFlow equivalent structure:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ffn_dim: int = None,  # Usually 4x hidden_dim
        dropout: float = 0.1,
        use_edge_bias: bool = True
    ):
        super().__init__()

        if ffn_dim is None:
            ffn_dim = hidden_dim * 4  # Standard transformer ratio

        # Pre-norm (more stable than post-norm for deep networks)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Multi-head attention
        self.attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_edge_bias=use_edge_bias
        )

        # Feed-forward network (2-layer MLP with expansion)
        # This is like: Dense(4H) -> GELU -> Dense(H)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),  # GELU is smoother than ReLU, used in modern transformers
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pre-LN Transformer block (more stable for deep networks).

        h: (N, H) node features
        Returns: (N, H) updated node features
        """
        # Self-attention with residual
        # Pattern: x = x + Sublayer(Norm(x))
        h = h + self.dropout(
            self.attention(
                self.norm1(h),
                edge_index,
                edge_attr
            )
        )

        # FFN with residual
        h = h + self.ffn(self.norm2(h))

        return h


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


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def create_model(
    hidden_dim: int = 256,
    num_egnn_layers: int = 3,
    num_evoformer_blocks: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1,
) -> ProteinBindingGNN:
    """Factory function to create model with default parameters."""
    return ProteinBindingGNN(
        node_input_dim=1304,
        edge_input_dim=4,
        hidden_dim=hidden_dim,
        num_egnn_layers=num_egnn_layers,
        num_evoformer_blocks=num_evoformer_blocks,
        num_heads=num_heads,
        dropout=dropout,
    )


