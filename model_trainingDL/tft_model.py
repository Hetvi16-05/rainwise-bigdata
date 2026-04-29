"""
RAINWISE — Temporal Fusion Transformer (TFT)
============================================
Architecture based on: "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting" (Lim et al., 2021)

Adapted for the Gujarat spatio-temporal rainfall dataset:
  - Static covariates : lat, lon, elevation_m, distance_to_river_m
  - Temporal inputs   : sin_doy, cos_doy, month_sin, month_cos,
                        rain3_mm, rain7_mm, precip_mm
  - Target            : rain_mm  (regression, mm/day)

Key components:
  1. Gated Residual Network  (GRN)  — learnable skip + suppression
  2. Variable Selection Network (VSN) — per-feature importance weights
  3. Static Covariate Encoder  — geography context vector
  4. LSTM Encoder              — local sequential processing
  5. Temporal Self-Attention   — long-range dependencies
  6. Quantile output head      — P10 / P50 / P90 predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Gated Residual Network (GRN)
# ─────────────────────────────────────────────
class GatedResidualNetwork(nn.Module):
    """
    Core building block of TFT.
    Allows the network to suppress irrelevant information via a sigmoid gate.
    Optional context vector (c) injects static covariate information.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1,
                 context_dim=None):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim

        # Main path
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Optional context injection
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_proj = None

        # Gating (GLU — Gated Linear Unit)
        self.gate = nn.Linear(output_dim, output_dim)

        # Skip connection projection if dims differ
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.skip_proj = None

        self.norm    = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.elu     = nn.ELU()

    def forward(self, x, context=None):
        # Skip connection
        residual = self.skip_proj(x) if self.skip_proj is not None else x

        # Main path: ELU activation
        h = self.fc1(x)
        if context is not None and self.context_proj is not None:
            h = h + self.context_proj(context)
        h = self.elu(h)
        h = self.dropout(h)
        h = self.fc2(h)

        # Gating
        g = torch.sigmoid(self.gate(h))
        h = g * h  # element-wise suppression

        # Residual + LayerNorm
        return self.norm(h + residual)


# ─────────────────────────────────────────────
# 2. Variable Selection Network (VSN)
# ─────────────────────────────────────────────
class VariableSelectionNetwork(nn.Module):
    """
    Learns WHICH input features matter most at each time step (or globally
    for static inputs).
    Outputs:
      - transformed: weighted sum of per-variable GRN outputs  (B, hidden)
      - weights:     softmax importances per variable           (B, num_vars)
    """
    def __init__(self, num_vars, input_dim, hidden_dim, dropout=0.1,
                 context_dim=None):
        super().__init__()
        self.num_vars = num_vars

        # One GRN per input variable
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim,
                                 dropout, context_dim)
            for _ in range(num_vars)
        ])

        # Softmax weight selector (over flattened all-variable concat)
        self.weight_grn = GatedResidualNetwork(
            num_vars * input_dim, hidden_dim, num_vars,
            dropout, context_dim
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_list, context=None):
        """
        x_list: list of (B, input_dim) tensors — one per variable
        """
        # Per-variable GRN transforms
        var_outputs = [grn(x, context)
                       for grn, x in zip(self.var_grns, x_list)]  # each: (B, hidden)

        # Compute selection weights from concatenated raw inputs
        x_concat = torch.cat(x_list, dim=-1)          # (B, num_vars * input_dim)
        weights   = self.softmax(
            self.weight_grn(x_concat, context)
        )                                               # (B, num_vars)

        # Weighted sum of variable outputs
        stacked  = torch.stack(var_outputs, dim=-1)    # (B, hidden, num_vars)
        selected = (stacked * weights.unsqueeze(1)).sum(-1)  # (B, hidden)

        return selected, weights


# ─────────────────────────────────────────────
# 3. Temporal Self-Attention (Multi-Head)
# ─────────────────────────────────────────────
class TemporalSelfAttention(nn.Module):
    """
    Interpretable multi-head attention that attends over TIME steps.
    Shared V weights (as in original TFT paper) for interpretability.
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, seq_len, hidden_dim)"""
        attn_out, attn_weights = self.attn(x, x, x)
        attn_out = self.dropout(attn_out)
        return self.norm(attn_out + x), attn_weights  # residual


# ─────────────────────────────────────────────
# 4. Full TFT Model
# ─────────────────────────────────────────────
class TemporalFusionTransformer(nn.Module):
    """
    Full TFT for Gujarat rainfall regression.

    Input format:
      static_inputs  : (B, num_static)   — lat, lon, elevation, river_dist
      temporal_inputs: (B, seq_len, num_temporal) — sin/cos, rain lags, precip

    Output:
      quantiles: (B, 3) for P10, P50, P90  [mm/day]
    """
    def __init__(
        self,
        num_static_vars:   int   = 4,
        num_temporal_vars: int   = 7,
        hidden_dim:        int   = 64,
        num_heads:         int   = 4,
        num_lstm_layers:   int   = 2,
        dropout:           float = 0.1,
        seq_len:           int   = 14,
        num_quantiles:     int   = 3,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.seq_len     = seq_len

        # ── (a) Input embedding: project each scalar feature to hidden_dim ──
        self.static_input_proj = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_static_vars)
        ])
        self.temporal_input_proj = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_temporal_vars)
        ])

        # ── (b) Static Variable Selection + Covariate Encoder ──
        self.static_vsn = VariableSelectionNetwork(
            num_vars=num_static_vars,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        # Static context vectors (injected into temporal VSN & GRNs)
        self.static_context_h  = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.static_context_c  = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.static_context_e  = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)

        # ── (c) Temporal Variable Selection (context-aware) ──
        self.temporal_vsn = VariableSelectionNetwork(
            num_vars=num_temporal_vars,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            context_dim=hidden_dim  # injected from static encoder
        )

        # ── (d) LSTM Encoder (local temporal processing) ──
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )

        # ── (e) Post-LSTM gate ──
        self.post_lstm_gate = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )

        # ── (f) Temporal Self-Attention ──
        self.temporal_attention = TemporalSelfAttention(hidden_dim, num_heads, dropout)

        # ── (g) Post-attention GRN ──
        self.post_attn_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )

        # ── (h) Output: Quantile head (P10, P50, P90) ──
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_quantiles)  # [P10, P50, P90]
        )

    def forward(self, static_inputs, temporal_inputs):
        """
        static_inputs  : (B, num_static_vars)
        temporal_inputs: (B, seq_len, num_temporal_vars)
        Returns        : (B, 3) — quantile predictions [P10, P50, P90]
        """
        B = static_inputs.size(0)

        # ── Step 1: Project static scalars → embeddings ──
        static_embedded = [
            self.static_input_proj[i](static_inputs[:, i:i+1])
            for i in range(len(self.static_input_proj))
        ]  # list of (B, hidden_dim)

        # ── Step 2: Static VSN ──
        static_vec, static_weights = self.static_vsn(static_embedded)
        # static_vec: (B, hidden_dim)

        # ── Step 3: Build static context vectors ──
        ctx_h = self.static_context_h(static_vec)  # LSTM hidden init
        ctx_c = self.static_context_c(static_vec)  # LSTM cell init
        ctx_e = self.static_context_e(static_vec)  # Temporal VSN context

        # ── Step 4: Temporal VSN over the sequence ──
        # temporal_inputs: (B, seq_len, num_temporal_vars)
        temporal_embedded_seq = []
        for t in range(temporal_inputs.size(1)):  # iterate over time steps
            step_feats = [
                self.temporal_input_proj[i](temporal_inputs[:, t, i:i+1])
                for i in range(len(self.temporal_input_proj))
            ]  # list of (B, hidden_dim)
            step_selected, _ = self.temporal_vsn(step_feats, context=ctx_e)
            temporal_embedded_seq.append(step_selected)

        # Stack → (B, seq_len, hidden_dim)
        temporal_seq = torch.stack(temporal_embedded_seq, dim=1)

        # ── Step 5: LSTM Encoder (init h,c from static context) ──
        # ctx_h / ctx_c: (B, hidden_dim) → expand to (num_layers, B, hidden_dim)
        num_layers = self.lstm_encoder.num_layers
        h0 = ctx_h.unsqueeze(0).repeat(num_layers, 1, 1)
        c0 = ctx_c.unsqueeze(0).repeat(num_layers, 1, 1)

        lstm_out, _ = self.lstm_encoder(temporal_seq, (h0, c0))
        # lstm_out: (B, seq_len, hidden_dim)

        # ── Step 6: Post-LSTM gate + residual ──
        lstm_out = self.post_lstm_gate(
            lstm_out.reshape(-1, self.hidden_dim)
        ).reshape(B, -1, self.hidden_dim)

        # ── Step 7: Temporal Self-Attention ──
        attn_out, attn_weights = self.temporal_attention(lstm_out)

        # ── Step 8: Post-attention GRN ──
        attn_out = self.post_attn_grn(
            attn_out.reshape(-1, self.hidden_dim)
        ).reshape(B, -1, self.hidden_dim)

        # ── Step 9: Use LAST time step for prediction ──
        final = attn_out[:, -1, :]  # (B, hidden_dim)

        # ── Step 10: Quantile output ──
        quantiles = self.quantile_head(final)  # (B, 3)
        return quantiles, static_weights


# ─────────────────────────────────────────────
# Quantile Loss (Pinball Loss)
# ─────────────────────────────────────────────
class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.
    quantiles: list of quantile levels, e.g. [0.1, 0.5, 0.9]
    """
    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)

    def forward(self, preds, target):
        """
        preds : (B, num_quantiles)
        target: (B,) or (B,1)
        """
        target = target.view(-1, 1)                    # (B, 1)
        q      = self.quantiles.to(preds.device)       # (num_quantiles,)
        errors = target - preds                        # (B, num_quantiles)
        loss   = torch.max(q * errors, (q - 1) * errors)
        return loss.mean()


if __name__ == "__main__":
    # Quick smoke test
    B, SEQ, S_VARS, T_VARS = 8, 14, 4, 7
    static  = torch.randn(B, S_VARS)
    temporal = torch.randn(B, SEQ, T_VARS)

    model = TemporalFusionTransformer(
        num_static_vars=S_VARS,
        num_temporal_vars=T_VARS,
        hidden_dim=64,
        num_heads=4,
        num_lstm_layers=2,
        seq_len=SEQ,
    )
    out, sw = model(static, temporal)
    print(f"✅ TFT smoke test passed!")
    print(f"   Output shape (quantiles): {out.shape}")   # (8, 3)
    print(f"   Static weights shape    : {sw.shape}")    # (8, 4)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters        : {total:,}")
