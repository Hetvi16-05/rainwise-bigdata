import torch
import torch.nn as nn

class RainfallMLP(nn.Module):
    """A Multi-Layer Perceptron for rainfall regression."""
    def __init__(self, input_dim):
        super(RainfallMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output is continuous mm of rain
        )

    def forward(self, x):
        return self.network(x)

class FloodDNN(nn.Module):
    """A Deep Neural Network for flood risk classification."""
    def __init__(self, input_dim):
        super(FloodDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1),
            nn.Sigmoid() # Output is probability 0-1
        )

    def forward(self, x):
        return self.network(x)

class FloodLSTM(nn.Module):
    """An LSTM model for sequence-based prediction."""
    def __init__(self, input_dim, hidden_dim, num_layers, output_type='classification'):
        super(FloodLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_type = output_type
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Seq_len, Features)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        
        if self.output_type == 'classification':
            return self.sigmoid(out)
        return out

class TabTransformer(nn.Module):
    """
    A Transformer-based architecture for Tabular Data.
    Projects each feature into an embedding space and uses Self-Attention 
     to learn inter-feature dependencies.
    """
    def __init__(self, input_dim, embed_dim=32, depth=2, heads=4, mlp_hidden=[128, 64], dropout=0.2):
        super(TabTransformer, self).__init__()
        
        # 1. Feature Projections (Embedding continuous features)
        self.projections = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(input_dim)
        ])
        
        # 2. Transformer Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=heads, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 3. Final MLP Head
        mlp_input_dim = input_dim * embed_dim
        layers = []
        curr_dim = mlp_input_dim
        for h in mlp_hidden:
            layers.extend([
                nn.Linear(curr_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout)
            ])
            curr_dim = h
            
        layers.append(nn.Linear(curr_dim, 1)) # Output for Regression
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, input_dim)
        
        # Project each column to embed_dim
        # Reshape to (Batch, input_dim, 1) for column-wise projection
        projected = []
        for i, proj in enumerate(self.projections):
            col_data = x[:, i:i+1] # (Batch, 1)
            projected.append(proj(col_data)) # (Batch, embed_dim)
            
        # Stack to (Batch, input_dim, embed_dim)
        x_trans = torch.stack(projected, dim=1)
        
        # Pass through Transformer
        x_trans = self.transformer(x_trans)
        
        # Flatten for MLP
        x_flat = x_trans.view(x_trans.size(0), -1)
        
        return self.mlp(x_flat)
