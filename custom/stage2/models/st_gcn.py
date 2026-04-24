import torch
import torch.nn as nn

class AdaptiveGCNEncoder(nn.Module):
    """
    H6-B: Adaptive Spatial Graph Convolutional Network.
    Learns an implicit adjacency matrix for the 25 joints to extract physical kinematic linkages.
    """
    def __init__(self, in_channels=3, hidden_dim=128, out_dim=256, num_nodes=25, num_layers=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.layers = nn.ModuleList()
        self.adjs = nn.ParameterList()
        
        dims = [in_channels] + [hidden_dim]*(num_layers-1) + [out_dim]
        for i in range(num_layers):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            # Learned adjacency matrix for each layer, initialized uniformly
            self.adjs.append(nn.Parameter(torch.ones(num_nodes, num_nodes) / num_nodes))
            
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, V, C)
        for i, (layer, adj) in enumerate(zip(self.layers, self.adjs)):
            # Adaptive GCN: A * X * W
            x = torch.matmul(adj, x)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return self.norm(x)
