import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..networks.blocks import MLPBlock


class MLPEncoder(nn.Module):
    def __init__(
        self, 
        in_features: int,
        hidden_dims: list[int],
        out_features: int,
        **block_kwargs
    ):
        """
        :param in_features: input dimension
        :param hidden_dims: list of hidden dimensions. Each element is the width of a residual block.
        :param out_features: bottleneck dimension
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.out_features = out_features
        
        # following residual block needs same input and output dimension
        self.input_proj = nn.Linear(in_features, hidden_dims[0])
        
        self.blocks = nn.ModuleList()
        for i, dim in enumerate(hidden_dims):
            self.blocks.append(MLPBlock(dim, **block_kwargs))
            if i < len(hidden_dims) - 1:
                self.blocks.append(nn.Linear(dim, hidden_dims[i+1]))
        
        self.output_proj = nn.Linear(hidden_dims[-1], out_features)
                
    def forward(self, x):
        x = x.flatten(1)
        x = self.input_proj(x)        
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        return x


class MLPDecoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int],
        out_features: int,
        **block_kwargs
    ):
        """
        :param in_features: bottleneck dimension
        :param hidden_dims: list of hidden dimensions. Each element is the width of a residual block.
        :param out_features: output dimension
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dims = hidden_dims

        self.input_proj = nn.Linear(in_features, hidden_dims[0])
        
        self.blocks = nn.ModuleList()
        for i, dim in enumerate(hidden_dims):
            self.blocks.append(MLPBlock(dim, **block_kwargs))
            if i < len(hidden_dims) - 1:
                self.blocks.append(nn.Linear(dim, hidden_dims[i+1]))
        
        self.output_proj = nn.Linear(hidden_dims[-1], out_features)
            
    def forward(self, x):
        x = x.flatten(1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)            
        x = self.output_proj(x)
        return x
