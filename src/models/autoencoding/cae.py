import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..networks.blocks import ResNetBlock, ConvNeXtBlock


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, block_type: str, hidden_dims: list[int], downsample: list[int | None], **block_kwargs):
        super().__init__()
        match block_type.lower():
            case "resnet":
                block_class = ResNetBlock
            case "convnext":
                block_class = ConvNeXtBlock
            case _:
                raise ValueError(f"Unknown block type: {block_type}")
        self.in_channels = in_channels
        self.block_type = block_type
        self.hidden_dims = hidden_dims
        self.downsample = downsample

        self.blocks = nn.ModuleList()
        for i, (out_channels, ds) in enumerate(zip(hidden_dims, downsample)):
            self.blocks.append(block_class(out_channels, **block_kwargs))
            if ds:
                self.blocks.append(nn.MaxPool2d(kernel_size=ds, stride=ds))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, in_channels, block_type: str, hidden_dims: list[int], upsample: list[int | None], **block_kwargs):
        super().__init__()
        match block_type.lower():
            case "resnet":
                block_class = ResNetBlock
            case "convnext":
                block_class = ConvNeXtBlock
            case _:
                raise ValueError(f"Unknown block type: {block_type}")
        self.in_channels = in_channels
        self.block_type = block_type
        self.hidden_dims = hidden_dims
        self.upsample = upsample

        self.blocks = nn.ModuleList()
        for i, (out_channels, us) in enumerate(zip(hidden_dims, upsample)):
            if us:
                self.blocks.append(nn.Upsample(scale_factor=us))
            self.blocks.append(block_class(out_channels, **block_kwargs))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x