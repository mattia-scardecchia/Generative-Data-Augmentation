import torch.nn as nn

from ..networks.blocks import ConvNeXtBlock, ResNetBlock


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dims: list[int],
        out_channels: int,
        block_type: str,
        downsample: list[int | None],
        **block_kwargs,
    ):
        """
        :param in_channels: input channels
        :param hidden_dims: list of hidden dimensions. Each element is the width of a residual block.
        :param out_channels: bottleneck channels
        :param block_type: resnet or convnext
        :param downsample: list of downsample factors. After each block, the spatial dimensions are reduced by this factor.
        """
        super().__init__()
        match block_type.lower():
            case "resnet":
                block_class = ResNetBlock
            case "convnext":
                block_class = ConvNeXtBlock
            case _:
                raise ValueError(f"Unknown block type: {block_type}")

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.block_type = block_type
        self.downsample = downsample

        self.input_layer = nn.Conv2d(
            in_channels, hidden_dims[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList()
        for i, (out_channels, ds) in enumerate(zip(hidden_dims, downsample)):
            self.blocks.append(block_class(out_channels, **block_kwargs))
            if i < len(hidden_dims) - 1:
                self.blocks.append(
                    nn.Conv2d(
                        out_channels, hidden_dims[i + 1], kernel_size=3, padding=1
                    )
                )
            if ds:
                self.blocks.append(nn.MaxPool2d(kernel_size=ds, stride=ds))

        self.output_layer = nn.Conv2d(
            hidden_dims[-1], self.out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dims: list[int],
        out_channels: int,
        block_type: str,
        upsample: list[int | None],
        **block_kwargs,
    ):
        super().__init__()
        match block_type.lower():
            case "resnet":
                block_class = ResNetBlock
            case "convnext":
                block_class = ConvNeXtBlock
            case _:
                raise ValueError(f"Unknown block type: {block_type}")
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.block_type = block_type
        self.upsample = upsample

        self.input_layer = nn.Conv2d(
            in_channels, hidden_dims[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList()
        for i, (out_channels, us) in enumerate(zip(hidden_dims, upsample)):
            self.blocks.append(block_class(out_channels, **block_kwargs))
            if i < len(hidden_dims) - 1:
                self.blocks.append(
                    nn.Conv2d(
                        out_channels, hidden_dims[i + 1], kernel_size=3, padding=1
                    )
                )
            if us:
                self.blocks.append(nn.Upsample(scale_factor=us))

        self.output_layer = nn.Conv2d(
            hidden_dims[-1], self.out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x
