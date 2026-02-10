import torch
import torch.nn as nn
import torch.nn.functional as F #Used for ReLu

class Depthwise_Pointwise_Conv(nn.Module): 

    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: tuple = (3, 3), 
        stride: tuple = (1, 1),
        padding: tuple = (1, 1), 
    ):
        
        super().__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels) 

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=stride,
            padding=(0,0),
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
            x = self.depthwise(x)
            x = self.bn1(x)
            x = F.relu(x)

            x = self.pointwise(x)
            x = self.bn2(x)
            x= F.relu(x) 

            return x 

class DSCNN(nn.Module): 

    def __init__(
        self, 
        n_classes: int = 3,         #number of keywords wanted for detection 
        n_mels: int = 40, 
        first_conv_filters: int = 64,
        first_conv_kernel: tuple = (10, 4),
        first_conv_stride: tuple = (2, 2),
        n_ds_blocks: int = 4,
        ds_filters: int = 64,
        ds_kernel: tuple = (3, 3),
        ds_stride: tuple = (1, 1),
    ):
        
        super().__init__()

        self.n_classes = n_classes
        self.n_mels = n_mels

        first_conv_padding = (
            (first_conv_kernel[0] - 1) // 2,
            (first_conv_kernel[1] - 1) // 2,
        )

        self.first_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv_filters,
            kernel_size=first_conv_kernel,
            stride=first_conv_stride,
            padding=first_conv_padding,
            bias=False,
        )
        self.first_bn = nn.BatchNorm2d(first_conv_filters)

        ds_padding = ((ds_kernel[0] - 1) // 2, (ds_kernel[1] - 1) // 2)

        #Like a generate block 
        self.ds_blocks = nn.ModuleList()  #create list 
        for i in range(n_ds_blocks): 
            in_ch = first_conv_filters if i == 0 else ds_filters
            self.ds_blocks.append(
                Depthwise_Pointwise_Conv(
                    in_channels=in_ch,
                    out_channels=ds_filters,
                    kernel_size= ds_kernel,
                    stride=ds_stride,
                    padding=ds_padding,
                )    
            )

        #Global Average Pooling 
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        #Final fully connected layer 
        self.fc = nn.Linear(ds_filters, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = F.relu(x)

        for ds_block in self.ds_blocks:
            x = ds_block(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def create_model(config: dict) -> DSCNN:

    model_cfg = config.get("model", {})
    preproc_cfg = config.get("preprocessing", {})

    # Get first conv settings
    first_conv_cfg = model_cfg.get("first_conv", {})
    first_conv_kernel = first_conv_cfg.get("kernel_size", [10, 4])
    first_conv_stride = first_conv_cfg.get("stride", [2, 2])
    first_conv_filters = first_conv_cfg.get("filters", 64)

    # Get DS block settings
    ds_cfg = model_cfg.get("ds_blocks", {})

    return DSCNN(
        n_classes=model_cfg.get("n_classes", 3),
        n_mels=preproc_cfg.get("n_mels", 40),
        first_conv_filters=first_conv_filters,
        first_conv_kernel=tuple(first_conv_kernel),
        first_conv_stride=tuple(first_conv_stride),
        n_ds_blocks=ds_cfg.get("n_blocks", 4),
        ds_filters=ds_cfg.get("filters", 64),
        ds_kernel=tuple(ds_cfg.get("kernel_size", [3, 3])),
        ds_stride=tuple(ds_cfg.get("stride", [1, 1])),
    )