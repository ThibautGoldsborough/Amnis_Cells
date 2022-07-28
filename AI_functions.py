import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import torch
from torch import nn
import torchvision
from lightly.data import LightlyDataset
from lightly.data import ImageCollateFunction
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torch.utils.data import DataLoader



# custom dataset
class CellDataset():
    def __init__(self, images,labels,ID, transforms=None):
        self.X = images
        self.Y=  labels
        self.Z= ID
        self.transforms = transforms
        
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i]
        label=self.Y[i]
        ID=self.Z[i]
        data = np.asarray(data).astype(np.uint8)

        if self.transforms:
            data1 = self.transforms(data)
            data2 = self.transforms(data)
        
        return (data1,data2),np.array((np.argmax(label),ID))

class CellDataset_supervised():
    def __init__(self, images,labels,ID, transforms=None):
        self.X = images
        self.Y=  labels
        self.Z= ID
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i]
        label=self.Y[i]
        ID=self.Z[i]
        data = np.asarray(data).astype(np.uint8)
        label = np.asarray(label).astype(np.float32)
        
        if self.transforms:
            data = self.transforms(data)
        
        return data,label,ID

def get_mean_std(loader):
        #https://stackoverflow.com/questions/48818619/pytorch-how-do-the-means-and-stds-get-calculated-in-the-transfer-learning-tutor
        mean = 0.
        std = 0.
        for images, _,_ in loader:
            images=images #Is this 0 supposed to be here ?
            batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)

        mean /= len(loader.dataset)
        std /= len(loader.dataset)
        return mean, std



def data_generator(images,labels,names,mini,train_test_split = 0.8,batch_size = 100,sample=False):
    #Split data into training and test, just looking at first images now

    train_data1=images[:int(train_test_split*len(images))]
    test_data1=images[int(train_test_split*len(images)):]

    train_labels=labels[:int(train_test_split*len(images))]
    test_labels=labels[int(train_test_split*len(images)):]

    train_ID=names[:int(train_test_split*len(images))]
    test_ID=names[int(train_test_split*len(images)):]

    #Transform data into tensors, normalize images
    transform_basic = transforms.Compose(
        [transforms.ToPILImage(),
      #  transforms.Lambda(lambda x: np.array(x) -mini ),  #This was added on 12/07
        transforms.ToTensor()
    ])

    train_data_basic = CellDataset_supervised(train_data1,train_labels,train_ID, transform_basic)
    #Create DataLoaders
    train_loader_basic = DataLoader(train_data_basic, batch_size=100, shuffle=True)
    mean_loader,std_loader=get_mean_std(train_loader_basic)


    #Transform data into tensors, normalize images
    transform_train = transforms.Compose(
        [transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180,fill=mini),
        transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[mean_loader], std=[std_loader])  # for grayscale images

    ])

    #Transform data into tensors, normalize images
    transform_test = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[mean_loader], std=[std_loader])  # for grayscale images
    ])

    train_data = CellDataset_supervised(train_data1,train_labels,train_ID, transform_train)
    test_data = CellDataset_supervised(test_data1,test_labels,test_ID, transform_test)

    #Oversampling
    if sample==True:
        counts=np.bincount(np.argmax(train_labels,axis=1))
        labels_weights = 1. / counts
        weights = labels_weights[np.argmax(train_labels,axis=1)]
        sampler = WeightedRandomSampler(weights, len(weights))

        #Create DataLoaders
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,sampler=sampler)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
         #Can use these for plotting
        return train_data,train_data1,train_labels,train_ID,test_data,batch_size,mean_loader,std_loader,sampler
    else:
        return [train_data,train_data1,train_labels,train_ID],[test_data,test_data1,test_labels,test_ID],batch_size,mean_loader,std_loader

#Barlow
class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, device, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss

class BarlowTwins(nn.Module):
    def __init__(self, backbone,latent_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(latent_dim, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

#RESNET18


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        channel_num: int = 3,
        num_classes: int = 70,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(channel_num, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)



def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    channel_num: int,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers,channel_num, **kwargs)
    return model


def resnet18(pretrained: bool = False, progress: bool = True,channel_num=3, **kwargs: Any) -> ResNet:
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress,channel_num, **kwargs)