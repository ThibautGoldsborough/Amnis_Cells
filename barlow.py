print("Python started running")
# Import required packages
import sys
import os
import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as Fpython
import torch.optim as optim
import random
from sklearn.metrics import confusion_matrix
import itertools
import math
import matplotlib.pyplot as plt
from lightly.data import LightlyDataset
from lightly.data import ImageCollateFunction
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss
import umap


print("All modules imported")


path=sys.argv[1]
num_epochs=int(sys.argv[2])
batch_size=int(sys.argv[3])

print(path)

# Insert filepath for local files  FOR THIBAUT
basepath = path
outpath = basepath


image_dim=64 #Dim of the final images


nuclear_channel="Ch7"
cellmask_channel="Ch1_mask"



df=pd.read_csv(outpath+"/cell_info.csv")

cell_names=df["Cell_ID"].to_numpy()

# if sum(df["Cell_ID"].to_numpy()!=cell_names)!=0:
 #   print("Error, dataframe cell ID do not match with entries saved during image processing step")



image_dict={}

for cell_name in cell_names:
    image_dict[cell_name]={}


# Find Channels
names=[]
for entry in os.listdir(outpath): #Read all files
    if os.path.isfile(os.path.join(outpath, entry)):
        if entry!='image_ID.npy':
            names.append(entry)


channels=[name[:-4] for name in names if name[-4:]=='.npy']

print("Channels found:",channels)

data_dict={}
for channel in channels:
    data_dict[channel]=np.load(outpath+"/"+channel+'.npy')

# Break up array

for channel in data_dict:
    dims=data_dict[channel].shape
    n=dims[0]//image_dim
    l=dims[1]//image_dim
    index=0
    for i in range(n):
        for j in range(l):
            img=data_dict[channel][i*image_dim:i*image_dim+image_dim,j*image_dim:j*image_dim+image_dim]
            image_dict[cell_names[index]][channel]=img
            index+=1


def to_onehot(my_list):
    return_list=[]
    for i,elem in enumerate(my_list):
        j=np.where(np.unique(labels)==elem)
        return_list.append(np.zeros((len(np.unique(my_list)))))
        return_list[-1][j]=1
    return np.array(return_list)

image_dict[0]['Ch1']


Channels=['Ch1']  #Channel to be fed to the NN

images_with_index = []
for image_i in image_dict:
    image=cv.merge([image_dict[image_i][i] for i in Channels])
    images_with_index.append((int(image_i),image))
    


images=np.array([image[1] for image in images_with_index])
names=np.array([image[0] for image in images_with_index])
labels=df['Cell_Type'].to_numpy()
assert sum(names!=df['Cell_ID'].to_numpy()) ==0  #Check that the order has been preserved
DNA_pos=df["DNA_pos"].to_numpy()
Touches_Boundary=df["Touches_boundary"].to_numpy()
labels=df['Cell_Type'].to_numpy()
idx_to_keep=np.array(DNA_pos==1,dtype=int)+np.array(Touches_Boundary==0,dtype=int)+np.array(labels==0,dtype=int)+np.array(labels==2,dtype=int)==3  #keep dnapos, no touch boundarym APC and Other
#Filter
images=images[idx_to_keep]
names=names[idx_to_keep]
labels=labels[idx_to_keep]
labels=to_onehot(labels)

cell_types=[0,1]

mini=int(round(abs(np.array(images).min()),0))
images=images+abs(np.array(images).min())
mean=np.array(images).mean()
maxi=np.array(images).max()
std=np.array(images).std()


#Split data into training and test, just looking at first images now

train_test_split = 0.8

train_data1=images[:int(train_test_split*len(images))]
test_data1=images[int(train_test_split*len(images)):]

train_labels=labels[:int(train_test_split*len(images))]
test_labels=labels[int(train_test_split*len(images)):]

train_ID=names[:int(train_test_split*len(images))]
test_ID=names[int(train_test_split*len(images)):]

#Transform data into tensors, normalize images
transform_basic = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor()
])

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


train_data_basic = CellDataset(train_data1,train_labels,train_ID, transform_basic)
#Create DataLoaders
train_loader_basic = DataLoader(train_data_basic, batch_size=100, shuffle=False)

#data=next(iter(train_loader_basic))[0] Don't delete this is useful

# Get information about CPU/GPU
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()
print(device)

def get_mean_std(loader):
    #https://stackoverflow.com/questions/48818619/pytorch-how-do-the-means-and-stds-get-calculated-in-the-transfer-learning-tutor
    mean = 0.
    std = 0.
    for images, _ in loader:
        images=images[0]
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std


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


train_data = CellDataset(train_data1,train_labels,train_ID, transform_train)
test_data = CellDataset(test_data1,test_labels,test_ID, transform_test)

#Oversampling
from torch.utils.data.sampler import WeightedRandomSampler
counts=np.bincount(np.argmax(train_labels,axis=1))
labels_weights = 1. / counts
weights = labels_weights[np.argmax(train_labels,axis=1)]
sampler = WeightedRandomSampler(weights, len(weights))


#Create DataLoaders
batch_size = 100
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,sampler=sampler)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


transform_train = transforms.Compose(
    [transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=180,fill=37),
    transforms.ToTensor(),
    transforms.Resize([64, 64]),

   transforms.Normalize(mean=[mean_loader], std=[std_loader]),  # for grayscale images
   ])
   
rotated = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize([64, 64]),
    transforms.Normalize(mean=[mean_loader], std=[std_loader]),  # for grayscale images
    transforms.RandomRotation(degrees=180,fill=0),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Lambda(lambda x: x if 0.5>np.random.rand() else transforms.functional.invert(x)-1),
    transforms.Lambda(lambda x: x *(1+0.2*np.random.randn()) ),
   # transforms.Lambda(lambda x: x + (0.1**0.5)*torch.randn(64, 64) )
    
    ]) 

train_data = CellDataset(train_data1,train_labels,train_ID, rotated)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)#,sampler=sampler)


class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, device=device, lambda_param=5e-3):
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



from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor



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
        num_classes: int = len(labels[0]),
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
        self.conv1 = nn.Conv2d(len(Channels), self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
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
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)



latent_dim=124
epochs=num_epochs

train_data = CellDataset(train_data1,train_labels,train_ID, rotated)
train_loader = DataLoader(train_data, batch_size=batch_size,sampler=sampler)  #shuffle=True
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)



class BarlowTwins(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(latent_dim, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

def get_val():
    total_loss = 0
    model.eval()
    for (x0, x1), _ in test_loader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()

    print("Validation Loss:",int(total_loss / len(test_loader)))
    return int(total_loss / len(test_loader))
    
def train():
    total_loss = 0
    model.train()
    for (x0, x1), _ in train_loader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(train_loader)
    return avg_loss


resnet = resnet18(num_classes=latent_dim)
backbone = resnet#nn.Sequential(*list(resnet.children())[:-1])
model = BarlowTwins(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


collate_fn = ImageCollateFunction(input_size=32)


criterion = BarlowTwinsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

print("Starting Training")

train_losses=[]
val_losses=[]
for epoch in range(epochs):
    avg_loss=train()
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
    train_losses.append(avg_loss)
    val_losses.append(get_val())
    
    
    
train_losses=[int(i.detach().cpu()) for i in train_losses]


plt.figure(figsize=(10,10))
plt.plot(train_losses)
plt.plot(val_losses)
plt.savefig(basepath+"/Results/Loss"+str(train_losses[-1])+".png",bbox_inches="tight")


train_data = CellDataset(train_data1,train_labels,train_ID,rotated)
train_loader = DataLoader(train_data, batch_size=124, shuffle=True)#,sampler=sampler)

barlow=np.zeros((len(train_data1),latent_dim))
names=np.zeros((len(train_data1)))
labels=np.zeros((len(train_data1)))
i=0
for (x0, x1), Y in train_loader:
    x0 = x0.to(device)
    x1 = x1.to(device)
    latents=model.backbone(x0).detach().cpu().numpy()
    names[i:i+len(latents)]=Y[:,1]
    labels[i:i+len(latents)]=Y[:,0]
    barlow[i:i+len(latents)]=latents
    i+=len(latents)
    
    
colors=[]
for label in labels:
    if label==0:
        colors.append('red')
    elif label==1:
        colors.append('blue')

fit = umap.UMAP()
u = fit.fit_transform(barlow)

df2=pd.DataFrame()
df2["Cell_ID"]=names
df2["U0"]=u[:,0]
df2["U1"]=u[:,1]
df2.to_csv(basepath+"/Results/Barlow"+str(train_losses[-1])+".csv", index = False, header=True)


plt.figure(figsize=(20,10),dpi=500)
plt.scatter(u[:,0],u[:,1],s=2,alpha=0.4,c=colors)
plt.savefig(basepath+"/Results/Barlow"+str(train_losses[-1])+".png",bbox_inches="tight")


print("Finished python script")