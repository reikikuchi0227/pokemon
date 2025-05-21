# 必要ライブラリのインポート

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import utils

# warning表示off
import warnings
warnings.simplefilter('ignore')

# デフォルトフォントサイズ変更
plt.rcParams['font.size'] = 14

# デフォルトグラフサイズ変更
plt.rcParams['figure.figsize'] = (6,6)

# デフォルトで方眼表示ON
plt.rcParams['axes.grid'] = True

# numpyの表示桁数設定
np.set_printoptions(suppress=True, precision=5)

# GPUチェック
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

test_transform = transforms.Compose([
    transforms.Resize(224,224),
    transforms.CenterCrop(224,224),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5])
    ])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize(224,224),
    transforms.CenterCrop(224,224),
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5),
    transforms.RandomErasing(p=0.5, scale=(9.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])

train_dataset = datasets.ImageFolder(root='pokemon_dataset', transform=train_transform)
val_dataset = datasets.ImageFolder(root='pokemon_dataset', transform=test_transform)

train_dataloader = 