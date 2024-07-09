!pip install einops
import os
import sys
import numpy as np
from tqdm import tqdm
from skimage import io
from PIL import Image
import torch
from skimage.morphology import label
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import numpy as np
import os
import torch.optim as optim
from torch.autograd import Variable
from skimage import io
from skimage.transform import resize

from einops import rearrange, repeat
from einops.layers.torch import Rearrange