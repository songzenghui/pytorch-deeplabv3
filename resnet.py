from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ResNet(nn.Module):
	"""docstring for ResNet"""
	def __init__(self):
		super(ResNet, self).__init__()
		self.conv0 = nn.Sequential(
			nn.Conv2d(3,64,kernel_size=7,stride=1,padding=3),
			nn.BacthNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
		self.residual_block1 = nn.Sequential(
			nn.Conv2d(64,256,kernel_size=3,stride=1,padding=1),
			nn.BacthNorm2d(256),
			nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
			nn.BacthNorm2d(256),
			nn.Conv2d(256,64,kernel_size=3,stride=2,padding=1),
			nn.BacthNorm2d(64),
			nn.ReLU())
		self.residual_block2 = nn.Sequential(
			nn.Conv2d(64,256,kernel_size=3,stride=1,padding=1),
			nn.BacthNorm2d(64),
			nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
			nn.BacthNorm2d(256),
			nn.Conv2d(256,64,kernel_size=3,stride=2,padding=1),
			nn.BacthNorm2d(64),
			nn.ReLU())
		self.residual_block3 = nn.Sequential(
			nn.Conv2d(64,256,kernel_size=3,stride=1,padding=1),
			nn.BacthNorm2d(256),
			nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
			nn.BacthNorm2d(256),
			nn.Conv2d(256,64,kernel_size=3,stride=2,padding=1),
			nn.BacthNorm2d(64),
			nn.ReLU())
		self.residual_block4 = nn.Sequential(
			nn.Conv2d(64,256,kernel_size=3,stride=1,padding=1,dilation=2),
			nn.BacthNorm2d(256),
			nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,dilation=4),
			nn.BacthNorm2d(256),
			nn.Conv2d(256,64,kernel_size=3,stride=1,padding=1,dilation=2),
			nn.BacthNorm2d(64),
			nn.ReLU())
		

	def forward():