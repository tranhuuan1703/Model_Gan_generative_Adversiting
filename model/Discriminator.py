import torch
from torch import nn
import numpy as np
import sys
sys.path.append("C:\\Users\\ACER\\Documents\\StackGan_model\\model")
from model.config import configuration
from model.layers import upsample, downsample, get_logist

cf = configuration()


class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(
        downsample(3, 64),
        downsample(64, 128),
        downsample(128, 256),
        downsample(256, 512))

    self.get_logist_ = get_logist(640, 1)
  def forward(self, x):
    x = self.main(x)
    return x