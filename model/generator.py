import torch
from torch import nn
import sys
sys.path.append("C:\\Users\\ACER\\Documents\\StackGan_model\\model\\")
from config import configuration
from model.layers import condition_agu, downsample, upsample


cf = configuration()

class process(nn.Module):
  def __init__(self, input_shape, output_shape):
    super(process, self).__init__()

    self.main = nn.Sequential(
        nn.Linear(input_shape, output_shape),
        nn.BatchNorm1d(output_shape),
        nn.ReLU(inplace=True)
    )

  def forward(self, ca):
    result = self.main(ca)
    return result

class generator(nn.Module):

  def __init__(self):
    super(generator, self).__init__()

    self.condition = condition_agu(1024, 128)
    self.up1 = upsample(1024, 512)
    self.up2 = upsample(512, 256)
    self.up3 = upsample(256, 128)
    self.up4 = upsample(128, 64)
    self.process = process(228, 16384)
    self.main = nn.Sequential(
        nn.Conv2d(64, 3, 3, stride=1, padding=1),
        nn.Tanh()
    )
  def forward(self, word_embedding, noise):

    mean_d, logvar, ca = self.condition(word_embedding)

    layer_ = torch.cat((ca, noise), axis=1)
    layer_ = self.process(layer_)

    feature_shape = layer_.view(-1, 1024, 4, 4)

    layer_up1 = self.up1(feature_shape)
    layer_up2 = self.up2(layer_up1)
    layer_up3 = self.up3(layer_up2)
    layer_up4 = self.up4(layer_up3)

    image_gen = self.main(layer_up4)

    return mean_d, logvar, image_gen