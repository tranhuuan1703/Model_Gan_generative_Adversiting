import torch
from torch import nn
import sys
sys.path.append("C:\\Users\\ACER\\Documents\\StackGan_model\\model")
from model.config import configuration
cf = configuration()

class condition_agu(nn.Module):

  def __init__(self, input_shape, output_shape):
    super(condition_agu, self).__init__()
    self.input_shape = input_shape
    self.out_shape = output_shape

    self.main = nn.Sequential(
        nn.Linear(input_shape, output_shape *2)
    )

  def forward(self, input_embedding):

    text_ = self.main(input_embedding)

    mean = text_[:, : 128]
    logvar = text_[:, 128 :]
    stdev = (logvar*0.5).exp()

    noise = torch.normal(0, 1, size = stdev.size()).to(cf.device)

    ca = torch.add(mean, torch.mul(stdev, noise))

    return mean, logvar, ca

class upsample(nn.Module):

  def __init__(self, input_shape, output_shape):
    super(upsample, self).__init__()

    self.main = nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(input_shape, output_shape, 3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(output_shape),
        nn.LeakyReLU(0.2)
    )

  def forward(self, features):
    result = self.main(features)
    return result


class downsample(nn.Module):

  def __init__(self, input_shape, output_shape):
    super(downsample, self).__init__()
    self.main = nn.Sequential(
        nn.Conv2d(input_shape, output_shape, 4, padding=1, stride=2, bias=True),
        nn.BatchNorm2d(output_shape),
        nn.LeakyReLU(0.2)
    )

  def forward(self, x):
    x = self.main(x)
    return x

# using caculator loss function
class get_logist(nn.Module):
  def __init__(self, input_shape, output_shape):
    super(get_logist, self).__init__()

    self.main = nn.Sequential(
        nn.Conv2d(input_shape, input_shape//2, 3, stride=1, padding=1),
        nn.BatchNorm2d(input_shape//2),
        nn.LeakyReLU(0.2),
        nn.Conv2d(input_shape//2, output_shape, 4, stride=1, padding=0)
    )
  def forward(self, feature, mean):

    mean = mean.view(-1, mean.size()[1], 1, 1)
    mean = mean.repeat(1, 1, 4, 4)

    x = torch.cat((feature, mean), axis=1)

    x = self.main(x)
    return x

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD