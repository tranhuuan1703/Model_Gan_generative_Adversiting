# Application model GAN network in banner generator
------
### Overview.
My project to talk about banner generator using by GAN model. Model plays an important role generator banner work. The dataset is collected include: text description and image. The dataset have 4095 data about different field such as car adversiting, sofa, .... With the dataset is small then the model generated images is not good. It's evaluated by some FID, PSRN, IS metrics. So in the future I try improve my model.
### Description.
#### Images.
The dataset comprises a diverse array of high-resolution images sourced from various domains, including but not limited to natural scenes, objects, and complex visual scenarios. Each image is meticulously annotated to provide context and highlight key features, ensuring a well-rounded representation of visual content
#### Text description.
the textual is the description of each image. The textual using VietNamese language to description image. The text is slogan or title of image... . The text has a relationship with image, It's talk about features in image.
#### Model GAN.
![alt text](https://github.com/tranhuuan1703/Model_Gan_generative_Adversiting/blob/main/workFlowImage.png)
---
#### Description model
workflow of my model has 3 step. The first, the textual description been word embedding by word2vec. It's convert have format 1024 dimensions. after it pass fully connected layer to become 256 dimension . This data dimension is divided two dimension, each dimension have 128 dimension. With each section I will use the following: the below section I multiply with noise layer (100 dimension), the rest I combine with result of the section multipled with noise layer. This is called **"Conditioning augmentation"**.</br>
```Python
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
```
*Conditioning augmentation layer*
The second, I want to show Stage-1. The result of **Conditioning augmentation** will combine with noise layer (100 dimenssion) and pass upsampling layer. Upsampling layer include convoluation layer, batchnormalization layer, activation Relu, activation LeakyRelu, and activation Tanh.
```Python
# Layer upsampling
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

```
the final I want to show discriminator stage to discrimination image real or image fake. this model use downsampling layer and convolution layer to predict image real or image fake (0, 1). 
```Python
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
```

### Result and experiment.
![alt text](https://github.com/tranhuuan1703/Model_Gan_generative_Adversiting/blob/main/image_result.png)<br>
---
metrics evaluted | FID | PRNS | IS |
--- | --- | --- | --- |
4095 data | 1.8e+64 | 27.7382 | 0.27654 |
### Summary.
My model Gan generator from text to image is one model not good. The model needs to be provided with an amount of data many times larger than the current data to training. In this post I just focus on how to model Gan network.
