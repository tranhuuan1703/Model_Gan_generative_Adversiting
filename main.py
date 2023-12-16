# import libraries
import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch
from torch import nn
from torchsummary import summary
from gensim.models import Word2Vec
from torchvision import transforms
from underthesea import word_tokenize
from torch import nn
from sklearn.manifold import TSNE
from tqdm import tqdm_notebook
from torch.autograd import Variable
import re

import os
import sys
sys.path.append("C:\\Users\\ACER\\Documents\\StackGan_model\\model")
from model.dataset import custom_data, remove_stop_word, replace_character, sentence_embedding
from model.config import configuration
from model.generator import generator
from model.Discriminator import Discriminator
from model.layers import KL_loss
cf = configuration()

# read file data
lsst_txt = []
with open(cf.data_txt, 'r', encoding='utf-8') as file:
  for i in range(9180):
    string = file.readline()
    if string != '\n':
      lsst_txt.append(string)

lsst_txt = [replace_character(x) for x in lsst_txt]

# remove stopword
stop_word_file = cf.stop_word_file

# remove stop word

with open(stop_word_file, 'r', encoding='utf-8') as file_stw:
  list_stword = file_stw.read().split("\n")

text_word_tokenize = [word_tokenize(sentence) for sentence in lsst_txt]

# load model
my_model_embedding = Word2Vec.load(cf.word2vec)
# word embedding
embedding_text = [sentence_embedding(string, my_model_embedding) for string in lsst_txt]
# handle file image
all_files = []
for root, dirs, files in os.walk(cf.url_image):
  for file in files:
      full_path = os.path.join(root, file)
      all_files.append(full_path)


dataset = custom_data(embedding_text, all_files)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=cf.batch_size, num_workers=cf.num_workers)


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

noise_rand = Variable(torch.rand(cf.batch_size, 100)).to(cf.device)
gen = generator().to(cf.device)
gen.apply(weights_init)


disc = Discriminator().to(cf.device)
disc.apply(weights_init)

crietion = nn.BCEWithLogitsLoss()

opt_g = torch.optim.Adam(gen.parameters(), lr=0.001, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(disc.parameters(), lr=0.001, betas=(0.5, 0.999))

real_label = torch.ones(cf.batch_size).to(cf.device)
fake_label = torch.zeros(cf.batch_size).to(cf.device)


def train(epoch, data_loader, model_gen, model_disc, opt_g, opt_d, crietion, noise):
  epochs = "Epochs: " + str(epoch)

  for batch, data in tqdm_notebook(enumerate(data_loader), desc=epochs):

    word_embedding, images = data
    images = images.to(cf.device)
    word_embedding = word_embedding.to(cf.device)

    mean_d, logvar, image_gen = model_gen(word_embedding, noise)

    # training discriminator
    model_disc.zero_grad()

    ouput_feaure_real = model_disc(images.detach())
    opt_output_feature_real = model_disc.get_logist_(ouput_feaure_real, mean_d.detach())
    loss_real = crietion(opt_output_feature_real.squeeze(), real_label)
    loss_real.backward()

    ouput_feaure_fake = model_disc(image_gen.detach())
    opt_output_feature_fake = model_disc.get_logist_(ouput_feaure_fake, mean_d.detach())
    loss_fake = crietion(opt_output_feature_fake.squeeze(), fake_label)
    loss_fake.backward()

    opt_output_wrong = model_disc.get_logist_(ouput_feaure_real[:(cf.batch_size-1)], mean_d[1:].detach())
    loss_wrong = crietion(opt_output_wrong.squeeze(), real_label[1:])
    # loss_wrong.backward()

    loss_d = loss_real + (loss_wrong + loss_fake)*0.5
    opt_d.step()

    # training generator
    model_gen.zero_grad()
    ouput_feaure_fake = model_disc(image_gen)
    opt_output_feature_fake = model_disc.get_logist_(ouput_feaure_fake, mean_d)
    loss_g = crietion(opt_output_feature_fake.squeeze(), real_label)

    kl_loss = KL_loss(mean_d, logvar)
    loss_G = loss_g + kl_loss * 2.0
    loss_G.backward()
    opt_g.step()
  print("Epoch: {epoch} || Loss Discriminator: {loss_total} || Loss Generator: {loss_G}".format(epoch = epoch, loss_total = loss_d, loss_G = loss_G))
  return loss_d.item(), loss_G.item()


lst_loss_total = []
lst_loss_g = []
for epoch in range(1, 1000):
  loss_total, loss_g = train(epoch, data_loader, gen, disc, opt_g, opt_d, crietion, noise_rand)

  lst_loss_total.append(loss_total)
  lst_loss_g.append(loss_g)
  if lst_loss_g[-1] < 0.8:
    torch.save(gen.state_dict(), cf.save_model_gen)
    torch.save(disc.state_dict(), cf.save_model_discriminatior)
    break