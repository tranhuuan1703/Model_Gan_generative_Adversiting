import torch
from torch import nn
import cv2
from torchvision import transforms
from underthesea import word_tokenize
import numpy as np
import re
import sys
sys.path.append("C:\\Users\\ACER\\Documents\\StackGan_model\\model")
from model.config import configuration
cf = configuration()



################################################
#           data preprocessing                 #
#                                              #
################################################

def replace_character(x):
  if x is None:
    return x
  else:
    try:
      x = x.replace('\n', ' ')
      x = x.replace('\t', ' ')
      x = x.replace('\r', ' ')

      bad_chars = "[!@#$%^&*()[]{};:,./<>?\|`~-=_+“”]"
      my_new_string = ''.join(map(lambda x: x if x not in bad_chars else ' ', x))
      my_new_string = my_new_string.lower()
      my_new_string = re.sub(' +', ' ', my_new_string)
      my_new_string = re.sub('[0-9]', '', my_new_string)
      return my_new_string
    except:
      pass

def remove_stop_word(string, list_stword):
  try:
    my_string = ' '.join(i for i in string.split(' ') if not i in list_stword)
    return my_string
  except:
    pass


def sentence_embedding(sentence, model):
  tokens = word_tokenize(sentence.lower())
  vectorized_tokens = [model.wv[token] for token in tokens if token in model.wv.index_to_key]

  if not vectorized_tokens:
    return None  # Nếu không có từ nào trong từ điển
  # vectorized_tokens = np.hstack(vectorized_tokens)
  # Trả về vector trung bình của các vector từ
  vectorized_tokens = np.array(vectorized_tokens)
  return np.mean(vectorized_tokens, axis=0)




################################################
##          data setup                        ##
##                                            ##
################################################
def processing(image, size_image):

  img_resize = cv2.resize(image, (size_image, size_image))

  fillter_noise = cv2.GaussianBlur(img_resize, (5,5), 1)

  norm = np.zeros((size_image, size_image))
  norm_img = cv2.normalize(fillter_noise, norm, 0, 255, cv2.NORM_MINMAX)

  # norm_img_ = norm_img / 255
  return norm_img

class custom_data(nn.Module):
  def __init__(self, txt, image_path):
    super(custom_data, self).__init__()
    self.txt = txt
    self.image_path = image_path

  def __len__(self):
    return len(self.image_path)

  def load_image(self, path):
    img = cv2.imread(path)
    return img

  def __getitem__(self, idx):
    text = self.txt[idx]
    image = self.load_image(self.image_path[idx])
    img_new = processing(image, 64)
    cvImg = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
    img = transforms.ToTensor()(cvImg)
    img = img.to(torch.float32)

    return text, img