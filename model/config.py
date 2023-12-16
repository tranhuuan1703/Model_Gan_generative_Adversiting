import torch

class configuration:

    def __init__(self):
        self.batch_size = 16
        self.dim_embedding = 1024
        self.noise = 100
        self.img_size = 64
        self.data_txt = "data/dataset/data_description.txt"
        self.url_model = "/data/model_w2v.model"
        self.stop_word_file = "data/VietNamese_txt.txt"
        self.url_image = "/data/dataset/data_image/"
        self.save_model_gen = '/data/model_gen.pth'
        self.save_model_discriminatior = '/data/model_dis.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

