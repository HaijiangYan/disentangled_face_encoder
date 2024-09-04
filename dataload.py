import numpy as np
import os
import scipy.misc as sm
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image


class MetaFace(data.Dataset):
    def __init__(self, path, size=128, GRAY_flag=True):
        '''
        construct a dataset for MHFED database
        :param img_dir: the root directory of the face database
        :param size: the size of output image
        :param RGB_flag: the flag to indicate whether load the image in RGB type
        '''

        emo_map = {'Neutral':0, 'Sadness':1, 'Joy':2, 'Anger':3, 'Disgust':4, 'Fear':5, 'Surprise':6}

        self.GRAY = GRAY_flag
        self.size = size

        # root directory of face dataset

        self.img_dir = []
        self.emo_labels = []
        self.id_dir = []
        for identity in os.listdir(path):
            id_dir = os.path.join(path, identity)
            id_dir_neutral = os.path.join(id_dir, 'Neutral')
            id_neutral_face = os.path.join(id_dir_neutral, os.listdir(id_dir_neutral)[0])
            for emo in os.listdir(id_dir):
                emo_dir = os.path.join(id_dir, emo)
                for img in os.listdir(emo_dir):
                    img_dir = os.path.join(emo_dir, img)
                    self.img_dir.append(img_dir)
                    self.emo_labels.append(emo_map[emo])
                    self.id_dir.append(id_neutral_face)
        self.emo_labels = torch.tensor(self.emo_labels)

        # transform for face dataset
        self.transform_basic = transforms.Compose([
            transforms.Resize([size, size]),
            transforms.ToTensor(),
        ])

        self.transform_advance = transforms.Compose([
            transforms.Resize(144),
            transforms.RandomCrop(self.size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
        ])


    def __getitem__(self, idx):

        # load face image and label and identity images
        face_input = self.img_prepro(self.img_dir[idx], self.transform_advance, self.GRAY)
        emo = self.emo_labels[idx].long()
        face_id = self.img_prepro(self.id_dir[idx], self.transform_basic, self.GRAY)
        face_emo = self.img_prepro(self.img_dir[idx], self.transform_basic, self.GRAY)

        return face_input, emo, face_id, face_emo


    def __len__(self):
        return len(self.emo_labels)

    def img_prepro(self, img_path, transform=None, GRAY=True):
        if GRAY:
            img = Image.open(img_path).convert('L')
        else:
            img = Image.open(img_path).convert('RGB')
        if transform is not None:
            img = transform(img)

        return img

