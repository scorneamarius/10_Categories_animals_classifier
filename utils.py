import torch
from torchvision.io import read_image
import cv2
import torch.utils
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def encode_labels(dataset_path):
    labels_dict = dict()
    for idx,dirname in enumerate(os.listdir(dataset_path)):
        labels_dict[dirname] = idx
    return labels_dict

def get_dataset(csv_file):
    return AnimalsTrainDataset(csv_file)

def get_dataLoader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = False)


class AnimalsTrainDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df_train = pd.read_csv(csv_file)
        
    def __len__(self):
        return self.df_train.shape[0]
    
    def __getitem__(self, index):
        path_filename = self.df_train.iloc[index]['image_path']
        label = int(self.df_train.iloc[index]['label'])

        image = cv2.resize(cv2.imread(path_filename), (224, 224))
        image = image.astype('float32') / 255.0
        image = np.reshape(image, (3, 224, 224))
        image = torch.from_numpy(image) 

        label_one_hot_encoded = np.zeros(10)
        label_one_hot_encoded[label] = 1
        label_one_hot_encoded = torch.from_numpy(label_one_hot_encoded)

        return image, label 

