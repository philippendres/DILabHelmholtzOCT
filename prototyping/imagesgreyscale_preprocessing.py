# Deprecated. See src/octsam/data/preprocessing
from scipy.io import loadmat
import numpy as np
from transformers import SamProcessor, SamModel
from datasets import Dataset, Image
import albumentations
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader as TorchDataset
from torch.optim import Adam
from tqdm import tqdm
from statistics import mean
import torch
from PIL import Image as PILImage
import datetime
import os
import cv2

def preprocess(data_path, preprocessing_folder):
    files = os.fsencode(os.path.join(data_path, "imagesgreyscale"))
    
    time = datetime.datetime.now().strftime('%y-%m-%d %H.%M.%S')
    images = []
    masks =[]
    for file in os.listdir(files):
        filename = os.fsdecode(file)
        image = cv2.imread(os.path.join(data_path, "imagesgreyscale", filename))
        mask = cv2.imread(os.path.join(data_path, "masks14", filename))
        #print(image.shape)
        if mask.shape != (496,512,3) or image.shape != (496,512,3):
            print(filename)
            print(mask.shape)
            print(image.shape)
            continue
        images.append(image)
        masks.append(mask)
    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)
    dataset = create_dataset(images=images, labels=masks)
    split = dataset.train_test_split(test_size=0.2, shuffle=True)
    split.save_to_disk(preprocessing_folder + time)


def transform_image(img):
    img = img.transpose(2,1,0)
    image = np.expand_dims(img, axis=3)
    image = np.repeat(255 - image, 3, axis=3)
    return image

def get_valid_idx(mask):
    idx = []
    for i in range(0,61):
        temp = mask[:,:,i]
        if np.nansum(temp) != 0:
            idx.append(i)
    return idx

def create_dataset(images, labels):
    dataset = Dataset.from_dict({"image": images,
                                 "label": labels})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())
    return dataset

data_path = "/vol/data/OCT_data"
preprocessing_folder = data_path + "/preprocessed/"
preprocess(data_path, preprocessing_folder)