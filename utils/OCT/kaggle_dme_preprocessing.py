from scipy.io import loadmat
import numpy as np
from transformers import SamProcessor, SamModel
from datasets import Dataset, Image
import albumentations
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader as TorchDataset
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
from PIL import Image as PILImage
from torch.utils.tensorboard import SummaryWriter
import datetime

def preprocess(data_path, preprocessing_folder, small):
    time = datetime.datetime.now().strftime('%y-%m-%d %H.%M.%S')
    images = []
    masks =[]
    fluids = ['manualFluid1', 'manualFluid2', 'automaticFluidDME']
    for fluid in fluids:
        print(fluid)
        for i in range(10):
            number = str(i+1).zfill(2)
            print("subject"+number)
            subject =loadmat(data_path+"Subject_"+number+".mat")
            s_images = subject['images']
            s_masks = subject[fluid]

            # Filter out invalid masks (e.g. only consisting of nan values)
            idx = get_valid_idx(s_masks)
            for i in idx:
                s_mask = s_masks[:,:, i]
                s_mask = s_mask[..., np.newaxis]
                s_image = s_images[:,:,i]
                s_image = s_image[..., np.newaxis]
                s_image = transform_image(s_image)
                fluid_numbers = np.unique(s_mask)
                # Masks could be added together for data augmentation
                for nr in fluid_numbers:
                    if nr == 0:
                        s_mask_binary = np.where(s_mask == nr, 0, 1).transpose(2,1,0)
                    else:
                        s_mask_binary = np.where(s_mask == nr, 1, 0).transpose(2,1,0)
                    masks.append(s_mask_binary)
                    images.append(s_image)
                    if small:
                        break
    images = np.concatenate(images, axis=0)
    masks = np.concatenate(masks, axis=0)
    dataset = create_dataset(images=images, labels=masks)
    #TODO: Explore Options for transformations
    #dataset.set_transform(transforms)
    split = dataset.train_test_split(test_size=0.2)
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

small = True
data_path = "../../data/OCT/Kaggle/DME/"
preprocessing_folder = data_path + "preprocessed/"
preprocess(data_path, preprocessing_folder, small)