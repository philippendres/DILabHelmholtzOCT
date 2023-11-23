import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pickle
from segment_anything.modeling import sam
from segment_anything import sam_model_registry
from segment_anything.utils import transforms
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, number_of_mats, number_of_shots, number_of_segmentations, longest_side, device, model):
        self.number_of_mats =number_of_mats
        self.number_of_shots = number_of_shots
        self.number_of_segmentations = number_of_segmentations
        self.longest_side = longest_side
        self.device = device
        self.transform = transforms.ResizeLongestSide(self.longest_side)
        sam_model = model
        self.sam_object = sam.Sam(sam_model.image_encoder,sam_model.prompt_encoder, sam_model.mask_decoder)
        self.sam_object.to(device=device)
    def __len__(self):
        return self.number_of_mats*self.number_of_shots*self.number_of_segmentations

    def __getitem__(self, idx):
        mat = idx //(self.number_of_shots*self.number_of_segmentations)
        shot = (idx - mat*(self.number_of_shots*self.number_of_segmentations)) //self.number_of_segmentations
        segmentation_number = (idx- mat*(self.number_of_shots*self.number_of_segmentations)) % self.number_of_segmentations
        number = str(mat+1).zfill(3)
        with open("..\data\Kaggle\preprocessed\\amd_"+number+".pkl", 'rb') as file:
            dict = pickle.load(file)
        image = dict['images'][shot]['image']
        image_RBG_uint8 = transform_image(image)
        image_transformed = self.transform.apply_image(image_RBG_uint8)
        image_torched = torch.as_tensor(image_transformed, device=self.device)
        input_image_preprocessing = image_torched.permute(2, 0, 1).contiguous()[ :, :, :]
        image_preprocessed = self.sam_object.preprocess(input_image_preprocessing)

        box = dict['images'][shot]['bbox'][segmentation_number]
        torched_box = torch.as_tensor(box, device=self.device)

        gt_binary_mask = dict['images'][shot]['segmentations'][segmentation_number]
        torched_gt = torch.as_tensor(gt_binary_mask, device=self.device)
        return image_preprocessed, torched_box[None, :].float(), torched_gt[None,:,:]

def transform_image(img):
    image = np.expand_dims(img, axis=2)
    image = np.repeat(255 - image, 3, axis=2)
    return image