#Transoforms .mat files into pickles with binary masks

import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
from matplotlib import pyplot as plt
from glob import glob
from os import path
from PIL import Image
import pickle

#goal: preprocess Kaggle datset into numpy arrays including images and masks

"""saves the data in a dict with the structure: 
{'image_info': ..., 'images': [{'image': ...,'segmentations': [..., ..., ...., ....]}, ...]"""
def process_amd(file_name):
    mat = loadmat(file_name)
    images = mat['images']
    layerMaps = mat['layerMaps']
    width = 512
    height = 1000
    number_of_images = 100
    segmentation_classes = 4
    dict ={}
    dict['image_info'] = {'number_of_images': number_of_images, 'width': width, 'height': height, 'file_name': file_name}
    dict['images'] = []
    for image_number in range(number_of_images):
        segmentation = []
        input_boxes = []
        for i in range(segmentation_classes):
            segmentation.append(np.zeros([width, height]))
            input_boxes.append(np.zeros([segmentation_classes]))
        for i in range(segmentation_classes):
            first_non_nan_value = True
            for j in range(height):
                previous_layerMap = 0
                if i == segmentation_classes-1:
                    layerMap = height-1
                    previous_layerMap = layerMaps[image_number,j,i-1]
                    if np.isnan(previous_layerMap):
                        continue
                else:
                    layerMap = layerMaps[image_number,j,i]
                    if np.isnan(layerMap):
                        continue
                    if i > 0:
                        previous_layerMap = layerMaps[image_number,j,i-1]
                if first_non_nan_value:
                    first_non_nan_value = False
                    input_boxes[i][0] = previous_layerMap
                    input_boxes[i][1] = j
                    input_boxes[i][2] = layerMap
                    input_boxes[i][3] = j
                input_boxes[i][0] = min(input_boxes[i][0], previous_layerMap)
                input_boxes[i][1] = min(input_boxes[i][1], j)
                input_boxes[i][2] = max(input_boxes[i][2], layerMap)
                input_boxes[i][3] = max(input_boxes[i][3], j)
                segmentation[i][int(previous_layerMap):int(layerMap),j] = 1

        dict['images'].append({'image':images[:,:,image_number], 'segmentations': segmentation, 'bbox': input_boxes})

    return dict

def save_preprocessed_dict(dict, save_name):
    with open(save_name + '.pkl', 'wb') as file:
        pickle.dump(dict, file)

#TODO: implement analogously for dme (with manualfluid)

mat_amd, mat_dme, mat_control = [],[],[]
"""
for i in range(10):
    number = str(i+1).zfill(2)
    mat_dme.append("..\data\Kaggle\AMD\Subject_"+number+".mat")

for i in range(10):
    number = str(i+1).zfill(3)
    mat_control.append("..\data\Kaggle\AMD\Farsiu_Ophthalmology_2013_Control_Subject_1"+number+".mat")
"""
for i in range(10):
    number = str(i+1).zfill(3)
    mat_amd.append("..\data\Kaggle\AMD\Farsiu_Ophthalmology_2013_AMD_Subject_1"+number+".mat")
    processed_dict = process_amd(mat_amd[i])
    save_preprocessed_dict(processed_dict, "..\data\Kaggle\AMD\preprocessed\\amd_"+number)





dict = process_amd(mat_amd[0])
test_number = 50
segmentations = dict['images'][test_number]['segmentations']
image = dict['images'][test_number]['image']
plt.imshow(image, cmap='gray')
plt.imshow(segmentations[0], cmap='jet', alpha=0.3)

