import os
import cv2
from tqdm import tqdm
import numpy as np
from datasets import Dataset, Image

def preprocess(dataset, raw_data_path, processed_data_path, preprocessing_config):
    preprocessing_config["print_status"] and print("Start preprocessing")
    if dataset == "custom":
        images, masks = preprocess_custom(raw_data_path, preprocessing_config)
    elif dataset == "dme":
        images, masks = preprocess_dme(raw_data_path, preprocessing_config)
    elif dataset == "amd":
        images, masks = preprocess_amd(raw_data_path)
    else:
        raise ValueError("dataset is not implemented")

    preprocessing_config["print_status"] and print("Preprocessed images and masks. Now creating dataset")
    dataset = create_dataset(images=images, labels=masks)
    split = dataset.train_test_split(test_size=preprocessing_config["test_size"], shuffle=preprocessing_config["shuffle"])
    preprocessing_config["print_status"] and print("Created dataset. Now writing to disk")

    save_directory = os.path.join(processed_data_path, preprocessing_config["additional_file_description"] + "preprocessed_at_" + preprocessing_config["time"])
    split.save_to_disk(save_directory)
    preprocessing_config["print_status"] and print("Finished")


def preprocess_dme(raw_data_path, preprocessing_config):
    images = []
    masks =[]

    # Function to filter out empty masks 
    def get_valid_idx(mask):
        idx = []
        for i in range(0,61):
            temp = mask[:,:,i]
            if np.sum(temp) != 0:
                idx.append(i)
        return idx
    
    for i in range(10):
        number = str(i+1).zfill(2)
        preprocessing_config["print_status"] and print("subject"+number)
        subject =loadmat(data_path+"Subject_"+number+".mat")
        s_images = subject['images']
        s_masks = subject[preprocessing_config["use_masks"]]

        # Filter out invalid masks (e.g. only consisting of zeros)
        s_masks[np.isnan(s_masks)] = 0
        idx = get_valid_idx(s_masks)
        for i in idx:
            s_mask = s_masks[:,:, i]
            s_image = s_images[:,:,i]

            # Transform image to RGB
            image = np.expand_dims(img, axis=2)
            image = np.repeat(image, 3, axis=2)
            
            masks.append(s_mask)
            images.append(s_image)
    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)
    return images, masks


def preprocess_custom(raw_data_path, preprocessing_config):
    files = os.fsencode(os.path.join(raw_data_path, "imagesgreyscale"))
    files_list = tqdm(os.listdir(files)) if preprocessing_config["print_status"] else os.listdir(files)
    images, masks = [],[]
    for file in files_list:
        filename = os.fsdecode(file)
        image = cv2.imread(os.path.join(raw_data_path, "imagesgreyscale", filename))
        mask = cv2.imread(os.path.join(raw_data_path, "masks14", filename))
        #print(image.shape)
        if mask.shape != (496,512,3) or image.shape != (496,512,3):
            if preprocessing_config["print_status"]:
                print("Skipped image of different size!")
                print(filename)
                print(mask.shape)
                print(image.shape)
            continue
        images.append(image)
        masks.append(mask[:,:,0])
    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)
    return images, masks

def preprocess_amd():
    raise NotImplementedError()

def create_dataset(images, labels):
    dataset = Dataset.from_dict({"image": images,
                                 "label": labels})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())
    return dataset

