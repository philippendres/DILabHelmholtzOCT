import wandb
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
import datetime

# Usage:
# 1. Execute kaggle_dme_preprocessing (after adjusting the parameters, like where the data should be stored)
# 2. Execute kaggle_dme_finetuning (after adjusting the paramters, like the data paths)
# Also: log into W&B for logging, or change the appearances of wandb
torch.cuda.empty_cache()

from imagesgreyscale_utils import training

mask_dict = (
    "background",
    "epiretinal membrane",
    "neurosensory retina",
    "intraretinal fluid",
    "subretinal fluid",
    "subretinal hyperreflective material",
    "retinal pigment epithelium",
    "pigment epithelial detachment",
    "posterior hyaloid membrane",
    "choroid border",
    "imaging artifacts",
    "fibrosis",
    "vitreous body",
    "image padding" 
)


# Configure W&B parameters
project_name = "OCT_kaggle_dme"
entity = "dilab-helmholtz"
# Choose base_model configurations
base_models = ["facebook/sam-vit-base", "facebook/sam-vit-huge", "facebook/sam-vit-large", "wanglab/medsam-vit-base"]
base_model = base_models[0]
data_path = "/vol/data/datasets/processed/custom/"

preprocessed_dataset = "23-12-18_14.50.43"

for i in [3]:
    time = datetime.datetime.now().strftime('%y-%m-%d_%H.%M.%S')
    checkpoint_path = "/vol/data/models/" +time

    display_name = mask_dict[i]
    #display_name = base_model + " " + time
    config = {
        "learning_rate": 1e-3,
        "weight_decay": 0,
        "epochs": 10,
        "batch_size": 16,
        "data_path": data_path,
        "dataset": data_path + preprocessed_dataset,
        "loss": "dice",
        "checkpoint": checkpoint_path,
        "time": time,
        "segnr": i
    }


    wandb.init(
        project=project_name,
        entity=entity,
        name=display_name,
        config=config
    )
    training(base_model, config)



