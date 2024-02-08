# Work in progress 

# Usage:
# 1. Execute preprocessing (after adjusting the parameters, like where the data should be stored)
# 2. Execute finetuning (after adjusting the paramters, like the data paths)
# Also: log into W&B for logging, or change the appearances of wandb

import albumentations
import torch
import os
import wandb
import datetime
from training_utils import training

torch.cuda.empty_cache()
time = datetime.datetime.now().strftime('%y-%m-%d_%H.%M.%S')

# Configure W&B parameters
project_name = "OCT_segmentation"
entity = "dilab-helmholtz"
# Choose base_model
base_models = ["facebook/sam-vit-base", "facebook/sam-vit-huge", "facebook/sam-vit-large", "wanglab/medsam-vit-base"]
base_model = base_models[0]
# Choose dataset
datasets = ["custom", "dme", "amd"]
dataset = datasets[0]
data_directory = "/vol/data"
processed_dataset = "default_preprocessed_at_23-12-22_16.11.35"
data_path = os.path.join(data_directory, "datasets", "processed", dataset, processed_dataset)
model_path = os.path.join(data_directory, "models", dataset)
# Choose display_name
display_name = ""
# Choose loss
losses = ["diceCE"]
loss = losses[0]
   


config = {
    "display_name": display_name,
    "base_model": base_model,
    "dataset": data_path,
    "checkpoint": model_path,
    "learning_rate": 1e-3,
    "weight_decay": 0,
    "epochs": 10,
    "batch_size": 8,
    "shuffle": False,
    "data_transforms": [],
    "optimizer": "adam",
    "loss": loss,
    "time": time,
    "topological":False,
    "nr_of_decoders": 14,
    "pseudo_coloring":"Bone"
}

# Choose the mode for display_samples
display_options = ["no", "predefined images", "random_equal_images", "random_changing_images"]
display_option = 3
config["display_samples"] = display_options[display_option]
if display_option == 1:
    config["train_samples"] = [1]
    config["valid_samples"] = [1]
elif display_option > 1:
    config["nr_of_train_samples"] = 3
    config["nr_of_valid_samples"] = 2
display_modes = ["single_masks", "all_masks"]
config["display_mode"] = display_modes[1]

# Choose segmentation mode
modes = ["single_mask", "all_masks_one_model", "all_masks_seperate_models" ]
mode = 1
config["mode"] = modes[1]

# Select mask
if mode == modes[0]:
    config["seg_nr"] = 1

# Provide dataset specific info
if dataset == "custom":
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
    
elif dataset == "dme":
    pass

mask_dict = {k: v for k, v in enumerate(mask_dict)}
config.update({
    "mask_dict": mask_dict,
})

if display_name == "":
    config[display_name] = config["mode"] + "_" + dataset


wandb.init(
    project=project_name,
    entity=entity,
    name=display_name,
    config=config,
    save_code=True,
    dir = "/vol/data/runs",
)

training(base_model, config)