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
import argparse
import cv2

torch.cuda.empty_cache()
time = datetime.datetime.now().strftime('%y-%m-%d_%H.%M.%S')

parser = argparse.ArgumentParser()

# W&B parameters
parser.add_argument("--project_name", type=str, default="OCT-Mikhail-experiments")
parser.add_argument("--entity", type=str, default="dilab-helmholtz")

# Model Info
# base_models = ["facebook/sam-vit-base", "facebook/sam-vit-huge", "facebook/sam-vit-large", "wanglab/medsam-vit-base"]
parser.add_argument("--base_model", type=str, default="facebook/sam-vit-base")
parser.add_argument("--loss", type=str, default="diceCE")

# Dataset type and location
# datasets = ["custom", "dme", "amd"]
parser.add_argument("--dataset", type=str, default="custom")
parser.add_argument("--data_directory", type=str, default="/vol/data")
parser.add_argument("--dataset_name", type=str, default="default_preprocessed_at_24-01-10_13.41.28")

#Training parameters
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--bs", type=int, default=2)
parser.add_argument("--shuffle", type=bool, default=False)
parser.add_argument("--optimizer", type=str, default="adam")


# Misc arguments
# display_options = ["none", "predefined", "random_equal", "random_changing"]
# display_idx - comma-separated indexes (no spaces)
parser.add_argument("--display_mode", type=str, default="predefined")
parser.add_argument("--display_idx", type=str, default="0, 1, 3")
parser.add_argument("--display_val_nr", type=int, default=1)
parser.add_argument("--display_train_nr", type=int, default=1)

modes = ["single_mask", "all_masks_one_model", "all_masks_seperate_models" ]
parser.add_argument("--mode", type=int, default=1)
parser.add_argument("--seg_nr", type=int, default=3)

OCV_COLORMAPS = {
    "Autumn": cv2.COLORMAP_AUTUMN, 
    "Bone": cv2.COLORMAP_BONE,
    "Cividis": cv2.COLORMAP_CIVIDIS, 
    "Cool": cv2.COLORMAP_COOL, 
    "Deepgreen": cv2.COLORMAP_DEEPGREEN,
    "Hot": cv2.COLORMAP_HOT,
    "HSV": cv2.COLORMAP_HSV,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Jet": cv2.COLORMAP_JET,
    "Magma": cv2.COLORMAP_MAGMA,
    "Ocean": cv2.COLORMAP_OCEAN,
    "Parula": cv2.COLORMAP_PARULA,
    "Pink": cv2.COLORMAP_PINK,
    "Plasma": cv2.COLORMAP_PLASMA,
    "Rainbow": cv2.COLORMAP_RAINBOW,
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Winter": cv2.COLORMAP_WINTER,
    "Spring": cv2.COLORMAP_SPRING,
    "Summer": cv2.COLORMAP_SUMMER,
    "Twilight shifted": cv2.COLORMAP_TWILIGHT_SHIFTED,
    "Twilight": cv2.COLORMAP_TWILIGHT,
    "Turbo": cv2.COLORMAP_TURBO,
    "grayscale": None
}
parser.add_argument("--pseudocolor", type=str, default="grayscale")

parser.add_argument("--display_name", type=str, default="")

parser.add_argument("--evaluate", type=bool, default=True)

parser.add_argument("--prompt", type=str, default="points")

parser.add_argument('--top', action='store_true')

args = parser.parse_args()


data_path = os.path.join(args.data_directory, "datasets", "processed", args.dataset, args.dataset_name)
model_path = os.path.join(args.data_directory, "models", args.dataset)

# Choose display_name
if args.display_name == "":
    display_name = f"{'{:.0e}'.format(args.lr)} lr,{'{:.0e}'.format(args.weight_decay)} wd,{args.bs} bs, {args.loss} loss, {args.pseudocolor}, {time}"
else:
    display_name = args.display_name
   


config = {
    "display_name": display_name,
    "base_model": args.base_model,
    "dataset": data_path,
    "checkpoint": model_path,
    "learning_rate": args.lr,
    "weight_decay": args.weight_decay,
    "epochs": args.epochs,
    "batch_size": args.bs,
    "shuffle": args.shuffle,
    "data_transforms": [],
    "optimizer": args.optimizer,
    "loss": args.loss,
    "time": time,
    "mode": modes[args.mode],
    "display_mode": args.display_mode, 
    "pseudocolor": OCV_COLORMAPS[args.pseudocolor],
    "evaluate": args.evaluate,
    "topological": args.top,
    "prompt_type": args.prompt
}

# Choose the mode for display_samples
if args.display_mode == "predefined":
    config["display_idx"] = list(map(int, args.display_idx.strip().split(",")))
elif args.display_mode != "none":
    config["display_val_nr"] = args.display_val_nr
    config["display_train_nr"] = args.display_train_nr


# TODO: Move to argparse?
display_modes = ["single_masks", "all_masks"]
#config["display_mode"] = display_modes[1]

# Select mask
if args.mode == 0:
    config["seg_nr"] = args.seg_nr

# Provide dataset specific info
if args.dataset == "custom":
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
    
elif args.dataset == "dme":
    pass

mask_dict = {k: v for k, v in enumerate(mask_dict)}
config.update({
    "mask_dict": mask_dict,
})


wandb.init(
    project=args.project_name,
    entity=args.entity,
    name=display_name,
    config=config,
    save_code=True,
    dir = "/vol/data/runs",
)

print("CONFIG: ", config)

training(args.base_model, config)