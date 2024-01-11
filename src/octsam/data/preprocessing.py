import os
import datetime
from preprocessing_utils import preprocess
import argparse

time = datetime.datetime.now().strftime('%y-%m-%d_%H.%M.%S')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="custom")
parser.add_argument("--data_directory", type=str, default="/vol/data/datasets")
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--shuffle", type=bool, default=True)
parser.add_argument("--dme_masks", type=str, default="manualLayers1")
args = parser.parse_args()

# Gather data path
#"amd": alias for https://www.kaggle.com/datasets/paultimothymooney/farsiu-2014
#"dme": alias for https://www.kaggle.com/datasets/paultimothymooney/chiu-2015
#"custom": alias for data obtained from Valentin
raw_data_path = os.path.join(args.data_directory, "raw", args.dataset)
processed_data_path = os.path.join(args.data_directory, "processed", args.dataset)


preprocessing_config = {
    "test_size": args.test_size,
    "shuffle": args.shuffle,
    "time": time,
    "print_status":True,
    "additional_file_description": "default_"
}

if args.dataset == "dme":
    #use_masks = [
    #    'manualLayers1', 'manualLayers2', 'automaticLayersDME', 'automaticLayersNormal', 
    #    'manualFluid1', 'manualFluid2', 'automaticFluidDME'
    #]
    separate_fluids = True
    preprocessing_config.update({
        "use_masks": args.dme_masks
    })
    preprocessing_config["additional_file_description"] = preprocessing_config["use_masks"] + "_" 
elif args.dataset == "amd":
    raise NotImplementedError()

preprocess(args.dataset, raw_data_path, processed_data_path, preprocessing_config)