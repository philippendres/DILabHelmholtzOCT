import os
import datetime
from preprocessing_utils import preprocess

time = datetime.datetime.now().strftime('%y-%m-%d_%H.%M.%S')

# Gather data path
datasets = ["custom", "dme", "amd"]
#"amd": alias for https://www.kaggle.com/datasets/paultimothymooney/farsiu-2014
#"dme": alias for https://www.kaggle.com/datasets/paultimothymooney/chiu-2015
#"custom": alias for data obtained from Valentin
dataset = datasets[0]
data_directory = "/vol/data/datasets"
raw_data_path = os.path.join(data_directory, "raw", dataset)
processed_data_path = os.path.join(data_directory, "processed", dataset)


preprocessing_config = {
    "test_size": 0.2,
    "shuffle": True,
    "time": time,
    "print_status":True,
    "additional_file_description": "default_"
}

if dataset == "dme":
    use_masks = [
        'manualLayers1', 'manualLayers2', 'automaticLayersDME', 'automaticLayersNormal', 
        'manualFluid1', 'manualFluid2', 'automaticFluidDME'
    ]
    separate_fluids = True
    preprocessing_config.update({
        "use_masks": use_masks[0]
    })
    preprocessing_config["additional_file_description"] = preprocessing_config["use_masks"] + "_" 
elif dataset == "amd":
    raise NotImplementedError()

preprocess(dataset, raw_data_path, processed_data_path, preprocessing_config)