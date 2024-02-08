from torch.utils.data import DataLoader
from torch.utils.data import DataLoader as TorchDataset
from torch.optim import Adam
import wandb
import monai
from tqdm import tqdm
import numpy as np
import datasets
from transformers import SamProcessor, SamModel
from statistics import mean
import torch
from torch.nn.utils.rnn import pad_sequence 
import torch.nn.functional as F
import cv2
import random
import time
from scipy.ndimage import label
import evaluate
import sklearn.metrics
# Model info
# base_models = ["facebook/sam-vit-base", "facebook/sam-vit-huge", "facebook/sam-vit-large", "wanglab/medsam-vit-base"]
base_model = "facebook/sam-vit-base"
checkpoint_path = "/vol/data/models/custom_24-01-24_23.49.46"
dataset_path = "/vol/data/datasets/processed/custom/default_preprocessed_at_24-01-10_13.41.28"
pseudocolor = "Bone"
prompt_type = "bboxes"

# TODO: Add to config
NO_BEST_WORST_SAMPLES = 2


###
def evaluate_metrics(model, dataset, config, processor):
    #processor = SamProcessor.from_pretrained(config["base_model"])
    #model = SamModel.from_pretrained(config["base_model"])
    #model.load_state_dict(torch.load(config["checkpoint"] + config["display_name"] + "_" + config["time"] +".pt"))
    #dataset = datasets.load_from_disk(config["dataset"])["test"]
    #dataset = SAMDataset(dataset=dataset, processor=processor, config=config)
    model.eval()
    metric_iou = evaluate.load("mean_iou")
    segmentations = []
    segmentations_probas = []
    ground_truths = []
    indexes = []
    category_accuracies = np.zeros(14)
    category_ious = np.zeros(14)
    category_f1 = np.zeros(14)
    category_dice = np.zeros(14)
    category_spec = np.zeros(14)
    category_sens = np.zeros(14)
    category_map = np.zeros(14)
    for i in range(14):
        segmentations.append([])
        ground_truths.append([])
        segmentations_probas.append([])
        indexes.append([])
    for i in tqdm(range(15)):
        with torch.no_grad():
            if (config["prompt_type"]=="points"):
                image, points, gt_masks, mask_values = dataset[i]
                inputs = processor(image, input_points= [points], return_tensors="pt")
            else:
                image, bboxes, gt_masks, mask_values = dataset[i]
                inputs = processor(image, input_boxes=[bboxes], return_tensors="pt")
            outputs= model(**inputs, multimask_output=False)
            masks = torch.zeros(1, 14, 496, 512)
            masks = F.interpolate(outputs.pred_masks.squeeze(2), (1024,1024), mode="bilinear", align_corners=False)
            masks = masks[..., : inputs["reshaped_input_sizes"][0,0], : inputs["reshaped_input_sizes"][0,1]]
            masks = F.interpolate(masks, (inputs["original_sizes"][0,0],inputs["original_sizes"][0,1]), mode="bilinear", align_corners=False)
            masks = torch.sigmoid(masks).squeeze().numpy()
            binary_masks = (masks > 0.5).astype(np.uint8)
            for c in range(len(mask_values)):
                if mask_values[c] == 0 and c > 0:
                    break
                segmentations[mask_values[c]].append(binary_masks[c])
                segmentations_probas[mask_values[c]].append(masks[c])
                ground_truths[mask_values[c]].append(gt_masks[c])
                indexes[mask_values[c]].append(i)
        
    for i in range(14):
        print(f"------------------CLASS: {mask_dict[i]}----------------------")
        metric_output = metric_iou.compute(
            predictions=segmentations[i],
            references=ground_truths[i],
            ignore_index=255,
            num_labels=2,
            reduce_labels=False,
        )
        category_accuracies[i] = metric_output['per_category_accuracy'][1]
        category_ious[i] = metric_output['per_category_iou'][1]
        flat_gt = np.array(ground_truths[i]).reshape(-1)
        flat_seg = np.array(segmentations[i]).reshape(-1)
        flat_segp = np.array(segmentations_probas[i]).reshape(-1)
        
        category_f1[i] = sklearn.metrics.f1_score(flat_gt, flat_seg)
        category_map[i] = sklearn.metrics.average_precision_score(flat_gt, flat_segp)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(flat_gt, flat_seg).ravel()
        category_sens[i] = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        category_spec[i] = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        category_dice[i] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
        
        sample_iou = []
        sample_accuracy = []
        sample_spec = []
        sample_sens = []
        sample_f1 = []
        sample_dice = []
        sample_ap = []
        
        for j in range(len(segmentations[i])):
            metric_output = metric_iou.compute(
                predictions=[segmentations[i][j]],
                references=[ground_truths[i][j]],
                ignore_index=255,
                num_labels=2,
                reduce_labels=False,
            )
            flat_gt = np.array(ground_truths[i][j]).reshape(-1)
            flat_seg = np.array(segmentations[i][j]).reshape(-1)
            flat_segp = np.array(segmentations_probas[i][j]).reshape(-1)
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(flat_gt, flat_seg).ravel()
            sample_iou.append(metric_output['per_category_iou'][1])
            sample_accuracy.append(metric_output['per_category_accuracy'][1])
            sample_spec.append(tn / (tn + fp) if (tn + fp) != 0 else 0.0)
            sample_sens.append(tp / (tp + fn) if (tp + fn) != 0 else 0.0)
            sample_f1.append(sklearn.metrics.f1_score(flat_gt, flat_seg))
            sample_dice.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0)
            sample_ap.append(sklearn.metrics.average_precision_score(flat_gt, flat_segp))
        
        avg_start_idx = len(sample_iou) // 2 - NO_BEST_WORST_SAMPLES // 2
        avg_end_idx = len(sample_iou) // 2 + NO_BEST_WORST_SAMPLES // 2
        idx = np.array(indexes[i])
        
        print(f"GENERAL REPORT:")
        print(metric_output)
        print(f"----IoU----:")
        print(f"{category_ious[i]} \ {np.mean(sample_iou)}")
        print(f"Best samples: {idx[np.argsort(sample_iou)[-NO_BEST_WORST_SAMPLES:]]}")
        print(f"Average samples: {idx[np.argsort(sample_iou)[avg_start_idx:avg_end_idx]]}")
        print(f"Worst samples: {idx[np.argsort(sample_iou)[:NO_BEST_WORST_SAMPLES]]}")
        print(f"----Accuracy----:")
        print(f"{category_accuracies[i]} \ {np.mean(sample_accuracy)}")
        print(f"Best samples: {idx[np.argsort(sample_accuracy)[-NO_BEST_WORST_SAMPLES:]]}")
        print(f"Average samples: {idx[np.argsort(sample_accuracy)[avg_start_idx:avg_end_idx]]}")
        print(f"Worst samples: {idx[np.argsort(sample_accuracy)[:NO_BEST_WORST_SAMPLES]]}")
        print(f"----Specificity----:")
        print(f"{category_spec[i]} \ {np.mean(sample_spec)}")
        print(f"Best samples: {idx[np.argsort(sample_spec)[-NO_BEST_WORST_SAMPLES:]]}")
        print(f"Average samples: {idx[np.argsort(sample_spec)[avg_start_idx:avg_end_idx]]}")
        print(f"Worst samples: {idx[np.argsort(sample_spec)[:NO_BEST_WORST_SAMPLES]]}")
        print(f"----Sensitivity----:")
        print(f"{category_sens[i]} \ {np.mean(sample_sens)}")
        print(f"Best samples: {idx[np.argsort(sample_sens)[-NO_BEST_WORST_SAMPLES:]]}")
        print(f"Average samples: {idx[np.argsort(sample_sens)[avg_start_idx:avg_end_idx]]}")
        print(f"Worst samples: {idx[np.argsort(sample_sens)[:NO_BEST_WORST_SAMPLES]]}")
        print(f"----F1----:")
        print(f"{category_f1[i]} \ {np.mean(sample_f1)}")
        print(f"Best samples: {idx[np.argsort(sample_f1)[-NO_BEST_WORST_SAMPLES:]]}")
        print(f"Average samples: {idx[np.argsort(sample_f1)[avg_start_idx:avg_end_idx]]}")
        print(f"Worst samples: {idx[np.argsort(sample_f1)[:NO_BEST_WORST_SAMPLES]]}")
        print(f"----Dice----:")
        print(f"{category_dice[i]} \ {np.mean(sample_dice)}")
        print(f"Best samples: {idx[np.argsort(sample_dice)[-NO_BEST_WORST_SAMPLES:]]}")
        print(f"Average samples: {idx[np.argsort(sample_dice)[avg_start_idx:avg_end_idx]]}")
        print(f"Worst samples: {idx[np.argsort(sample_dice)[:NO_BEST_WORST_SAMPLES]]}")
        print(f"----AP----:")
        print(f"{category_map[i]} \ {np.mean(sample_ap)}")
        print(f"Best samples: {idx[np.argsort(sample_ap)[-NO_BEST_WORST_SAMPLES:]]}")
        print(f"Average samples: {idx[np.argsort(sample_ap)[avg_start_idx:avg_end_idx]]}")
        print(f"Worst samples: {idx[np.argsort(sample_ap)[:NO_BEST_WORST_SAMPLES]]}")
        
        
        f = open(config["results_path"], "a")
        f.write(str(metric_output)+"\n")
        f.close()
    f = open(config["results_path"], "a")
    f.write("Category_accuracies:" + str(list(category_accuracies))+"\n"+"Category_ious:"+str(list(category_ious))+"\n")
    f.close()
    mean_iou = np.mean(category_ious)
    mean_accuracy = np.mean(category_accuracies)
    f = open(config["results_path"], "a")
    f.write("Mean_accuracy:" + str(mean_accuracy)+"\n"+"Mean_iou:"+str(mean_iou))
    f.close()

class SAMDataset(TorchDataset):
    def __init__(self, dataset, processor, config):
        self.dataset = dataset
        self.processor = processor
        self.config = config

    def __len__(self):
        return len(self.dataset)

    def get_bboxes_and_gt_masks(self, ground_truth_mask):
        # get bounding boxes from mask
        structure = np.ones((3, 3), dtype=np.int32)
        bboxes, gt_masks = [],[]
        mask_values= np.unique(ground_truth_mask)
        final_mask_values = []
        #Comment for background prediction
        #mask_values = mask_values[1:]
        for v in mask_values: 
            binary_gt_mask = np.where(ground_truth_mask == v, 1.0, 0.0)
            labeled_gt_mask, ncomponents = label(binary_gt_mask, structure)
            for c in range(ncomponents):
                final_mask_values.append(v)
                x_indices, y_indices = np.where(labeled_gt_mask== c+1)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                # add perturbation to bounding box coordinates
                H, W = ground_truth_mask.shape
                x_min = max(0, x_min + np.random.randint(-10, 10))
                x_max = min(W, x_max + np.random.randint(-10, 10))
                y_min = max(0, y_min + np.random.randint(-10, 10))
                y_max = min(H, y_max + np.random.randint(-10, 10))
                bbox = [x_min, y_min, x_max, y_max]
                bboxes.append(bbox)
                gt_mask = np.where(labeled_gt_mask== c+1, 1.0, 0.0)
                gt_masks.append(gt_mask)
        return bboxes, gt_masks, final_mask_values

    def get_points_and_gt_masks(self, ground_truth_mask):
        structure = np.ones((3, 3), dtype=np.int32)
        points, gt_masks = [],[]
        mask_values= np.unique(ground_truth_mask)
        final_mask_values = []
        #Comment for background prediction
        #mask_values= mask_values[1:]
        for v in mask_values: 
            binary_gt_mask = np.where(ground_truth_mask == v, 1.0, 0.0)
            labeled_gt_mask, ncomponents = label(binary_gt_mask, structure)
            for c in range(ncomponents):
                final_mask_values.append(v)
                x_indices, y_indices = np.where(labeled_gt_mask== c+1)
                rand_idx = random.randrange(0, len(x_indices))
                points.append([[x_indices[rand_idx], y_indices[rand_idx]]])
                gt_mask = np.where(labeled_gt_mask== c+1, 1.0, 0.0)
                gt_masks.append(gt_mask)
        return points, gt_masks, final_mask_values

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = np.array(item["image"])
        if (self.config["pseudocolor"] != None):
            image = cv2.applyColorMap(image[:, :, 0], self.config["pseudocolor"])
        ground_truth_mask = np.array(item["label"])
        if (self.config["prompt_type"]=="points"):
            points, gt_masks, mask_values= self.get_points_and_gt_masks(ground_truth_mask)
            return [image, points, gt_masks, mask_values]
        else:
            bboxes, gt_masks, mask_values= self.get_bboxes_and_gt_masks(ground_truth_mask)
            return [image, bboxes, gt_masks, mask_values]

#Colormap
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
# mask_dict
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

processor = SamProcessor.from_pretrained(base_model)
model = SamModel.from_pretrained(base_model)
model.load_state_dict(torch.load(checkpoint_path +".pt"))
dataset = datasets.load_from_disk(dataset_path)["test"]
config ={
    "pseudocolor": OCV_COLORMAPS[pseudocolor],
    "prompt_type": prompt_type,
    "mask_dict": mask_dict,
    "results_path": checkpoint_path + ".txt"
}
dataset = SAMDataset(dataset=dataset, processor=processor, config=config)
evaluate_metrics(model, dataset, config, processor)