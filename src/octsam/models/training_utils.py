# Work in progress

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
#from random import random
from scipy.ndimage import label

def training(base_model, config):
    processor, model = prepare_model(base_model)
    train_dataset, train_dataloader = prepare_data(processor, config["dataset"], "train", config)
    valid_dataset, valid_dataloader = prepare_data(processor, config["dataset"], "test", config)
    optimizer = Adam(model.mask_decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    #TODO: Test other losses, implement topological loss
    seg_loss = monai.losses.DiceCELoss(sigmoid=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    config["display_mode"] != "none" and display_samples(model, processor, device, train_dataset, "train", config)
    config["display_mode"] != "none" and display_samples(model, processor, device, valid_dataset, "test", config)
    for epoch in range(config["epochs"]):
        model.train()
        train_epoch_loss = 0
        for batch in tqdm(train_dataloader):
            # forward pass
            with torch.no_grad():
                image, points, gt_masks = batch
                gt_masks = gt_masks.to(device)
                optimizer.zero_grad()
                inputs = processor(image, input_points=points, return_tensors="pt").to(device)
            outputs = model(**inputs, multimask_output=False)
             
            #postprocessing
            masks = F.interpolate(outputs.pred_masks.squeeze(2), (1024,1024), mode="bilinear", align_corners=False)
            masks = masks[..., : 992, : 1024]
            masks = F.interpolate(masks, (496,512), mode="bilinear", align_corners=False)
            # compute loss
            train_loss = seg_loss(masks.squeeze(), gt_masks)
            # backward pass (compute gradients of parameters w.r.t. loss
            train_loss.backward()
            # optimize
            optimizer.step()
            train_epoch_loss += train_loss.item()
        train_epoch_loss = train_epoch_loss/len(train_dataloader)
        wandb.log({"train/train_loss": train_epoch_loss, "train/epoch": epoch})
        valid_epoch_loss = validate_model(model, processor, valid_dataloader, seg_loss, config)
        print(f'EPOCH: {epoch}, Train Loss: {train_epoch_loss}, Valid Loss: {valid_epoch_loss}')
        config["display_mode"] != "none" and display_samples(model, processor, device, train_dataset, "train", config)
        config["display_mode"] != "none" and display_samples(model, processor, device, valid_dataset, "test", config)
    torch.save(model.state_dict(), config["checkpoint"] + config["display_name"] + "_" + config["time"] +".pt")
    wandb.finish()

def prepare_model(base_model):
    processor = SamProcessor.from_pretrained(base_model)
    model = SamModel.from_pretrained(base_model)
    #make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    return processor, model

def prepare_data(processor, dataset, split, config):
    dataset = datasets.load_from_disk(dataset)[split]
    config["data_transforms"] and dataset.set_transform(data_transforms(operations=config["data_transforms"]))
    dataset = SAMDataset(dataset=dataset, processor=processor, config=config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], collate_fn=custom_collate)
    return dataset, dataloader

def data_transforms(batch, operations):
    transform = albumentations.Compose(operations)
    transformed_images, transformed_masks = [], []
    for image, seg_mask in zip(batch["image"], batch["label"]):
        image, seg_mask = np.array(image), np.array(seg_mask)
        transformed = transform(image=image, mask=seg_mask)
        transformed_images.append(transformed["image"])
        transformed_masks.append(transformed["mask"])
    batch["pixel_values"] = transformed_images
    batch["label"] = transformed_masks
    return batch

def display_samples(model, processor, device, dataset, split, config):
    model.eval()
    if config["display_mode"] == "predefined":
        idx = config["display_idx"]
    elif config["display_mode"] != "none":
        if config["display_mode"] == "random_equal":
            random.seed(17)
        elif config["display_mode"] == "random_changing":
            random.seed(time.time())
        if split == "train":
            idx = [random.randint(0, len(dataset) - 1) for i in range(config["display_train_nr"])]
        else:
            idx = [random.randint(0, len(dataset) - 1) for i in range(config["display_val_nr"])]
    img = []
    for i in idx:
        image, points, gt_masks = dataset[i]
        class_labels = config["mask_dict"]
        with torch.no_grad():
            inputs = processor(image, input_points=[points], return_tensors="pt")
            outputs = model(**inputs.to(device), multimask_output=False)
            #masks = F.interpolate(outputs.pred_masks[:,:,0,:,:], (1024,1024), mode="bilinear", align_corners=False)
            #masks = masks[..., : 992, : 1024]
            #masks = F.interpolate(masks, (496,512), mode="bilinear", align_corners=False)
            #masks = torch.argmax(masks, dim=1).squeeze()
            masks = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])
            gt_masks = torch.tensor(np.array(gt_masks))
            image_masks = {
                "pred": {"mask_data": masks[0].squeeze().cpu().float().numpy(), "class_labels": {0: 'background', 1: 'pred'}},
                "gt": {"mask_data": 2*gt_masks.numpy(), "class_labels": {0: 'background', 2: 'gt'}}
            }
            img.append(wandb.Image(
                image,
                masks=image_masks,
            ))
        
        wandb.log({split+"_samples" :img})
    model.train()

def validate_model(model, processor, valid_dl, seg_loss, config, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    valid_epoch_loss = 0.
    with torch.inference_mode():
        for batch in tqdm(valid_dl):
            # forward pass
            image, points, gt_masks = batch
            gt_masks = gt_masks.to(device)
            inputs = processor(image, input_points=points, return_tensors="pt").to(device)
            outputs = model(**inputs, multimask_output=False)
            masks = F.interpolate(outputs.pred_masks.squeeze(2), (1024,1024), mode="bilinear", align_corners=False)
            masks = masks[..., : 992, : 1024]
            masks = F.interpolate(masks, (496,512), mode="bilinear", align_corners=False)
            # compute loss
            valid_loss = seg_loss(masks.squeeze(), gt_masks)
            valid_epoch_loss += valid_loss
        valid_epoch_loss = valid_epoch_loss/len(valid_dl)
        wandb.log({"val/valid_loss": valid_epoch_loss})
    return valid_epoch_loss

class SAMDataset(TorchDataset):
    def __init__(self, dataset, processor, config):
        self.dataset = dataset
        self.processor = processor
        self.config = config

    def __len__(self):
        return len(self.dataset)

    """
    def get_bboxes_and_gt_masks(self, ground_truth_mask):
        # get bounding boxes from mask
        bboxes, gt_masks = [],[]
        mask_values, mask_counts = np.unique(ground_truth_mask, return_counts=True)
        #Comment for background prediction
        #mask_values, mask_counts = mask_values[1:], mask_counts[1:]
        for v in mask_values: 
            x_indices, y_indices = np.where(ground_truth_mask == v)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            H, W = ground_truth_mask.shape
            x_min = max(0, x_min)
            x_max = min(W, x_max)
            y_min = max(0, y_min)
            y_max = min(H, y_max)
            bbox = [x_min, y_min, x_max, y_max]
            bboxes.append(bbox)
            gt_mask = np.where(ground_truth_mask == v, 1.0, 0.0)
            gt_masks.append(gt_mask)
        return bboxes, gt_masks, mask_values, mask_counts
    """
    def get_points_and_gt_masks(self, ground_truth_mask):
        points, gt_masks = [],[]
        mask_values, mask_counts = np.unique(ground_truth_mask, return_counts=True)
        structure = np.ones((3, 3), dtype=np.int32)
        rand_mask_value = random.randrange(0, len(mask_values))
        #for v in mask_values: 
        gt_mask = np.where(ground_truth_mask == mask_values[rand_mask_value], 1.0, 0.0)
        labeled_gt_mask, ncomponents = label(gt_mask, structure)
        point = []
        for c in range(ncomponents):
            x_indices, y_indices = np.where(labeled_gt_mask == c+1)
            rand_idx = random.randrange(0, len(x_indices))
            point.append([x_indices[rand_idx], y_indices[rand_idx]])
        return point, gt_mask

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = np.array(item["image"])
        if (self.config["pseudocolor"] != None):
            image = cv2.applyColorMap(image[:, :, 0], self.config["pseudocolor"])
        ground_truth_mask = np.array(item["label"])
        # get point prompt
        points, gt_masks = self.get_points_and_gt_masks(ground_truth_mask)
        return [image, points, gt_masks]
    
def custom_collate(data):
    images = [d[0] for d in data]   
    bboxes = [torch.tensor(d[1]) for d in data]
    gt_masks = [torch.tensor(np.array(d[2])) for d in data]

    images = torch.tensor(np.array(images))
    bboxes = pad_sequence(bboxes, batch_first=True)
    gt_masks = pad_sequence(gt_masks, batch_first=True)

    return [images, bboxes, gt_masks]

