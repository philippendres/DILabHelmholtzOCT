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

def training(base_model, config):
    processor, model = prepare_model(base_model)
    train_dataset, train_dataloader = prepare_data(processor, config["dataset"], "train", config)
    valid_dataset, valid_dataloader = prepare_data(processor, config["dataset"], "test", config)
    optimizer = Adam(model.mask_decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    #TODO: Test other losses, implement topological loss
    seg_loss = monai.losses.DiceCELoss(include_background = True, sigmoid=True, squared_pred=True, reduction='mean')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    config["display_samples"] != "no" and display_samples(model, processor, device, train_dataset, "train", config)
    config["display_samples"] != "no" and display_samples(model, processor, device, valid_dataset, "valid", config)
    for epoch in range(config["epochs"]):
        model.train()
        train_epoch_loss = 0
        first = True
        for batch in tqdm(train_dataloader):
            if first:
                first = False
                continue
            # forward pass
            with torch.no_grad():
                image, bboxes, gt_masks, mask_values, mask_counts = batch
                gt_masks = gt_masks.to(device)
                optimizer.zero_grad()
                inputs = processor(image, input_boxes=bboxes, return_tensors="pt").to(device)
            outputs = model(**inputs, multimask_output=False)
            #postprocessing
            interpolated_mask = F.interpolate(outputs.pred_masks[0], (1024,1024), mode="bilinear", align_corners=False)
            interpolated_mask = interpolated_mask[..., : 992, : 1024]
            interpolated_mask = F.interpolate(interpolated_mask, (496,512), mode="bilinear", align_corners=False)
            masks = interpolated_mask
            #masks = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])
            # compute loss
            train_loss = 0
            
            for i, m in enumerate(mask_values.squeeze()):
                total_mask_count = torch.sum(mask_counts.squeeze()[i])
                #total_class_count = torch.sum(torch.where(m>0,1.0,0.0))
                #print(1/total_mask_count * seg_loss(masks[i].squeeze(), gt_masks.squeeze()[i]))
                train_loss += 1/total_mask_count * seg_loss(masks[i].squeeze(), gt_masks.squeeze()[i])
            # backward pass (compute gradients of parameters w.r.t. loss)
            
            train_loss.backward()
            # optimize
            optimizer.step()
            train_epoch_loss += train_loss.item()
            break
        wandb.log({"train/train_loss": train_epoch_loss, "train/epoch": epoch})
        valid_epoch_loss = validate_model(model, processor, valid_dataloader, seg_loss, config)
        print(f'EPOCH: {epoch}, Train Loss: {train_epoch_loss}, Valid Loss: {valid_epoch_loss}')
        config["display_samples"] != "no" and display_samples(model, processor, device, train_dataset, "train", config)
        config["display_samples"] != "no" and display_samples(model, processor, device, valid_dataset, "valid", config)
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
    dataset = SAMDataset(dataset=dataset, processor=processor)
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
    idx = select_display_indices(dataset, config)
    img = []
    for i in idx:
        image, bboxes, gt_masks, mask_values, mask_counts = dataset[i]
        class_labels = config["mask_dict"]
        with torch.no_grad():
            
            inputs = processor(image, input_boxes=[bboxes], return_tensors="pt")
            outputs = model(**inputs.to(device), multimask_output=False)
            masks = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])
            if config["display_mode"] == "single_masks":
                image_masks = {}
                for i, m in enumerate(mask_values):
                    image_masks.update({
                        config["mask_dict"][m] + "_" + "pred": {"mask_data": masks[0][i,:,:,:].squeeze().cpu().float().numpy(), "class_labels": {0: "background", 1: class_labels[m]}},
                        config["mask_dict"][m] + "_" + "gt": {"mask_data": 2*gt_masks[i], "class_labels": {0: "background", 2: class_labels[m]}},
                    })
            elif config["display_mode"] == "all_masks":
                mask, gt = np.zeros_like(gt_masks[0]), np.zeros_like(gt_masks[0])
                enumerated_values = [(i, mask_values[i], mask_counts[i]) for i in range(len(mask_values))]
                enumerated_values.sort(key=lambda a: a[2], reverse=True)
                for m in enumerated_values:
                    gt += m[1] * gt_masks[m[0]]
                    boolean_mask = masks[0][m[0],:,:,:].squeeze().cpu()
                    mask[boolean_mask] = m[1]*masks[0][m[0],:,:,:].squeeze().cpu().float()[boolean_mask]
                image_masks = {
                    "pred": {"mask_data": mask, "class_labels": class_labels},
                    # class_labels could be modified to show the difference better
                    "gt": {"mask_data": gt, "class_labels": class_labels}
                }
            img.append(wandb.Image(
                image,
                masks=image_masks,
            ))
        
        wandb.log({split+"_samples" :img})
    model.train()

def select_display_indices(dataset, config):
    return [1]

def validate_model(model, processor, valid_dl, seg_loss, config, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    epoch_loss = 0.
    with torch.inference_mode():
        for batch in tqdm(valid_dl):
            # forward pass
            image, bboxes, gt_masks, mask_values, mask_counts = batch
            gt_masks = gt_masks.to(device)
            inputs = processor(image, input_boxes=bboxes, return_tensors="pt").to(device)
            outputs = model(**inputs, multimask_output=False)
            masks = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])
            # compute loss
            train_loss = 0
            
            for i, m in enumerate(mask_values.squeeze()):
                total_mask_count = torch.sum(mask_counts.squeeze()[i])
                total_class_count = torch.sum(torch.where(m>0,1.0,0.0))
                train_loss += 1/total_mask_count * seg_loss(masks[0].squeeze()[i].float(), gt_masks.squeeze()[i])
            epoch_loss += train_loss
            break
        wandb.log({"val/valid_loss": epoch_loss})
    return epoch_loss

class SAMDataset(TorchDataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def get_bboxes_and_gt_masks(self, ground_truth_mask):
        # get bounding boxes from mask
        bboxes, gt_masks = [],[]
        mask_values, mask_counts = np.unique(ground_truth_mask, return_counts=True)
        mask_values, mask_counts = mask_values[1:], mask_counts[1:]
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

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = np.array(item["image"])
        ground_truth_mask = np.array(item["label"])
        # get bounding box prompt
        bboxes, gt_masks, mask_values, mask_counts = self.get_bboxes_and_gt_masks(ground_truth_mask)
        # prepare image and prompt for the model
        #inputs = self.processor(image, input_boxes=[bboxes], return_tensors="pt")
        # remove batch dimension which the processor adds by default
        #inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        # add ground truth segmentation
        return [image, bboxes, gt_masks, mask_values, mask_counts]
    
def custom_collate(data):
    images = [d[0] for d in data]   
    bboxes = [torch.tensor(d[1]) for d in data]
    gt_masks = [torch.tensor(np.array(d[2])) for d in data]
    mask_values = [torch.tensor(d[3]) for d in data]
    mask_counts = [torch.tensor(d[4]) for d in data]

    images = torch.tensor(np.array(images))
    bboxes = pad_sequence(bboxes, batch_first=True)
    gt_masks = pad_sequence(gt_masks, batch_first=True)
    mask_values = pad_sequence(mask_values, batch_first=True)
    mask_counts = pad_sequence(mask_counts, batch_first=True)

    return [images, bboxes, gt_masks, mask_values, mask_counts]

