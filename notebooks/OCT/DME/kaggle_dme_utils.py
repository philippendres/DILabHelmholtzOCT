import datasets
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
from torch.utils.tensorboard import SummaryWriter
import datetime
import wandb

def training(base_model, config):
    processor = SamProcessor.from_pretrained(base_model)
    model = SamModel.from_pretrained(base_model)
    #make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    train_dataset = datasets.load_from_disk(config["dataset"])["train"]
    train_dataset.set_transform(transforms)
    train_sam_dataset = SAMDataset(dataset=train_dataset, processor=processor)
    train_dataloader = DataLoader(train_sam_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_dataset = datasets.load_from_disk(config["dataset"])["test"]
    valid_dataset.set_transform(transforms)
    valid_sam_dataset = SAMDataset(dataset=valid_dataset, processor=processor)
    valid_dataloader = DataLoader(valid_sam_dataset, batch_size=config["batch_size"], shuffle=True)
    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(model.mask_decoder.parameters(), lr=config["learning_rate"], weight_decay=0)
    #TODO: Test other losses, implement topological loss
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    idx = 7
    train_image = train_dataset[idx]["pixel_values"]
    valid_image = valid_dataset[idx]["pixel_values"]
    train_gt = np.array(train_dataset[idx]["label"])
    valid_gt = np.array(valid_dataset[idx]["label"])
    prompt = [0,0,254,254]
    class_labels = {1: "segmentation"}
    with torch.no_grad():
        inputs = processor(train_image, input_boxes=[[prompt]], return_tensors="pt").to("cuda")
        model.eval()
        outputs = model(**inputs, multimask_output=False)
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        mask_img = wandb.Image(
            train_image,
            masks={
                "predictions": {"mask_data": medsam_seg, "class_labels": class_labels},
                "ground_truth": {"mask_data": train_gt, "class_labels": class_labels},
            },
        )
        wandb.log({"predictions" :[mask_img]})
    for epoch in range(config["epochs"]):
        model.train()
        train_epoch_loss = 0
        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            train_loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            train_loss.backward()
            # optimize
            optimizer.step()
            train_epoch_loss += train_loss.item()
        wandb.log({"train/train_loss": train_epoch_loss,
                   "train/epoch": epoch})
        valid_epoch_loss = validate_model(model, valid_dataloader, seg_loss, epoch)

        print(f'EPOCH: {epoch}, Train Loss: {train_epoch_loss}, Valid Loss: {valid_epoch_loss}')
        with torch.no_grad():
            inputs = processor(train_image, input_boxes=[[prompt]], return_tensors="pt").to("cuda")
            model.eval()
            outputs = model(**inputs, multimask_output=False)
            medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            mask_img = wandb.Image(
                train_image,
                masks={
                    "predictions": {"mask_data": medsam_seg, "class_labels": class_labels},
                    "ground_truth": {"mask_data": train_gt, "class_labels": class_labels},
                },
            )
            wandb.log({"predictions" :[mask_img]})
    torch.save(model.state_dict(), config["checkpoint"] + "chkpt.pt")
    wandb.finish()


def transforms(batch):
    transform = albumentations.Compose([albumentations.Resize(256, 256),])
    transformed_images, transformed_masks = [], []
    for image, seg_mask in zip(batch["image"], batch["label"]):
        image, seg_mask = np.array(image), np.array(seg_mask)
        seg_mask[0,0] = 1
        transformed = transform(image=image, mask=seg_mask)
        transformed_images.append(transformed["image"])
        transformed_masks.append(transformed["mask"])
    batch["pixel_values"] = transformed_images
    batch["label"] = transformed_masks
    return batch

def validate_model(model, valid_dl, seg_loss, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    valid_epoch_loss = 0.
    with torch.inference_mode():
        for batch in tqdm(valid_dl):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            valid_loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            # backward pass (compute gradients of parameters w.r.t. loss)
            valid_epoch_loss += valid_loss.item()
        wandb.log({"val/valid_loss": valid_epoch_loss})
    return valid_epoch_loss




def visualize_seg_mask(image: np.ndarray, mask: np.ndarray):
    color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_seg[mask == 0, :] = [255,255,255]
    #color_seg[mask == 1, :] = [255,255,255]
    color_seg[mask == 1, :] = [255,0,0]
    color_seg = color_seg[..., ::-1]  # convert to BGR
    img = np.array(image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map
    img = img.astype(np.uint8)
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def get_bounding_box(ground_truth_map):
    """
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min)
    x_max = min(W, x_max)
    y_min = max(0, y_min)
    y_max = min(H, y_max)
    bbox = [x_min, y_min, x_max, y_max]
    """
    bbox =[0,0,255,255]
    return bbox

class SAMDataset(TorchDataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["pixel_values"]
        ground_truth_mask = np.array(item["label"])
        # get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)
        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask
        return inputs

#TODO: Implement visualization of samples during the training
#Visualization
"""
visualize_seg_mask(
    np.array(dataset[index]["pixel_values"]),
    np.array(dataset[index]["label"])
)
"""
"""
fig, axes = plt.subplots()
axes.imshow(np.array(image))
ground_truth_seg = np.array(example["label"])
show_mask(ground_truth_seg, axes)
axes.title.set_text(f"Ground truth mask")
axes.axis("off")
"""




"""
Evaluation:
idx = 3
# load image
image = dataset[idx]["pixel_values"]

# get box prompt based on ground truth segmentation map
ground_truth_mask = np.array(dataset[idx]["label"])
prompt = get_bounding_box(ground_truth_mask)

# prepare image + box prompt for the model
inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
model.eval()

# forward pass
with torch.no_grad():
    outputs = model(**inputs, multimask_output=False)

# apply sigmoid
medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

#Plotting
fig, axes = plt.subplots()
axes.imshow(np.array(image))
show_mask(medsam_seg, axes)
axes.title.set_text(f"Predicted mask")
axes.axis("off")

#Ground truth
fig, axes = plt.subplots()
axes.imshow(np.array(image))
show_mask(ground_truth_mask, axes)
axes.title.set_text(f"Ground truth mask")
axes.axis("off")

### Compare SAM:
model_sam = SamModel.from_pretrained("facebook/sam-vit-base")
model_sam.to(device)
model_sam.eval()
with torch.no_grad():
    outputs = model_sam(**inputs, multimask_output=False)
# apply sigmoid
medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
fig, axes = plt.subplots()
axes.imshow(np.array(image))
show_mask(medsam_seg, axes)
axes.title.set_text(f"Predicted mask")
axes.axis("off")


### Compare MedSAM_checkpoint:
processor_medsam = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
model_medsam = SamModel.from_pretrained("wanglab/medsam-vit-base").to(device)
model_sam.eval()
with torch.no_grad():
    outputs = model_medsam(**inputs, multimask_output=False)
# apply sigmoid
medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
fig, axes = plt.subplots()
axes.imshow(np.array(image))
show_mask(medsam_seg, axes)
axes.title.set_text(f"Predicted mask")
axes.axis("off")
"""