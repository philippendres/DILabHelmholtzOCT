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

#TODO: Try other alternatives as base models
# alternatives: ["facebook/sam-vit-base", "facebook/sam-vit-huge", "facebook/sam-vit-large", "wanglab/medsam-vit-base"]
base_model_name = "facebook/sam-vit-base"
processor = SamProcessor.from_pretrained(base_model_name)
model = SamModel.from_pretrained(base_model_name)

def transform_image(img):
    img = img.transpose(2,1,0)
    image = np.expand_dims(img, axis=3)
    image = np.repeat(255 - image, 3, axis=3)
    return image

def get_valid_idx(mask):
    idx = []
    for i in range(0,61):
        temp = mask[:,:,i]
        if np.nansum(temp) != 0:
            idx.append(i)
    return idx

def create_dataset(images, labels):
    dataset = Dataset.from_dict({"image": images,
                                 "label": labels})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset

def transforms(examples):
    transformed_images, transformed_masks = [], []
    for image, seg_mask in zip(examples["image"], examples["label"]):
        image, seg_mask = np.array(image), np.array(seg_mask)
        transformed = transform(image=image, mask=seg_mask)
        transformed_images.append(transformed["image"])
        transformed_masks.append(transformed["mask"])
    examples["pixel_values"] = transformed_images
    examples["label"] = transformed_masks
    return examples

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
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

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

#make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

mat_dme =[]
for i in range(10):
    number = str(i+1).zfill(2)
    mat_dme.append("..\data\Kaggle\Subject_"+number+".mat")


#TODO: Extend Code to multiple subjects
nr = 0
images = loadmat(mat_dme[nr])['images']
#TODO: Extend Code to manualFluid2
masks = loadmat(mat_dme[nr])['manualFluid1']
idx = get_valid_idx(masks)
masks = masks[:,:, idx]
images = images[:,:,idx]
images = transform_image(images)
#TODO: Extend Code to individual fluids (values 1,.. in the mask)
masks = np.where(masks == 0, 0, 1).transpose(2,1,0)
dataset = create_dataset(images=images, labels=masks)

#TODO: Explore Options for transformations
transform = albumentations.Compose([albumentations.Resize(256, 256),])
dataset.set_transform(transforms)

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
train_dataset = SAMDataset(dataset=dataset, processor=processor)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# Note: Hyperparameter tuning could improve performance here
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
#TODO: Test other losses, implement topological loss
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

# configure number of epochs
num_epochs = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
        # forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        # backward pass (compute gradients of parameters w.r.t. loss)
        optimizer.zero_grad()
        loss.backward()

        # optimize
        optimizer.step()
        epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')

torch.save(model.state_dict(), "../data/Kaggle/model_checkpoints/firsttry.pt")

#TODO Implement comparison with SAM and MedSAM (i.e. evaluation of common metrices)
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


### Compare MedSAM:
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


