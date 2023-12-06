from kaggle_dme_utils import transforms, show_mask, get_bounding_box
from transformers import SamProcessor, SamModel
import torch
import numpy as np
from PIL import Image
import datasets
import matplotlib.pyplot as plt
import evaluate

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

base_models = ["facebook/sam-vit-base", "facebook/sam-vit-huge", "facebook/sam-vit-large", "wanglab/medsam-vit-base"]
base_model = base_models[0]
data_path = "../../data/OCT/Kaggle/DME/preprocessed/"
preprocessed_dataset = "23-12-04 20.54.30"
checkpoint_path = "../../models/OCT/model_checkpoints/23-12-06 13.22.42chkpt.pt"


train_dataset = datasets.load_from_disk(data_path + preprocessed_dataset)["train"]
train_dataset.set_transform(transforms)

valid_dataset = datasets.load_from_disk(data_path + preprocessed_dataset)["test"]
valid_dataset.set_transform(transforms)

# let's take a random training example
idx = 1

# load image
train_image = train_dataset[idx]["pixel_values"]
valid_image = valid_dataset[idx]["pixel_values"]

image = valid_image
# get box prompt based on ground truth segmentation map
ground_truth_mask = np.array(valid_dataset[idx]["label"])
prompt = get_bounding_box(ground_truth_mask)
prompt = [0,0,254,254]

model_name = "facebook/sam-vit-base"
#model_name = "wanglab/medsam-vit-base"
processor = SamProcessor.from_pretrained(model_name)
model = SamModel.from_pretrained(model_name)
model.to("cuda")

#states = torch.load(checkpoint_path)
#model.load_state_dict(states, strict=False)

inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to("cuda")
model.eval()
# forward pass
with torch.no_grad():
    outputs = model(**inputs, multimask_output=False)
# apply sigmoid
medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
"""
plt.ion()
fig, axes = plt.subplots()

axes.imshow(np.array(image))
#show_mask(medsam_seg, axes)
show_mask(medsam_seg, axes)
axes.title.set_text(f"Predicted mask MedSAM")
#axes.title.set_text(f"Ground Truth")
#axes.title.set_text(f"Raw")
#show_box(prompt, axes)
axes.axis("off")
plt.show()
"""
metric = evaluate.load("mean_iou")
print("SAM:")
metric = metric.compute(
    predictions=[medsam_seg],
    references=[ground_truth_mask],
    num_labels=2,
    ignore_index=255,
    reduce_labels=False,
)
print(metric)