import numpy as np
import pickle
from segment_anything.utils import transforms
device = "cuda"
import torch
from segment_anything.modeling import sam
from segment_anything import sam_model_registry
from torch.utils.data import DataLoader
import data_loading_kaggle
from torch.utils.data import Dataset
from torchvision import datasets
from torch.nn.functional import threshold, normalize

number_of_mats = 1
number_of_shots = 100
number_of_segmentations = 4
longest_side = 1000
device = "cuda"

sam_checkpoint = "../data/SAM_checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_model.to(device=device)


full_dataset = data_loading_kaggle.CustomImageDataset(number_of_mats, number_of_shots, number_of_segmentations, longest_side, device, sam_model)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters())
loss_fn = torch.nn.MSELoss()

epochs = 1
for e in range(epochs):
    running_loss = 0.
    last_loss = 0.
    for i,data in enumerate(train_dataloader):
        if i ==0:
            print("Training started")
        image, bbox, gt = data
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(image)
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None,
            )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        upscaled_masks = sam_model.postprocess_masks(low_res_masks, (256,256), (512,1000)).to(device)
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(device)
        loss = loss_fn(binary_mask, gt.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

torch.save(sam_model.state_dict(), "../data/Kaggle/model_checkpoints/chkpt1.pt")