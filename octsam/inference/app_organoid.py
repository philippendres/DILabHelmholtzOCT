#backend
import torch
from transformers import SamModel, SamProcessor
import torch.nn.functional as F
import gradio as gr
from gradio_image_prompter import ImagePrompter
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
#model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
checkpoint_path = "/vol/data/models/Organoid/topo+geom_box-prompt"
model=torch.load(checkpoint_path +".pth").to(device)

def inference(img, pixel, prompt_type):
    model.eval()
    with torch.no_grad():
        if (prompt_type=="points"):
            inputs = processor(img, input_points= [[pixel]], return_tensors="pt").to(device)
        else:
            inputs = processor(img, input_boxes=[[pixel]], return_tensors="pt").to(device)
        outputs= model(**inputs, multimask_output=False)
        masks = F.interpolate(outputs.pred_masks.squeeze(2), (1024,1024), mode="bilinear", align_corners=False)
        masks = masks[..., : inputs["reshaped_input_sizes"][0,0], : inputs["reshaped_input_sizes"][0,1]]
        masks = F.interpolate(masks, (inputs["original_sizes"][0,0],inputs["original_sizes"][0,1]), mode="bilinear", align_corners=False)
        masks = torch.sigmoid(masks).cpu().squeeze().numpy()
        binary_masks = (masks > 0.5).astype(np.uint8)
    return binary_masks


def segment(inputs):
    img = inputs["image"]
    masks = []
    for i in range(len(inputs["points"])):
        pixel = list(map(int, inputs["points"][i]))
        prompt = [pixel[0], pixel[1], pixel[3],pixel[4]]
        print(prompt)
        if pixel[3]==0 and pixel[4]==0:
            #point prompt
            mask = inference(img, [prompt[0], prompt[1]], "points")
            point = np.zeros(img.shape[:2])
            point[prompt[1]-1:prompt[1]+2, prompt[0]-1:prompt[0]+2] = 1
            masks.append((point,"point"))
            masks.append((mask, "mask"))
        else:
            #bbox prompt
            mask = inference(img, prompt, "bbox")
            masks.append((prompt,"box"))
            masks.append((mask, "mask"))
    return (img, masks)

demo = gr.Interface(
    segment,
    ImagePrompter(show_label=True),
    [gr.AnnotatedImage(
            color_map={"mask": "#ff0000", "box": "#00ff00", "point": "#0000ff"}
        )],
)

demo.launch(share=True)