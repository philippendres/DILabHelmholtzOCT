# Work in progress
from typing import Dict, List, Optional, Tuple, Union
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
#from torch_topological.nn import SignatureLoss
#from torch_topological.nn import VietorisRipsComplex
import copy
import evaluate
import sklearn
NO_BEST_WORST_SAMPLES = 3

def training(base_model, config):
    if config["topological"]:
        #vr = VietorisRipsComplex(dim=1)
        #topo_loss = SignatureLoss()
        lamda = 1.0
    processor, model = prepare_model(base_model)
    train_dataset, train_dataloader = prepare_data(processor, config["dataset"], "train", config)
    valid_dataset, valid_dataloader = prepare_data(processor, config["dataset"], "test", config)
    #TODO: Test other losses, implement topological loss
    seg_loss = monai.losses.DiceCELoss(softmax=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    mask_decoders = [copy.deepcopy(model.mask_decoder) for i in range(config["nr_of_decoders"])]
    mask_decoders_params = []
    for i in range(config["nr_of_decoders"]):
        for j in list(mask_decoders[i].parameters()):
            mask_decoders_params.append(j)

    optimizer = Adam(mask_decoders_params, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    config["display_samples"] != "no" and display_samples(model, mask_decoders, processor, device, train_dataset, "train", config)
    config["display_samples"] != "no" and display_samples(model, mask_decoders, processor, device, valid_dataset, "valid", config)
    for epoch in range(config["epochs"]):
        model.train()
        for i in range(config["nr_of_decoders"]):
            mask_decoders[i].train()
        train_epoch_loss = 0
        for batch in tqdm(train_dataloader):
            # forward pass
            with torch.no_grad():
                image, bboxes, gt_masks, mask_values, mask_counts = batch
                gt_masks = gt_masks.to(device)
                optimizer.zero_grad()
            
            inputs = processor(image, input_boxes=bboxes, return_tensors="pt").to(device)
            outputs=custom_forward(model, mask_decoders, **inputs.to(device), multimask_output=False)
            
            #postprocessing
            masks = torch.zeros(gt_masks.shape[0], 14,1, 496, 512)
            for i in range(config["nr_of_decoders"]):
                masks_i = F.interpolate(outputs[1][i][:,:,0,:,:], (1024,1024), mode="bilinear", align_corners=False)
                masks_i = masks_i[..., : 992, : 1024]
                masks[:,i,:,:,:] = F.interpolate(masks_i, (496,512), mode="bilinear", align_corners=False) 
            
            # compute loss
            train_loss = 0
            batched_mask_count = torch.sum(mask_counts, dim=0)
            train_loss = seg_loss(masks.squeeze().to("cuda"), gt_masks)
            if config["topological"]:
                (B,C,H,W) = masks.shape
                masks = masks.view(B*C, H, W)
                gt_masks = gt_masks.view(B*C, H, W)
                pi_mask = vr(masks)
                pi_gt_mask = vr(gt_masks)
                train_loss += lamda * topo_loss([masks, pi_mask], [gt_masks, pi_gt_mask])
            # backward pass (compute gradients of parameters w.r.t. loss
            train_loss.backward()
            # optimize
            optimizer.step()
            train_epoch_loss += train_loss.item()
        train_epoch_loss = train_epoch_loss/len(train_dataloader)
        wandb.log({"train/train_loss": train_epoch_loss, "train/epoch": epoch})
        valid_epoch_loss = validate_model(model, mask_decoders, processor, valid_dataloader, seg_loss, config)
        print(f'EPOCH: {epoch}, Train Loss: {train_epoch_loss}, Valid Loss: {valid_epoch_loss}')
        config["display_samples"] != "no" and display_samples(model, mask_decoders,processor, device, train_dataset, "train", config)
        config["display_samples"] != "no" and display_samples(model, mask_decoders, processor, device, valid_dataset, "valid", config)
    torch.save(model.state_dict(), config["checkpoint"] + config["display_name"] + "_" + config["time"] +".pt")
    for i in range(config["nr_of_decoders"]):
        torch.save(mask_decoders[i].state_dict(), config["checkpoint"] + config["display_name"] + "_" + config["time"] + "_" +str(i)+".pt")
    evaluate_metrics(model, mask_decoders, valid_dataset, config, processor)
    wandb.finish()

"""
def evaluate_metrics(model, mask_decoders, dataset, config, processor):
    metric = evaluate.load("mean_iou")
    segmentations = []
    ground_truths = []
    for i in range(len(dataset)):
        image, bboxes, gt_masks, mask_values, mask_counts = dataset[i]
        gt_masks = torch.tensor(np.array(gt_masks))
        class_labels = config["mask_dict"]
        with torch.no_grad():
            inputs = processor(image, input_boxes=[[[0,0,496,512]]], return_tensors="pt")
            outputs=custom_forward(model, mask_decoders, **inputs.to("cuda"), multimask_output=False)
            #outputs = model(**inputs.to(device), multimask_output=False)
            masks = torch.zeros(1, 14, 496, 512)
            for i in range(config["nr_of_decoders"]):
                #masks = F.interpolate(outputs.pred_masks[:,:,0,:,:], (1024,1024), mode="bilinear", align_corners=False)
                masks_i = F.interpolate(outputs[1][i][:,:,0,:,:], (1024,1024), mode="bilinear", align_corners=False)
                masks_i = masks_i[..., : 992, : 1024]
                masks[:,i,:,:] = F.interpolate(masks_i, (496,512), mode="bilinear", align_corners=False)    
            masks = torch.argmax(masks, dim=1).squeeze()
            gt_masks = torch.argmax(gt_masks, dim=0)
        segmentations.append(masks)
        ground_truths.append(gt_masks)
    metric = metric.compute(
        predictions=segmentations,
        references=ground_truths,
        ignore_index=255,
        num_labels=14,
        reduce_labels=False,
    )
    print(metric)
"""
def evaluate_metrics(model, mask_decoders, dataset, config, processor):
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
    category_sample_accuracies = np.zeros(14)
    category_sample_ious = np.zeros(14)
    category_sample_f1 = np.zeros(14)
    category_sample_dice = np.zeros(14)
    category_sample_spec = np.zeros(14)
    category_sample_sens = np.zeros(14)
    category_sample_map = np.zeros(14)
    for i in range(14):
        segmentations.append([])
        ground_truths.append([])
        segmentations_probas.append([])
        indexes.append([])
    for i in tqdm(range(len(dataset))):
        image, bboxes, gt_masks, mask_values, mask_counts = dataset[i]
        gt_masks = torch.tensor(np.array(gt_masks))
        class_labels = config["mask_dict"]
        with torch.no_grad():
            inputs = processor(image, input_boxes=[[[0,0,496,512]]], return_tensors="pt")
            outputs=custom_forward(model, mask_decoders, **inputs.to("cuda"), multimask_output=False)
            #outputs = model(**inputs.to(device), multimask_output=False)
            masks = torch.zeros(1, 14, 496, 512)
            for i in range(config["nr_of_decoders"]):
                #masks = F.interpolate(outputs.pred_masks[:,:,0,:,:], (1024,1024), mode="bilinear", align_corners=False)
                masks_i = F.interpolate(outputs[1][i][:,:,0,:,:], (1024,1024), mode="bilinear", align_corners=False)
                masks_i = masks_i[..., : 992, : 1024]
                masks[:,i,:,:] = F.interpolate(masks_i, (496,512), mode="bilinear", align_corners=False)    
            masks_compact = torch.argmax(masks, dim=1).squeeze()
            gt_masks = torch.argmax(gt_masks, dim=0)
        values = torch.unique(gt_masks)
        soft_masks = torch.softmax(masks,1)
        for v in values:
            binary_mask = torch.where(masks_compact==v, 1.0, 0.0)
            binary_gt_masks = torch.where(gt_masks==v, 1.0, 0.0)
            segmentations[v].append(binary_mask)
            segmentations_probas[v].append(soft_masks[0,v,:,:])
            ground_truths[v].append(binary_gt_masks)
            indexes[v].append(i)
        
    for i in range(14):
        print(f"------------------CLASS: {config['mask_dict'][i]}----------------------")
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
            metric_output_tmp = metric_iou.compute(
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
            sample_iou.append(metric_output_tmp['per_category_iou'][1])
            sample_accuracy.append(metric_output_tmp['per_category_accuracy'][1])
            sample_spec.append(tn / (tn + fp) if (tn + fp) != 0 else 0.0)
            sample_sens.append(tp / (tp + fn) if (tp + fn) != 0 else 0.0)
            sample_f1.append(sklearn.metrics.f1_score(flat_gt, flat_seg))
            sample_dice.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0)
            sample_ap.append(sklearn.metrics.average_precision_score(flat_gt, flat_segp))
            
        category_sample_accuracies[i] = np.mean(sample_accuracy)
        category_sample_dice[i] = np.mean(sample_dice)
        category_sample_ious[i] = np.mean(sample_iou)
        category_sample_f1[i] = np.mean(sample_f1)
        category_sample_sens[i] = np.mean(sample_sens)
        category_sample_spec[i] = np.mean(sample_spec)
        category_sample_map[i] = np.mean(sample_ap)
        
        avg_start_idx = len(sample_iou) // 2 - NO_BEST_WORST_SAMPLES // 2
        avg_end_idx = len(sample_iou) // 2 + NO_BEST_WORST_SAMPLES // 2
        idx = np.array(indexes[i])
        
        print(f"GENERAL REPORT:")
        print(metric_output)
        print(f"----IoU----:")
        print(f"{category_ious[i]} \ {category_sample_ious[i]}")
        print(f"Best samples: {idx[np.argsort(sample_iou)[-NO_BEST_WORST_SAMPLES:]]}")
        print(f"Average samples: {idx[np.argsort(sample_iou)[avg_start_idx:avg_end_idx]]}")
        print(f"Worst samples: {idx[np.argsort(sample_iou)[:NO_BEST_WORST_SAMPLES]]}")
        print(f"----Accuracy----:")
        print(f"{category_accuracies[i]} \ {category_sample_accuracies[i]}")
        print(f"Best samples: {idx[np.argsort(sample_accuracy)[-NO_BEST_WORST_SAMPLES:]]}")
        print(f"Average samples: {idx[np.argsort(sample_accuracy)[avg_start_idx:avg_end_idx]]}")
        print(f"Worst samples: {idx[np.argsort(sample_accuracy)[:NO_BEST_WORST_SAMPLES]]}")
        print(f"----Specificity----:")
        print(f"{category_spec[i]} \ {category_sample_spec[i]}")
        print(f"Best samples: {idx[np.argsort(sample_spec)[-NO_BEST_WORST_SAMPLES:]]}")
        print(f"Average samples: {idx[np.argsort(sample_spec)[avg_start_idx:avg_end_idx]]}")
        print(f"Worst samples: {idx[np.argsort(sample_spec)[:NO_BEST_WORST_SAMPLES]]}")
        print(f"----Sensitivity----:")
        print(f"{category_sens[i]} \ {category_sample_sens[i]}")
        print(f"Best samples: {idx[np.argsort(sample_sens)[-NO_BEST_WORST_SAMPLES:]]}")
        print(f"Average samples: {idx[np.argsort(sample_sens)[avg_start_idx:avg_end_idx]]}")
        print(f"Worst samples: {idx[np.argsort(sample_sens)[:NO_BEST_WORST_SAMPLES]]}")
        print(f"----F1----:")
        print(f"{category_f1[i]} \ {category_sample_f1[i]}")
        print(f"Best samples: {idx[np.argsort(sample_f1)[-NO_BEST_WORST_SAMPLES:]]}")
        print(f"Average samples: {idx[np.argsort(sample_f1)[avg_start_idx:avg_end_idx]]}")
        print(f"Worst samples: {idx[np.argsort(sample_f1)[:NO_BEST_WORST_SAMPLES]]}")
        print(f"----Dice----:")
        print(f"{category_dice[i]} \ {category_sample_dice[i]}")
        print(f"Best samples: {idx[np.argsort(sample_dice)[-NO_BEST_WORST_SAMPLES:]]}")
        print(f"Average samples: {idx[np.argsort(sample_dice)[avg_start_idx:avg_end_idx]]}")
        print(f"Worst samples: {idx[np.argsort(sample_dice)[:NO_BEST_WORST_SAMPLES]]}")
        print(f"----AP----:")
        print(f"{category_map[i]} \ {category_sample_map[i]}")
        print(f"Best samples: {idx[np.argsort(sample_ap)[-NO_BEST_WORST_SAMPLES:]]}")
        print(f"Average samples: {idx[np.argsort(sample_ap)[avg_start_idx:avg_end_idx]]}")
        print(f"Worst samples: {idx[np.argsort(sample_ap)[:NO_BEST_WORST_SAMPLES]]}")
        
    print(f"----------GLOBAL----------")
    print("Category_accuracies:" + str(list(category_accuracies))+"\n"+"Category_ious:"+str(list(category_ious)))
    print(f"Category_specificity: {category_spec}")
    print(f"Category_sensitivity: {category_sens}")
    print(f"Category_dice: {category_dice}")
    print(f"Category_ap: {category_map}")
    mean_iou = np.mean(category_ious)
    mean_accuracy = np.mean(category_accuracies)
    mean_spec = np.mean(category_spec)
    mean_sens = np.mean(category_sens)
    mean_dice = np.mean(category_dice)
    mean_map = np.mean(category_map)
    print("Mean_accuracy:" + str(mean_accuracy)+"\n"+"Mean_iou:"+str(mean_iou))
    print(f"Mean specificity: {mean_spec}")
    print(f"Mean sensitivity: {mean_sens}")
    print(f"Mean dice: {mean_dice}")
    print(f"Mean mAP: {mean_map}")
    
    print(f"----------SAMPLE----------")
    print("Category_accuracies:" + str(list(category_sample_accuracies))+"\n"+"Category_ious:"+str(list(category_sample_ious)))
    print(f"Category_specificity: {category_sample_spec}")
    print(f"Category_sensitivity: {category_sample_sens}")
    print(f"Category_dice: {category_sample_dice}")
    print(f"Category_ap: {category_sample_map}")
    mean_iou = np.mean(category_sample_ious)
    mean_accuracy = np.mean(category_sample_accuracies)
    mean_spec = np.mean(category_sample_spec)
    mean_sens = np.mean(category_sample_sens)
    mean_dice = np.mean(category_sample_dice)
    mean_map = np.mean(category_sample_map)
    print("Mean_accuracy:" + str(mean_accuracy)+"\n"+"Mean_iou:"+str(mean_iou))
    print(f"Mean specificity: {mean_spec}")
    print(f"Mean sensitivity: {mean_sens}")
    print(f"Mean dice: {mean_dice}")
    print(f"Mean mAP: {mean_map}")

def prepare_model(base_model):
    processor = SamProcessor.from_pretrained(base_model)
    model = SamModel.from_pretrained(base_model)
    #make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    return processor, model

def custom_forward(model, mask_decoders,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict

    if pixel_values is None and image_embeddings is None:
        raise ValueError("Either pixel_values or image_embeddings must be provided.")

    if pixel_values is not None and image_embeddings is not None:
        raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

    if input_points is not None and len(input_points.shape) != 4:
        raise ValueError(
            "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
            " got {}.".format(input_points.shape),
        )
    if input_boxes is not None and len(input_boxes.shape) != 3:
        raise ValueError(
            "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
            " got {}.".format(input_boxes.shape),
        )
    if input_points is not None and input_boxes is not None:
        point_batch_size = input_points.shape[1]
        box_batch_size = input_boxes.shape[1]
        if point_batch_size != box_batch_size:
            raise ValueError(
                "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                    point_batch_size, box_batch_size
                )
            )

    image_positional_embeddings = model.get_image_wide_positional_embeddings()
    # repeat with batch size
    batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
    image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

    vision_attentions = None
    vision_hidden_states = None

    if pixel_values is not None:
        vision_outputs = model.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeddings = vision_outputs[0]

        if output_hidden_states:
            vision_hidden_states = vision_outputs[1]
        if output_attentions:
            vision_attentions = vision_outputs[-1]

    if input_points is not None and input_labels is None:
        input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

    if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
        raise ValueError(
            "The batch size of the image embeddings and the input points must be the same. ",
            "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),
            " if you want to pass multiple points for the same image, make sure that you passed ",
            " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
            " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
        )

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        input_masks=input_masks,
    )
    low_res_masks_list, iou_predictions_list, mask_decoder_attentions_list = [],[],[]
    for mask_decoder in mask_decoders:
        low_res_masks, iou_predictions, mask_decoder_attentions = mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        low_res_masks_list.append(low_res_masks)
        iou_predictions_list.append(iou_predictions)
        mask_decoder_attentions_list.append(mask_decoder_attentions)

    if not return_dict:
        output = (iou_predictions_list, low_res_masks_list)
        if output_hidden_states:
            output = output + (vision_hidden_states,)

        if output_attentions:
            output = output + (vision_attentions, mask_decoder_attentions_list)
        return output

    return (
        iou_scores:=iou_predictions_list,
        pred_masks:=low_res_masks_list,
        vision_hidden_states:=vision_hidden_states,
        vision_attentions:=vision_attentions,
        mask_decoder_attentions:=mask_decoder_attentions_list,
    )

def prepare_data(processor, dataset, split, config):
    dataset = datasets.load_from_disk(dataset)[split]
    config["data_transforms"] and dataset.set_transform(data_transforms(operations=config["data_transforms"]))
    dataset = SAMDataset(dataset=dataset)
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

def display_samples(model, mask_decoders, processor, device, dataset, split, config):
    model.eval()
    idx = select_display_indices(dataset, config)
    img = []
    for i in idx:
        image, bboxes, gt_masks, mask_values, mask_counts = dataset[i]
        gt_masks = torch.tensor(np.array(gt_masks))
        class_labels = config["mask_dict"]
        with torch.no_grad():
            inputs = processor(image, input_boxes=[[[0,0,496,512]]], return_tensors="pt")
            outputs=custom_forward(model, mask_decoders, **inputs.to(device), multimask_output=False)
            #outputs = model(**inputs.to(device), multimask_output=False)
            masks = torch.zeros(1, 14, 496, 512)
            for i in range(config["nr_of_decoders"]):
                #masks = F.interpolate(outputs.pred_masks[:,:,0,:,:], (1024,1024), mode="bilinear", align_corners=False)
                masks_i = F.interpolate(outputs[1][i][:,:,0,:,:], (1024,1024), mode="bilinear", align_corners=False)
                masks_i = masks_i[..., : 992, : 1024]
                masks[:,i,:,:] = F.interpolate(masks_i, (496,512), mode="bilinear", align_corners=False)    
            masks = torch.argmax(masks, dim=1).squeeze()
            gt_masks = torch.argmax(gt_masks, dim=0)
            image_masks = {
                "pred": {"mask_data": masks.cpu().numpy(), "class_labels": class_labels},
                # class_labels could be modified to show the difference better
                "gt": {"mask_data": gt_masks.numpy(), "class_labels": class_labels}
            }
            img.append(wandb.Image(
                image,
                masks=image_masks,
            ))
        
    wandb.log({split+"_samples" :img})
    model.train()

def select_display_indices(dataset, config):
    return [0,1,3]

def validate_model(model, mask_decoders, processor, valid_dl, seg_loss, config, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    for i in range(config["nr_of_decoders"]):
        mask_decoders[i].eval()
    epoch_loss = 0.
    with torch.inference_mode():
        for batch in tqdm(valid_dl):
            # forward pass
            image, bboxes, gt_masks, mask_values, mask_counts = batch
            gt_masks = gt_masks.to(device)
            inputs = processor(image, input_boxes=bboxes, return_tensors="pt").to(device)
            outputs=custom_forward(model, mask_decoders, **inputs.to(device), multimask_output=False)
            #postprocessing
            masks = torch.zeros(gt_masks.shape[0], 14,1, 496, 512)
            for i in range(config["nr_of_decoders"]):
                #masks = F.interpolate(outputs.pred_masks[:,:,0,:,:], (1024,1024), mode="bilinear", align_corners=False)
                masks_i = F.interpolate(outputs[1][i][:,:,0,:,:], (1024,1024), mode="bilinear", align_corners=False)
                masks_i = masks_i[..., : 992, : 1024]
                masks[:,i,:,:,:] = F.interpolate(masks_i, (496,512), mode="bilinear", align_corners=False) 

            # compute loss
            train_loss = 0
            
            train_loss = seg_loss(masks.squeeze().to("cuda"), gt_masks)
            epoch_loss += train_loss
        epoch_loss = epoch_loss/len(valid_dl)
        wandb.log({"val/valid_loss": epoch_loss})
    return epoch_loss

class SAMDataset(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_bboxes_and_gt_masks(self, ground_truth_mask):
        # get bounding boxes from mask
        bboxes, gt_masks = [],[]
        mask_values, mask_counts = np.unique(ground_truth_mask, return_counts=True)
        #Comment for background prediction
        #mask_values, mask_counts = mask_values[1:], mask_counts[1:]
        for v in range(14): #mask_values: 
            bboxes = [[0,0,496,512]]
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
