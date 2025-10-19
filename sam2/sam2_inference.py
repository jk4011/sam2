
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import load_video_simple, load_video_frames_from_file_list


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

frame_names = None

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))



class SAM2:
    def __init__(self, image_path_list, model_type="l"):
        self.image_path_list = image_path_list
        self.images, self.video_height, self.video_width = \
            load_video_frames_from_file_list(image_path_list)

        self.predictor = self.init_predictor(model_type)


    def init_predictor(self, model_type="l"):
        assert model_type in ["t", "s", "b+", "l"]
        model_type_full = {
            "t": "tiny",
            "s": "small",
            "b+": "base_plus",
            "l": "large",
        }[model_type]
        sam2_checkpoint = f"./checkpoints/sam2.1_hiera_{model_type_full}.pt"
        model_cfg = f"configs/sam2.1/sam2.1_hiera_{model_type}.yaml"
        predictor: SAM2VideoPredictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

        return predictor


    def inference_img(self, points, labels=None, visualize_masks=True, mask_threshold=-2):

        if points.max() < 1:
            points = points * torch.tensor([self.video_width, self.video_height])
            points = points.round()

        if labels is None:
            labels = np.array([1] * len(points), np.int32)
        
        inference_state = self.predictor.init_state_from_tensors(self.images, self.video_height, self.video_width)
        self.predictor.reset_state(inference_state)

        video_segments = {}  # video_segments contains the per-frame segmentation results
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )
        plt.close("all")
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {0}")
        plt.imshow(Image.open(self.image_path_list[0]))
        
        show_points(points, labels, plt.gca())
        show_mask((out_mask_logits[0] > mask_threshold).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])


    def inference_video(self, points, labels=None, visualize_masks=False, vis_frame_stride=10, mask_threshold=-2):

        if points.max() < 1:
            points = points * torch.tensor([self.video_width, self.video_height])
            points = points.round()
        
        if labels is None:
            labels = np.array([1] * len(points), np.int32)
        
        inference_state = self.predictor.init_state_from_tensors(self.images, self.video_height, self.video_width)
        self.predictor.reset_state(inference_state)

        video_segments = []  # video_segments contains the per-frame segmentation results
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments.append(out_mask_logits[0] > mask_threshold)
        
        video_segments = torch.cat(video_segments)
        
        if visualize_masks:
            plt.close("all")
            for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
                plt.figure(figsize=(6, 4))
                plt.title(f"frame {out_frame_idx}")
                plt.imshow(Image.open(self.image_path_list[out_frame_idx]))
                out_mask = video_segments[out_frame_idx].cpu().numpy()
                show_mask(out_mask, plt.gca())

        return video_segments

