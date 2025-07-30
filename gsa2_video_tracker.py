import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images


GSA2_PATH = os.path.dirname(__file__)


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_frame_names(video_dir):
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    return frame_names


class GSA2VideoTracker:
    def __init__(
            self,
            device = "cpu",
            sam2_ckpt_path = "{GSA2_PATH}/checkpoints/sam2.1_hiera_large.pt",
            sam2_cfg_path = "configs/sam2.1/sam2.1_hiera_l.yaml",
            gd_model_id = "IDEA-Research/grounding-dino-base",
        ):
        self.device = device
        sam2_ckpt_path = sam2_ckpt_path.format(GSA2_PATH=GSA2_PATH)
        self._init_sam2(sam2_ckpt_path, sam2_cfg_path)
        self._init_gd(gd_model_id)


    def _init_sam2(self, ckpt_path, cfg_path):
        self.video_predictor = build_sam2_video_predictor(cfg_path, ckpt_path)
        sam2_image_model = build_sam2(cfg_path, ckpt_path)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)


    def _init_gd(self, model_id):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)


    def process_prompt(
            self,
            prompt_type,
            objects,
            inference_state,
            ann_frame_idx,
            masks,
            input_boxes,
        ):
        assert prompt_type in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

        # If you are using point prompts, we uniformly sample positive points based on the mask
        if prompt_type == "point":
            # sample the positive points from mask for each objects
            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

            for object_id, (label, points) in enumerate(zip(objects, all_sample_points), start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )
        # Using box prompt
        elif prompt_type == "box":
            for object_id, (label, box) in enumerate(zip(objects, input_boxes), start=1):
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )
        # Using mask prompt is a more straightforward way
        elif prompt_type == "mask":
            for object_id, (label, mask) in enumerate(zip(objects, masks), start=1):
                labels = np.ones((1), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    mask=mask
                )

        return out_obj_ids, out_mask_logits


    def predict(
            self,
            video_dir,
            text,
            ann_frame_idx=0,
            box_threshold=0.25,
            text_threshold=0.3,
            prompt_type="box",
            save_dir = None,
        ):
        # init video predictor state
        inference_state = self.video_predictor.init_state(video_path=video_dir)

        # scan all the JPEG frame names in this directory
        frame_names = get_frame_names(video_dir)

        # Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
        # prompt grounding dino to get the box coordinates on specific frame
        img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
        image = Image.open(img_path)

        # run Grounding DINO on the image
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )

        # prompt SAM image predictor to get the mask for the object
        self.image_predictor.set_image(np.array(image.convert("RGB")))

        # process the detection results
        input_boxes = results[0]["boxes"].cpu().numpy()
        objects = results[0]["labels"]

        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # convert the mask shape to (n, H, W)
        if masks.ndim == 3:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        # Step 3: Register each object's positive points to video predictor with seperate add_new_points call
        out_obj_ids, out_mask_logits = self.process_prompt(
            prompt_type,
            objects,
            inference_state,
            ann_frame_idx,
            masks,
            input_boxes,
        )

        # Step 4: Propagate the video predictor to get the segmentation results for each frame
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Step 5: Visualize the segment results across the video and save them
        if save_dir is not None:
            self.save_result(
                save_dir,
                video_dir,
                frame_names,
                objects,
                video_segments,
            )

        return video_segments


    def save_result(
            self,
            save_dir,
            video_dir,
            frame_names,
            objects,
            video_segments,
        ):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP), smart_position=True)
        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))

        id_to_objects = {i: obj for i, obj in enumerate(objects, start=1)}
        for frame_idx, segments in video_segments.items():
            img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))
            
            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)
            
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                mask=masks, # (n, h, w)
                class_id=np.array(object_ids, dtype=np.int32),
            )
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[id_to_objects[i] for i in object_ids])
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

        # Step 6: Convert the annotated frames to video
        output_video_path = os.path.join(save_dir, "video.mp4")
        create_video_from_images(save_dir, output_video_path)
