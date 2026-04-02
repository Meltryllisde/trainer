import torch
from PIL import Image
import numpy as np
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results
model = build_sam3_image_model()
processor = Sam3Processor(model=model, confidence_threshold=0.8)
def sam(image_path, obj):
    image = Image.open(image_path).convert("RGB")
    inference_state  = processor.set_image(image)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=obj)
    # boxes = inference_state['boxes']
    # TODO: 获取掩膜 img_width*img_height*1
    masks = inference_state['masks'].cpu().numpy()
    return masks
def IoU(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union
def process_images(img_1, img_2, obj):
    mask1 = sam(img_1, obj)
    mask2 = sam(img_2, obj)
    iou_score = IoU(mask1, mask2)
    return iou_score
##########      python -m pip install --force-reinstall "setuptools<82"      ##########
    # - conda create -n flymyai python=3.12
    # - source activate flymyai
    # - cd /storage/v-jinpewang/lab_folder/zc_workspace/code/sam3
    # - pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    # - pip install -e .
    # - pip install -e ".[notebooks]"
    # - python -m pip install --force-reinstall "setuptools<82"
    # - pip install pandas