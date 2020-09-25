# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args[0])
    cfg.merge_from_list(args[3])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args[2]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args[2]
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args[2]
    cfg.freeze()
    return cfg

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args_list = [
        "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 
        "chair1.jpg", 
        0.6, 
        ["MODEL.WEIGHTS", "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"],
        "chair1_masked.jpg"
    ]
    cfg = setup_cfg(args_list)

    demo = VisualizationDemo(cfg)

    # use PIL, to be consistent with evaluation
    img = read_image(args_list[1], format="BGR")
    predictions, visualized_output = demo.run_on_image(img)

    masks = predictions['instances'].get_fields()["pred_masks"]
    
    print(len(masks))
    for mask in masks:
        for i in range(0, len(mask), 10):
            for j in range(0, len(mask[0]), 10):
                print(1 if mask[i][j] else 0, end=" ")
            print()
    

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
    if cv2.waitKey(0) == 27:
        visualized_output.save(args_list[4])
        print("")
        
