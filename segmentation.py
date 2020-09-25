import multiprocessing as mp
import os
import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from modules.predictor import VisualizationDemo

# constants
WINDOW_NAME = "IOU Segmentation"
FILE_NAME = 1

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

def get_only_instance_image(input_file, masks, height, width, output_file=None):
    '''
    input_file  : string, 파일 이름
    masks       : 2차원 list, width / height 크기, True 면 그 자리에 객체가 있는것, 아니면 없음.
    width       : int, 너비
    height      : int, 높이
    output_file : string, output_file 이름
    '''
    if output_file == None:
        output_file = input_file.split(".")[0] + "_masked" + input_file.split(".")[1]
    original = cv2.imread(input_file)
    masked_image = np.zeros([height, width ,3], dtype=np.uint8)

    for h in range(0, height):
        for w in range(0, width):
                for c in range(0, 3):
                    masked_image[h][w][c] = (original[h][w][c] if masks[h][w] else 0)
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, masked_image)
    if cv2.waitKey(0) == 27:
        visualized_output.save(output_file)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args_list = [
        "modules/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 
        "Image/chair1.jpg", 
        0.6, 
        ["MODEL.WEIGHTS", "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"],
        "chair1_masked.jpg"
    ]
    cfg = setup_cfg(args_list)

    demo = VisualizationDemo(cfg)

    # use PIL, to be consistent with evaluation
    img = read_image(args_list[1], format="BGR")
    predictions, visualized_output = demo.run_on_image(img)

    # 계산한 prediction에서 mask를 가져옴.
    masks = predictions['instances'].get_fields()["pred_masks"]
    masks = masks.tolist()  # masks 는 TF value의 tensor 값들
    (height, width) = predictions['instances'].image_size
    instance_number = len(predictions['instances'])

    for i in range(0, instance_number):
        get_only_instance_image(args_list[FILE_NAME], masks[i], height, width)
    