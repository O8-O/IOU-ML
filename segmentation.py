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
dir_x = [0, 0, 1, -1]
dir_y = [1, -1, 0, 0]

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

def get_largest_part(mask, height, width, attach_ratio=0.15):
    '''
        TODO : attach_ration 이유 정하기
        mask    : list_2D, True False 배열의 mask
        mask의 가장 큰 True 부분을 찾고, 그 부분만 True이고, 나머지는 False로 Output 한다.
    '''
    divided_class, class_boundary, class_count, class_length = get_divied_class(mask, height, width)

    # 가장 큰것 하나 고르기
    max_indx = [1]
    for c in range(1, class_length):
        if class_count[c] > class_count[max_indx[0] - 1]:
            max_indx = [c + 1]

    # 비율별로 큰것과 비슷한 것은 다 담기
    for c in range(0, class_length):
        if class_count[c] > class_count[max_indx[0] - 1] * (1 - attach_ratio) and class_count[c] < class_count[max_indx[0] - 1] * attach_ratio:
            max_indx.append(c + 1)

    # 근처에 있는 n 픽셀 거리 안에있는것도 모으기
    for mi in max_indx:
        add_n_pixel_mask(divided_class, mi, class_boundary[mi-1], height, width)
    
    # 비슷한 것들을 모아서 하나의 Mask로 만들기.
    return set_selected_class(divided_class, max_indx, height, width)

def set_selected_class(divided_class, selected_class, height, width):
    '''
    Set True at selected class number. If else, set False.
    '''
    largest_part_mask = [[False for _ in range(0, width)] for _ in range(0, height)]
    for h in range(0, height):
        for w in range(0, width):
            if divided_class[h][w] in selected_class:
                largest_part_mask[h][w] = True

    return largest_part_mask

def set_mask_class(mask, visited, divided_class, class_length, start_index, img_size):
    '''
    mask[h][w] 에서 시작해서 연결된 모든 곳의 좌표에 divided_class list에다가 class_length 값을 대입해 놓는다.
    그 class 의 갯수를 return
    '''
    count = 1
    que = [(start_index[0], start_index[1])]
    boundary_coordinate = []

    # BFS로 Mask 처리하기.
    while que:
        now = que[0]
        del que[0]
        # 방문한 곳은 방문하지 않음.
        if visited[now[0]][now[1]]:
            continue
        
        # Class Dividing 처리.
        visited[now[0]][now[1]] = True
        if mask[now[0]][now[1]]:
            divided_class[now[0]][now[1]] = class_length
            count += 1
        
        # 경계를 체크하기 위한 Flag
        zero_boundary = False
        for direction in range(0, 4):
            if can_go(now[0], now[1], img_size[0], img_size[1], direction=direction):
                if mask[now[0] + dir_x[direction]][now[1] + dir_y[direction]]:
                    que.append((now[0] + dir_x[direction], now[1] + dir_y[direction]))
                else:
                    # 근처에 0 Class ( 아무것도 없는 공간 == mask[x][y] 가 Flase ) 가 있다면, 경계선이다.
                    zero_boundary = True
        if zero_boundary:
            boundary_coordinate.append((now[0], now[1]))
    return count, boundary_coordinate

def can_go(x, y, height, width, direction=None, x_diff=False, y_diff=False):
    '''
    주어진 범위 밖으로 나가는지 체크
    x , y : 시작 좌표
    height, width : 세로와 가로 길이
    direction : 방향 index of [동, 서, 남, 북]
    x_diff, y_diff : 만약 특정 길이만큼 이동시, 범위 밖인지 체크하고 싶을 때.
    '''
    if direction == None:        
        x_check = x + x_diff > -1 and x + x_diff < height
        y_check = y + y_diff > -1 and y + y_diff < width
    else:
        x_check = x + dir_x[direction] > -1 and x + dir_x[direction] < height
        y_check = y + dir_y[direction] > -1 and y + dir_y[direction] < width
    return x_check and y_check

def get_divied_class(mask, height, width):
    '''
        mask    : list_2D, True False 배열의 mask
        return
            divided_class : 0 ~ N list_2D. 각각은 아무것도 없으면 0, 아니면 각 class number.
            class_count : 각 class들의 숫자.
            class_length : class의 갯수
    '''
    # Initializing.
    divided_class = [[0 for _ in range(0, width)] for _ in range(0, height)]
    visited = [[False for _ in range(0, width)] for _ in range(0, height)]
    class_boundary = []
    class_count = []
    class_length = 0

    for h in range(0, height):
        for w in range(0, width):
            if visited[h][w]:
                continue
            if mask[h][w]:
                # BFS로 True로 되어있는 부분을 탐색.
                class_length += 1
                count, boundary_coordinate = set_mask_class(mask, visited, divided_class, class_length, (h, w), (height, width))
                class_count.append(count)
                class_boundary.append(boundary_coordinate)
    
    return divided_class, class_boundary, class_count, class_length

def add_n_pixel_mask(divided_class, selected_class, boundary, height, width, n=10):
    '''
    경계를 돌면서 n 픽셀 거리 안에 있는 것들을 현재 Class에 포함시키는 함수.
    '''
    for b in boundary:
        select_outside_pixel(divided_class, selected_class, height, width, b[0], b[1], n)

def select_outside_pixel(divided_class, selected_class, height, width, x, y, n):
    # 주어진 경계에 대해서 만약 거리 안에 있다면 그 class로 변환.
    for x_diff in range(-1*n, n+1, 1):
        for y_diff in range(-1*n, n+1, 1):
            if can_go(x, y, height, width, x_diff=x_diff, y_diff=y_diff):
                if get_pixel_distance(x, y, x + x_diff, y + y_diff) <= n:
                    divided_class[x + x_diff][y + y_diff] = selected_class

def get_pixel_distance(now_x, now_y, dest_x, dest_y):
    return abs(now_x - dest_x) + abs(now_y - dest_y)

def print_list_sparse(li, height, width, density=7):
    '''
        li : printing list.
        height, width : List Size
        density : 얼마나 띄엄띄엄 list를 출력 할 것인지.
    '''
    for h in range(0, height, density):
        for w in range(0, width, density):
            print(li[h][w], end=" ")
        print()

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
        mask = get_largest_part(masks[i], height, width)
        get_only_instance_image(args_list[FILE_NAME], mask, height, width)
    
