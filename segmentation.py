import multiprocessing as mp
import matrix_processing
import image_processing
from matrix_processing import contours_to_coord
import utility
import sys

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

from modules.predictor import VisualizationDemo
from utility import divided_class_into_image

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

def merge_around(divided_class, class_number, class_total, class_border, class_count, merge_indx, width, height):
	large_class_number = matrix_processing.get_around_largest_area(divided_class, width, height, class_border[merge_indx], class_number[merge_indx])
	if large_class_number == 0:
		# Set into 0 Class
		matrix_processing.set_area(divided_class, class_total[merge_indx], 0)
		del class_number[merge_indx]
		del class_total[merge_indx]
		del class_border[merge_indx]
		del class_count[merge_indx]
		return class_number, class_total, class_border, class_count, len(class_total)
	else:
		# Not in 0 class.
		merging_list = [class_number.index(large_class_number), merge_indx]
		return merge_divided_group(divided_class, class_number, class_total, class_border, class_count, merging_list, width, height)

def merge_small_size(divided_class, class_number, class_total, class_border, class_count, width, height, min_value=200):
	'''
	일정 크기 이하의 영역들을 하나로 합치기 위한 함수.
	divided_class : 2D List for image class.
	class_number : 각 Class들의 numbering set.
	Class_total ~ Class Count : 다른것들과 동일
	min_value : 이 숫자보다 작은 크기의 영역들은 근처의 다른 곳에 합쳐지게 된다.
	'''
	indx = get_small_class(class_count, min_value)
	while indx != -1:
		class_number, class_total, class_border, class_count, _ = \
			merge_around(divided_class, class_number, class_total, class_border, class_count, indx, width, height)
		indx = get_small_class(class_count, min_value)
	
	return class_number, class_total, class_border, class_count, len(class_number) 

def get_small_class(class_count, n):
	# Return first list index which is smaller then n.
	# If there is no class smaller then n, return -1.
	for i in range(len(class_count)):
		if class_count[i] < n :
			return i
	return -1

def merge_divided_group(divided_class, class_numbers, class_total, class_border, class_count, merge_group_index, width, height):
	'''
	merge_group_index에 적힌 class들을 하나의 class로 결합한다.
	divided_class : 2D List // 각 Class 숫자들이 적혀있음. 
		- 함수 진행 뒤 변경됨.
	class_numbers : 각 index에서의 class 숫자들. 처음에는 1 ~ N 이지만, 합쳐지면서 바뀔 수 있다.
	dividec_class ~ class_count : 다른것과 동일.
	merge_group_index : divided_class에서 서로 하나로 합칠 그룹의 index 들. 대표가 될 가장 큰 Group이 [0]이다.
	'''
	class_num = len(class_total)
	ret_class_numbers = []
	ret_class_total = []
	ret_class_border = []
	ret_class_count = []
	merge_base_index = merge_group_index[0]
	
	for i in range(class_num):
		if not(i in merge_group_index and i != merge_base_index):
			ret_class_numbers.append(class_numbers[i])
			ret_class_total.append(class_total[i])
			ret_class_border.append(class_border[i])
			ret_class_count.append(class_count[i])

	for i in range(class_num):
		if i in merge_group_index and i != merge_base_index:
			ret_class_total[merge_base_index] += class_total[i]
			ret_class_count[merge_base_index] += class_count[i]

	matrix_processing.set_area(divided_class, ret_class_total[merge_base_index], ret_class_numbers[merge_base_index])
	if len(merge_group_index) == 2:
		ret_class_border[merge_base_index] += matrix_processing.check_border(divided_class, class_border[merge_group_index[1]], width, height)
	else:
		for i in range(class_num):
			if i in merge_group_index and i != merge_base_index:
				ret_class_border[merge_base_index] += matrix_processing.check_border(divided_class, class_border[i], width, height)

	return ret_class_numbers, ret_class_total, ret_class_border, ret_class_count, len(ret_class_total), 

def merge_same_color(divided_class, class_numbers, class_total, class_border, class_count, largest_mask, width, height, sim_score=180):
	'''
	유사도 sim_score 이내의 같은 plate들을 모아서 return.
	'''
	same_flag = True
	class_length = len(class_numbers)
	ret_class_numbers = class_numbers
	ret_class_total = class_total
	ret_class_border = class_border
	ret_class_count = class_count

	while same_flag:
		same_flag = False
		# Make average class colors.
		class_color = image_processing.get_class_color(largest_mask, ret_class_total, ret_class_count)
		color_map = utility.get_color_distance_map(class_color, class_length)
		
		merge_group_index = []
		for i in range(class_length):
			for j in range(i + 1, class_length):
				if color_map[i][j] < sim_score:
					if i not in merge_group_index:
						merge_group_index.append(i)
						same_flag = True
					merge_group_index.append(j)
			if same_flag:
				break
		if len(merge_group_index) == 0:
			break
		else:
			for i in range(1, len(merge_group_index)):
				del class_color[merge_group_index[i] - i + 1]
		ret_class_numbers, ret_class_total, ret_class_border, ret_class_count, class_length = \
			merge_divided_group(divided_class, ret_class_numbers, ret_class_total, ret_class_border, ret_class_count, merge_group_index, width, height)

	# Make average class colors with last reamins.
	class_color = image_processing.get_class_color(largest_mask, ret_class_total, ret_class_count)
	return ret_class_numbers, ret_class_total, ret_class_border, ret_class_count, class_length, class_color

def delete_unavailable_color(divided_class, class_numbers, class_total, class_border, class_count, largest_mask, min_add_up=40):
	class_length = len(class_numbers)
	ret_class_numbers = []
	ret_class_total = []
	ret_class_border = []
	ret_class_count = []
	class_color = image_processing.get_class_color(largest_mask, class_total, class_count)

	zero_list = []
	for i in range(class_length):
		if class_color[i][0] + class_color[i][1] + class_color[i][2] < min_add_up:
			zero_list.append(i)
			continue
		ret_class_numbers.append(class_numbers[i])
		ret_class_total.append(class_total[i])
		ret_class_border.append(class_border[i])
		ret_class_count.append(class_count[i])

	for z in range(len(zero_list)):	
		matrix_processing.set_area(divided_class, class_total[zero_list[z]], 0)

	return ret_class_numbers, ret_class_total, ret_class_border, ret_class_count, len(ret_class_total)

def get_mask(fileName, cfg):
	'''
	fileName 내의 가장 큰 sgemenation 된 부분의 image 를 return.
	'''
	demo = VisualizationDemo(cfg)
	# use PIL, to be consistent with evaluation
	img = read_image(fileName, format="BGR")
	predictions, visualized_output = demo.run_on_image(img)

	# 계산한 prediction에서 mask를 가져옴.
	masks = predictions['instances'].get_fields()["pred_masks"]
	masks = masks.tolist()  # masks 는 TF value의 tensor 값들
	(height, width) = predictions['instances'].image_size
	instance_number = len(predictions['instances'])
	# Mask 칠한 이미지 중에서 가장 큰것만 가지고 진행함.
	largest_mask = []
	largest_mask_coord = []
	largest_mask_map = []
	largest_mask_number = -1

	for i in range(0, instance_number):
		mask = matrix_processing.get_largest_part(masks[i], width, height)
		masked_image, mask_num = image_processing.get_only_instance_image(fileName, mask, width, height)
		if mask_num > largest_mask_number:
			largest_mask = masked_image
			largest_mask_number = mask_num
			largest_mask_map = mask
		
	return largest_mask, largest_mask_number, largest_mask_map, (width, height) 

def get_segmented_image(inputFile):
	mp.set_start_method("spawn", force=True)
	args_list = [
		"modules/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 
		inputFile, 
		0.6, 
		["MODEL.WEIGHTS", "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"],
		"no_name"
	]
	cfg = setup_cfg(args_list)

	return get_mask(args_list[1], cfg)

def get_divided_class(inputFile, clipLimit=16.0, tileGridSize=(16, 16), start=60, diff=150, delete_line_n=20, border_n=6, border_k=2, merge_min_value=180, sim_score=30, merge_mode_color=False):
	'''
	predict masking image and get divided_class.
	'''
	largest_mask, largest_index, mask_map, (width, height) = get_segmented_image(inputFile)
	# 만약 Detectron이 감지하지 못한경우
	if largest_index == -1:
		largest_mask = utility.read_image(inputFile)
		(height, width, _) = largest_mask.shape
		mask_map = [[True for _ in range(width)] for _ in range(height)]

	# 잘린 이미지를 통해 외곽선을 얻어서 진행.
	contours, _ = image_processing.get_contours(largest_mask, clipLimit=clipLimit, tileGridSize=tileGridSize, start=start, diff=diff)
	coords = matrix_processing.contours_to_coord(contours)

	# 작은 Line은 삭제.
	coords = matrix_processing.delete_line_threshold(coords, line_n=delete_line_n)
	cycle_list = []
	noncycle_list = []

	for c in coords:
		cycled, noncycled = matrix_processing.divide_cycle(c)
		if len(cycled) != 0:
			cycle_list += cycled
		if len(noncycled) != 0:
			noncycle_list += noncycled
	
	# 잘린 외곽선들을 True-False List로 바꿔서 각각 가장 가까운 곳에 연결.
	tf_map = utility.make_tf_map(noncycle_list, width, height)
	for nc in noncycle_list:
		# 가장자리가 될 포인트를 잡는다.
		border_point = matrix_processing.find_border_k_tf_map(tf_map, nc, width, height, n=border_n, k=border_k, hard_check=False)
		for b in border_point:
			# 가장자리에서 가장 가까운 외곽선으로 연결한다.
			matrix_processing.connect_nearest_point(tf_map, b, width, height, nc)

	# 나누어진 면적들을 DFS로 각각 가져온다. tf_map 은 true false 에서 숫자가 써있는 Map 이 된다.
	divided_class, class_total, class_border, class_count, class_length = matrix_processing.get_image_into_divided_plate(tf_map, width, height)
	# 또한 나눈 선들도 각 면적에 포함시켜 나눈다.
	matrix_processing.contours_to_divided_class(tf_map, divided_class, class_total, class_border, class_count, width, height)

	class_number = list(range(1, class_length + 1))
	# 작은 Size는 주변에 다시 넣는다. 이 때, 작은 값부터 천천히 올라가는 방법을 사용한다.
	for min_value in range(30, merge_min_value, 30):
		class_number, class_total, class_border, class_count, class_length = \
		merge_small_size(divided_class, class_number, class_total, class_border, class_count, width, height, min_value=min_value)

	# 원래 Segmentation 돤것에서 나가는 것은 삭제한다.
	class_number, class_total, class_border, class_count, class_length = \
	out_mask_delete(mask_map, class_number, class_total, class_border, class_count, class_length, out_pixel_threshold=0)

	class_color = image_processing.get_class_color(largest_mask, class_total, class_count)
	if merge_mode_color:
		# 비슷한 색끼리도 모아준다.
		class_number, class_total, class_border, class_count, class_length, class_color = \
		merge_same_color(divided_class, class_number, class_total, class_border, class_count, largest_mask, width, height, sim_score=sim_score)
	
	return divided_class, class_number, class_total, class_border, class_count, class_length, class_color, largest_mask, width, height

def out_mask_delete(mask_map, class_number, class_total, class_border, class_count, class_length, out_pixel_threshold=0):
	ret_class_number = []
	ret_class_total = []
	ret_class_border = []
	ret_class_count = []
	
	for i in range(class_length):
		pass_flag = True
		out_pixel_num = 0

		for coord in class_total[i]:
			if not mask_map[coord[1]][coord[0]]:
				if out_pixel_num < out_pixel_threshold:
					out_pixel_num += 1
					continue
				pass_flag = False
				break

		if pass_flag:
			ret_class_number.append(class_number[i])
			ret_class_total.append(class_total[i])
			ret_class_border.append(class_border[i])
			ret_class_count.append(class_count[i])
			
	return ret_class_number, ret_class_total, ret_class_border, ret_class_count, len(ret_class_total)

def divided_into_classed_color_based(image, divided_class, class_total, class_number_max, width, height, div_threshold=60):
	'''
	Image의 색을 기준으로, 현재 주어진 Class Total의 Pixel들을 Clustering 한다. 나눠진 class_number는 class_number_max + 1 부터 차례로 부여된다.
	만약 색상이 크게 차이나지 않는다면 나누지 않음.
	image : 색을 참조할 Image.
	class_total : 나눌 Coord List.
	class_number_max : 부여할 Class Number 기준.
	div_threshold : 나눌 Color Space Diff 기준.
	'''
	# Get each pixel`s colors.
	tf_map = utility.make_tf_map([class_total], width, height, border=False)
	total_length = len(class_total)
	visited = [False for _ in range(total_length)]
	visited[0] = True

	class_index_divided = [[0]]
	class_total_divided = [[class_total[0]]]
	class_index = [-1 for _ in range(total_length)]
	class_index[0] = 0
	now_class_index_max = 1
	tf_map[class_total[0][1]][class_total[0][0]] = False

	for i in range(total_length - 1):
		for direction in range(4):
			if utility.can_go(class_total[i][0], class_total[i][1], width, height, direction=direction):
				x = class_total[i][0] + utility.dir_x[direction]
				y = class_total[i][1] + utility.dir_y[direction]
				if not tf_map[y][x]:
					continue
				tf_map[y][x] = False
				j = class_total.index((x, y))
				if utility.get_cielab_distance(image[y][x], image[class_total[i][1]][class_total[i][0]]) < div_threshold:
					class_index_divided[class_index[i]].append(j)
					class_total_divided[class_index[i]].append(class_total[j])
					class_index[j] = class_index[i]
				else:
					class_index_divided.append([j])
					class_total_divided.append([class_total[j]])
					class_index[j] = now_class_index_max
					now_class_index_max += 1
	
	class_number = [class_number_max + i for i in range(1, now_class_index_max + 1)]
	for i in range(len(class_number)):
		matrix_processing.set_area(divided_class, class_total_divided[i], class_number[i])
	class_border = [matrix_processing.check_border(divided_class, class_total_divided[i], width, height) for i in range(len(class_number))]
	class_count = [len(class_total_divided[i]) for i in range(len(class_number))]
	return class_number, class_total_divided, class_border, class_count
