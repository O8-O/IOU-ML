import multiprocessing as mp
import matrix_processing
import image_processing
import utility

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

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
			ret_class_border[merge_base_index] += class_border[i]
			ret_class_count[merge_base_index] += class_count[i]

	matrix_processing.set_area(divided_class, ret_class_total[merge_base_index], ret_class_numbers[merge_base_index])
	ret_class_border[merge_base_index] = matrix_processing.check_border(divided_class, ret_class_border[merge_base_index], width, height)

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
	largest_mask_number = -1

	for i in range(0, instance_number):
		mask = matrix_processing.get_largest_part(masks[i], width, height)
		masked_image, mask_num = image_processing.get_only_instance_image(fileName, mask, width, height)
		if mask_num > largest_mask_number:
			largest_mask = masked_image
			largest_mask_number = mask_num
	return largest_mask, largest_mask_number, (width, height) 

def get_divided_class(inputFile, outputFile):
	'''
	predict masking image and get divided_class.
	'''
	mp.set_start_method("spawn", force=True)
	args_list = [
		"modules/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 
		inputFile, 
		0.6, 
		["MODEL.WEIGHTS", "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"],
		outputFile
	]
	cfg = setup_cfg(args_list)

	largest_mask, _, (width, height) = get_mask(args_list[1], cfg)

	# 잘린 이미지를 통해 외곽선을 얻어서 진행.
	contours, heirarchy = image_processing.get_contours(largest_mask)
	coords = matrix_processing.contours_to_coord(contours)
	coords = matrix_processing.delete_line_threshold(coords, line_n=40)
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
		border_point = matrix_processing.find_border_k_tf_map(tf_map, nc, width, height, n=6, k=2, hard_check=False)
		for b in border_point:
			# 가장자리에서 가장 가까운 외곽선으로 연결한다.
			matrix_processing.connect_nearest_point(tf_map, b, width, height, nc)
	
	# TF Image searching.
	coord_image = utility.coord_to_image(coords, width, height)
	tf_image = utility.tf_map_to_image(tf_map, width, height)
	utility.show_with_plt([tf_image, coord_image])

	# 나누어진 면적들을 DFS로 각각 가져온다. tf_map 은 true false 에서 숫자가 써있는 Map 이 된다.
	divided_class, class_total, class_border, class_count, class_length = matrix_processing.get_image_into_divided_plate(tf_map, width, height)
	# 또한 나눈 선들도 각 면적에 포함시켜 나눈다.
	matrix_processing.contours_to_divided_class(tf_map, divided_class, class_total, class_border, class_count, width, height)

	return divided_class, class_total, class_border, class_count, class_length, largest_mask, width, height

if __name__ == "__main__":
	divided_class, class_total, class_border, class_count, class_length, largest_mask, width, height = get_divided_class("Image/chair1.jpg", "chair1_masked.jpg")

	# 일정 크기보다 작은 면적들은 근처에 뭐가 제일 많은지 체크해서 통합시킨다.
	class_number, class_total, class_border, class_count, class_length = \
	merge_small_size(divided_class, list(range(1, class_length + 1)), class_total, class_border, class_count, width, height, min_value=120)
	
	class_number, class_total, class_border, class_count, class_length, class_color = \
	merge_same_color(divided_class, class_number, class_total, class_border, class_count, largest_mask, width, height, sim_score=60)
	
	printing_class = utility.calc_space_with_given_coord(class_number, class_total, \
		[(529, 53), (386, 164), (503, 194), (324, 291), (246, 338), (167, 384), (45, 382), (65, 165), (441, 167), (312, 167), (492, 197), (414, 189), (329, 128), (510, 186), (479, 183), (46, 352), (242, 452), (362, 202), (326, 198)])
	
	dc_image = utility.divided_class_into_image(divided_class, class_number, class_color, width, height, class_number)
	dri_image = utility.divided_class_into_real_image(divided_class, largest_mask, width, height, printing_class)

	utility.show_with_plt([dc_image, dri_image])
