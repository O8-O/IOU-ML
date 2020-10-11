import multiprocessing as mp
import sys
import cv2
import numpy as np
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

def divide_cycle(coords):
	# Divide coords into cycle and noncycle list.
	length = len(coords)
	before = []
	cycle_list = []
	noncycle_list = []
	indx = 0
	while indx < length:
		now = coords[indx]
		if now not in before:
			before.append(now)
		else:
			# If contour[indx] is in before list, it`s Cycle.
			now_index = before.index(now)
			if now_index != 0:
				cycle_list.append(before[now_index:])
				noncycle_list.append(before[0:now_index])
			else:
				cycle_list.append(before)
			before = []
		indx += 1
	noncycle_list.append(before)
	return cycle_list, noncycle_list

def find_border_k_tf_map(tf_map, coord, width, height, n=5, k=4, hard_check=False):
	'''
	주어진 점 coord[i] 에서 n 왼쪽과 n 오른쪽 / n 윗쪽과 n 아래쪽만큼 범위 내에서 특정 갯수 k개만큼 있는 것이 차이나면 외곽으로 본다.
	tf_map[coord[i][1] - n][coord[i][0] - n] == True 값이 없다면 외곽으로 본다.
	'''
	border_point = []
	for c in coord:
		if is_coord_border(tf_map, c, width, height, n, k, hard_check=hard_check):
			border_point.append(c)
	return border_point

def is_coord_border(tf_map, coord, width, height, n, k, hard_check=False):
	'''
	주어진 coord 좌표가 한 contours의 외곽인지 ( 따라서 이어줘야 하는 것인지 ) 판별 하는 함수.
	'''
	check_lr = False	# Left and Right Check.
	check_ud = False	# Up and Down check.
	l_count = 0	# Left Coord number.
	r_count = 0 # Right Coord number.
	u_count = 0 # Up Coord number.
	d_count = 0 # Down Coord number.

	# Check LR
	for y_diff in range(-1 * n , n):
		for x_diff in range(-1 * n, 0):
			if utility.can_go(coord[0], coord[1], width, height, x_diff=x_diff, y_diff=y_diff):
				if tf_map[coord[1] + y_diff][coord[0] + x_diff]:
					# 해당 좌표로 이동 가능하면서, 해당 좌표의 값이 True 인 경우,
					l_count += 1
	for y_diff in range(-1 * n , n):
		for x_diff in range(1, n + 1):
			if utility.can_go(coord[0], coord[1], width, height, x_diff=x_diff, y_diff=y_diff):
				if tf_map[coord[1] + y_diff][coord[0] + x_diff]:
					# 해당 좌표로 이동 가능하면서, 해당 좌표의 값이 True 인 경우,
					r_count += 1
	# Check UD
	for x_diff in range(-1 * n , n):
		for y_diff in range(-1 * n, 0):
			if utility.can_go(coord[0], coord[1], width, height, x_diff=x_diff, y_diff=y_diff):
				if tf_map[coord[1] + y_diff][coord[0] + x_diff]:
					# 해당 좌표로 이동 가능하면서, 해당 좌표의 값이 True 인 경우,
					u_count += 1
	for x_diff in range(-1 * n , n):
		for y_diff in range(1, n + 1):
			if utility.can_go(coord[0], coord[1], width, height, x_diff=x_diff, y_diff=y_diff):
				if tf_map[coord[1] + y_diff][coord[0] + x_diff]:
					# 해당 좌표로 이동 가능하면서, 해당 좌표의 값이 True 인 경우,
					d_count += 1
	check_lr = abs(l_count - r_count) < k
	check_ud = abs(u_count - d_count) < k
	# hard_check가 체크되어 있다면, 두 조건을 모두 달성해야 True, 아니라면 둘 중 하나만 달성해도 괜찮음.
	# 외곽이라면 True 아니면 False를 Return 해야하므로, not을 붙여준다.
	if hard_check:
		return not( check_lr and check_ud )
	else:
		return not( check_lr or check_ud )

def get_image_into_divided_plate(tf_map, width, height):
	'''
		tf_map : 경계선이 True로, 나머진 False로 구분되어있는 tf_map.
		return
			divided_class : 0 ~ N list_2D. 각각은 아무것도 없으면 0, 아니면 각 class number.
			class_count : 각 class들의 숫자.
			class_length : class의 갯수
	'''
	# Initializing.
	divided_class = [[0 for _ in range(0, width)] for _ in range(0, height)]
	visited = [[False for _ in range(0, width)] for _ in range(0, height)]
	class_total = []
	class_boundary = []
	class_count = []
	class_length = 0

	for h in range(0, height):
		for w in range(0, width):
			if visited[h][w]:
				continue
			if tf_map[h][w] == False:
				# BFS로 False 되어있는 부분을 탐색. True 되어있는 부분은 넘어가지 않는다.
				class_length += 1
				count, total_list, boundary_coordinate = set_tf_map_class(tf_map, visited, divided_class, class_length, (w, h), (width, height))
				class_count.append(count)
				class_total.append(total_list)
				class_boundary.append(boundary_coordinate)
	
	return divided_class, class_total, class_boundary, class_count, class_length

def set_tf_map_class(tf_map, visited, divided_class, class_length, start_index, img_size):
	'''
	mask[h][w] 에서 시작해서 연결된 모든 곳의 좌표에 divided_class list에다가 class_length 값을 대입해 놓는다.
	그 class 의 갯수를 return
	'''
	count = 1
	que = [start_index]
	boundary_coordinate = []
	total_list = []

	# BFS로 tf_map 처리하기.
	while que:
		now = que[0]
		del que[0]
		# 방문한 곳은 방문하지 않음.
		if visited[now[1]][now[0]]:
			continue
		
		# Class Dividing 처리.
		visited[now[1]][now[0]] = True
		# False 인 부분만 입력.
		if not tf_map[now[1]][now[0]]:
			total_list.append((now[0], now[1]))
			divided_class[now[1]][now[0]] = class_length
			count += 1
		
		# 경계를 체크하기 위한 Flag
		zero_boundary = False
		for direction in range(0, 4):
			if utility.can_go(now[0], now[1], img_size[0], img_size[1], direction=direction):
				# tf_map 의 False 인 부분만 찾아서 Plane 화 해야한다.
				if not tf_map[now[1] + utility.dir_y[direction]][now[0] + utility.dir_x[direction]]:
					que.append((now[0] + utility.dir_x[direction], now[1] + utility.dir_y[direction]))
				else:
					# 근처에 tf_map[x][y] 가 True 인 것이 있다면, 경계선이다.
					zero_boundary = True
		if zero_boundary:
			boundary_coordinate.append((now[0], now[1]))
	return count, total_list, boundary_coordinate

def get_average_color(image, total_class, tc_num):
	aver_r = 0
	aver_g = 0
	aver_b = 0

	for coord in total_class:
		rgb = image[coord[1]][coord[0]]
		aver_r += rgb[0]
		aver_g += rgb[1]
		aver_b += rgb[2]
	
	return [int(aver_r/tc_num), int(aver_g/tc_num), int(aver_b/tc_num)]

def get_around_pixel_list(divided_class, width, height, w, h):
	class_kind = []
	class_number = []
	class_coord = []
	for diff in [(0, -1), (0, 1), (1, -1), (1, 0), (1, 1), (-1, -1), (-1, 0), (-1, 1)]:
		if utility.can_go(h, w, height, width, x_diff=diff[0], y_diff=diff[1]):
			if divided_class[h + diff[0]][w + diff[1]] in class_kind:
				class_number[class_kind.index(divided_class[h + diff[0]][w + diff[1]])] += 1
				class_coord[class_kind.index(divided_class[h + diff[0]][w + diff[1]])].append((w + diff[1], h + diff[0]))
			else:
				class_kind.append(divided_class[h + diff[0]][w + diff[1]])
				class_number.append(0)
				class_coord.append([(w + diff[1], h + diff[0])])
	return class_kind, class_number, class_coord

def get_around_pixel(divided_class, width, height, w, h):
	# 8 방위 중에서 가장 많은 Class Number를 가져온다. 0이 가장 많아도 그냥 0으로 가져온다.
	class_kind, class_number, _ = get_around_pixel_list(divided_class, width, height, w, h)
	
	# 가장 많은 것을 가져와서 Return.
	largest_index = 0
	for i in range(len(class_kind)):
		if class_number[i] > class_number[largest_index]:
			largest_index = i
		elif class_number[i] == class_number[largest_index]:
			if class_kind[i] > class_number[largest_index]:
				largest_index = i
	
	return class_kind[largest_index]

def contours_to_divided_class(tf_map, divided_class, class_total, class_border, class_count, width, height):
	# divided_class 내부에 있는 0 class들을 근처의 다른 Class로 설정한다.
	doing_queue = []
	for h in range(height):
		for w in range(width):
			# 만약 경계선이라면
			if tf_map[h][w] == True:
				doing_queue.append((w, h))
	
	do_time = len(doing_queue) * 3
	while len(doing_queue) != 0:
		do_time -= 1
		now = doing_queue[0]
		del doing_queue[0]
		now_class = get_around_pixel(divided_class, width, height, now[0], now[1])
		if now_class != 0:
			class_total[now_class - 1].append(now)
			class_border[now_class - 1].append(now)
			class_count[now_class - 1] += 1
			divided_class[now[1]][now[0]] = now_class
		else:
			doing_queue.append(now)
		if do_time < 0:
			break

def get_around_largest_area(divided_class, width, height, class_border, my_class):
	'''
	근처의 영역 중에 가장 많이 자신과 붙어있는 영역의 Class Number를 Return.
	'''
	total_kind = []
	total_number = []
	total_coord = []

	for coord in class_border:
		class_kind, class_number, class_coord = get_around_pixel_list(divided_class, width, height, coord[0], coord[1])
		# 모든 Return 값에 대해서
		for ck in class_kind:
			# 이미 있던거면 class coord 가 있던건지 체크해서 입력
			if ck in total_kind:
				ck_index = total_kind.index(ck)
				for cc in class_coord[class_kind.index(ck)]:
					# 모든 coord return 값에 대해서 없는것만 추가하고 숫자를 늘림.
					if cc not in total_coord[ck_index]:
						total_number[ck_index] += 1
						total_coord[ck_index].append(cc)
			else:
				# 처음 나온 Class 면 추가한다.
				ck_index = class_kind.index(ck)
				total_kind.append(ck)
				total_number.append(class_number[ck_index])
				total_coord.append(class_coord[ck_index])

	largest_number = -1
	largest_index = -1
	for i in range(len(total_number)):
		# 자신이 아닌 가장 큰 Class를 뽑아온다.
		if total_number[i] > largest_number and total_kind[i] != my_class:
			largest_number = total_number[i]
			largest_index = i
	
	if largest_number == -1:
		return -1
	return total_kind[largest_index]

def merge_around(divided_class, class_number, class_total, class_border, class_count, merge_indx, width, height):
	large_class_number = get_around_largest_area(divided_class, width, height, class_border[merge_indx], class_number[merge_indx])
	if large_class_number == 0:
		# Set into 0 Class
		set_area(divided_class, class_total[merge_indx], 0)
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

def set_area(divided_class, class_total, change_into):
	for coord in class_total:
		divided_class[coord[1]][coord[0]] = change_into

def is_border(divided_class, coord, width, height):
	# Check if given coord is outside of area.
	neighbor_list = []
	for direction in range(4):
		if utility.can_go(coord[0], coord[1], width, height, direction=direction):
			neighbor = divided_class[coord[1] + utility.dir_y[direction]][coord[0] + utility.dir_x[direction]]
			if neighbor not in neighbor_list:
				neighbor_list.append(neighbor)
	return len(neighbor_list) > 1
				
def check_border(divided_class, class_border, width, height):
	# Get only class border coordination.
	ret_class_border = []
	for coord in class_border:
		if is_border(divided_class, coord, width, height):
			ret_class_border.append(coord)
	return ret_class_border

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

	set_area(divided_class, ret_class_total[merge_base_index], ret_class_numbers[merge_base_index])
	ret_class_border[merge_base_index] = check_border(divided_class, ret_class_border[merge_base_index], width, height)

	return ret_class_numbers, ret_class_total, ret_class_border, ret_class_count, len(ret_class_total), 

def get_class_color(image, class_total, class_count):
	class_color = []
	for i in range(len(class_total)):
		class_color.append(get_average_color(image, class_total[i], class_count[i]))
	return class_color

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
		class_color = get_class_color(largest_mask, ret_class_total, ret_class_count)
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
	class_color = get_class_color(largest_mask, ret_class_total, ret_class_count)
	return ret_class_numbers, ret_class_total, ret_class_border, ret_class_count, class_length, class_color

def delete_unavailable_color(divided_class, class_numbers, class_total, class_border, class_count, largest_mask, min_add_up=40):
	class_length = len(class_numbers)
	ret_class_numbers = []
	ret_class_total = []
	ret_class_border = []
	ret_class_count = []
	class_color = get_class_color(largest_mask, class_total, class_count)

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
		set_area(divided_class, class_total[zero_list[z]], 0)

	return ret_class_numbers, ret_class_total, ret_class_border, ret_class_count, len(ret_class_total)

def calc_space_with_given_coord(class_number, class_total, given_coord):
	'''
	사용자가 입력한 좌표들로, 그 좌표가 우리가 가지고있는 class_total 좌표계 내에 존재할시. 그 class_number를 모아서 return 해 준다.
	'''
	ret_class_number = []

	for coord in given_coord:
		for cn in range(len(class_total)):
			if coord in class_total[cn]:
				if class_number[cn] not in ret_class_number:
					ret_class_number.append(class_number[cn])
				break

	return ret_class_number

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
	# Mask 칠한 이미지 중에서 가장 큰것만 가지고 진행함.
	largest_mask = []
	largest_mask_number = -1

	for i in range(0, instance_number):
		mask = matrix_processing.get_largest_part(masks[i], height, width)
		masked_image, mask_num = matrix_processing.get_only_instance_image(args_list[FILE_NAME], mask, height, width)
		if mask_num > largest_mask_number:
			largest_mask = masked_image
			largest_mask_number = mask_num
		
	# 잘린 이미지를 통해 외곽선을 얻어서 진행.
	contours, heirarchy = image_processing.get_contours(largest_mask)
	coords = image_processing.contours_to_coord(contours)
	coords = image_processing.delete_line_threshold(coords, line_n=40)
	cycle_list = []
	noncycle_list = []

	for c in coords:
		cycled, noncycled = divide_cycle(c)
		if len(cycled) != 0:
			cycle_list += cycled
		if len(noncycled) != 0:
			noncycle_list += noncycled
	
	# 잘린 외곽선들을 True-False List로 바꿔서 각각 가장 가까운 곳에 연결.
	tf_map = utility.make_tf_map(noncycle_list, width, height)
	for nc in noncycle_list:
		# 가장자리가 될 포인트를 잡는다.
		border_point = find_border_k_tf_map(tf_map, nc, width, height, n=5, k=2, hard_check=False)
		for b in border_point:
			# 가장자리에서 가장 가까운 외곽선으로 연결한다.
			matrix_processing.connect_nearest_point(tf_map, b, width, height, nc)
	
	# TF Image searching.
	coord_image = utility.coord_to_image(coords, width, height)
	tf_image = utility.tf_map_to_image(tf_map, width, height)
	utility.show_with_plt([tf_image, coord_image])

	# 나누어진 면적들을 DFS로 각각 가져온다. tf_map 은 true false 에서 숫자가 써있는 Map 이 된다.
	divided_class, class_total, class_border, class_count, class_length = get_image_into_divided_plate(tf_map, width, height)
	# 또한 나눈 선들도 각 면적에 포함시켜 나눈다.
	contours_to_divided_class(tf_map, divided_class, class_total, class_border, class_count, width, height)

	# 일정 크기보다 작은 면적들은 근처에 뭐가 제일 많은지 체크해서 통합시킨다.
	class_number, class_total, class_border, class_count, class_length = \
	merge_small_size(divided_class, range(1, class_length + 1), class_total, class_border, class_count, width, height, min_value=120)
	
	class_number, class_total, class_border, class_count, class_length, class_color = \
	merge_same_color(divided_class, range(1, class_length + 1), class_total, class_border, class_count, largest_mask, width, height, sim_score=60)
	
	printing_class = calc_space_with_given_coord(class_number, class_total, \
		[(529, 53), (386, 164), (503, 194), (324, 291), (246, 338), (167, 384), (45, 382), (65, 165), (441, 167), (312, 167), (492, 197), (414, 189), (329, 128), (510, 186), (479, 183), (46, 352), (242, 452), (362, 202), (326, 198)])
	
	dc_image = utility.divided_class_into_image(divided_class, class_number, class_color, width, height, class_number)
	dri_image = utility.divided_class_into_real_image(divided_class, largest_mask, width, height, printing_class)

	utility.show_with_plt([dc_image, dri_image])
