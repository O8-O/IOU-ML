import multiprocessing as mp
import os
import math
import sys
import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from modules.predictor import VisualizationDemo

# constants
WINDOW_NAME = "IOU Segmentation"
FILE_NAME = 1
INT_MAX = sys.maxsize
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

def get_only_instance_image(input_file, masks, height, width, output_file=None, show=False):
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
	if show:
		cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
		cv2.imshow(WINDOW_NAME, masked_image)
		if cv2.waitKey(0) == 27:
			visualized_output.save(output_file)
	return masked_image

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

def get_contours(frame, start=190, diff=30):
	'''
	외곽선과 그 외곽선의 그 계층관계를 Return ( contours, heirachy )
	frame = cv2.imread 값.
	'''
	# Converting the image to grayscale.
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Smoothing without removing edges.
	gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)

	# Histogram Normalization
	gray_CLAHE = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(16, 16)).apply(gray)
	gray_filtered = cv2.bilateralFilter(gray_CLAHE, 7, 50, 50)

	# Using the Canny filter to get contours
	edges_high_thresh = cv2.Canny(gray_filtered, start, start + diff)

	# Using Contours, search Contours.
	return cv2.findContours(edges_high_thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

def contours_to_coord(contours):
	coords = []
	for cinstances in contours:
		temp = []
		for c in cinstances:
				now = (c[0][0], c[0][1])
				if now not in temp:
					temp.append(now)
		coords.append(temp)
	return coords

def delete_line_threshold(contours, line_n=40):
	contour_len = list(map(len, contours))

	ret_contours = []

	for i in range(0, len(contours)):
		if contour_len[i] > line_n:
			ret_contours.append(contours[i])
	return ret_contours

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

def make_tf_map(coords, width, height):
	tf_map = [[False for _ in range(width)] for _ in range(height)]
	for coord in coords:
		for c in coord:
			tf_map[c[1]][c[0]] = True
	for x in range(width):
		tf_map[0][x] = True
		tf_map[-1][x] = True
	for y in range(height):
		tf_map[y][0] = True
		tf_map[y][-1] = True
	return tf_map

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
			if can_go(coord[0], coord[1], width, height, x_diff=x_diff, y_diff=y_diff):
				if tf_map[coord[1] + y_diff][coord[0] + x_diff]:
					# 해당 좌표로 이동 가능하면서, 해당 좌표의 값이 True 인 경우,
					l_count += 1
	for y_diff in range(-1 * n , n):
		for x_diff in range(1, n + 1):
			if can_go(coord[0], coord[1], width, height, x_diff=x_diff, y_diff=y_diff):
				if tf_map[coord[1] + y_diff][coord[0] + x_diff]:
					# 해당 좌표로 이동 가능하면서, 해당 좌표의 값이 True 인 경우,
					r_count += 1
	# Check UD
	for x_diff in range(-1 * n , n):
		for y_diff in range(-1 * n, 0):
			if can_go(coord[0], coord[1], width, height, x_diff=x_diff, y_diff=y_diff):
				if tf_map[coord[1] + y_diff][coord[0] + x_diff]:
					# 해당 좌표로 이동 가능하면서, 해당 좌표의 값이 True 인 경우,
					u_count += 1
	for x_diff in range(-1 * n , n):
		for y_diff in range(1, n + 1):
			if can_go(coord[0], coord[1], width, height, x_diff=x_diff, y_diff=y_diff):
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

def connect_nearest_point(tf_map, point, width, height, before, n=5, diff_n=5, max_itor=30, side_diff_k=0.2):
	# Find Nearest Point at given point.
	# Do not search before[-n:] ( if before length is longer than n, else search in before[:])
	nearest_point = (-1, -1)
	p_index = before.index(point)
	one_more = False

	if p_index < diff_n:
		(x_diff, y_diff) = get_average_diff(before[p_index + 1:p_index + diff_n + 1])
	elif p_index > len(before) - diff_n:
		(x_diff, y_diff) = get_average_diff(before[p_index - diff_n:p_index])
	else:
		# 중간에 좌표가 있을 때, 양 옆으로의 좌표 크기 차이가 많이 안나는 경우에만 연결함.
		(x_diff, y_diff) = get_average_diff(before[p_index + 1:p_index + diff_n + 1])
		(x_diff_else, y_diff_else) = get_average_diff(before[p_index - diff_n:p_index])
		large_xy = x_diff * y_diff_else if x_diff * y_diff_else > x_diff_else * y_diff else x_diff_else * y_diff
		small_xy = x_diff_else * y_diff if x_diff * y_diff_else > x_diff_else * y_diff else x_diff * y_diff_else
		if large_xy * (1 - side_diff_k) < small_xy and small_xy * (1 + side_diff_k)  > large_xy:
			one_more = True
	
	for i in range(max_itor):
		now = (point[0] + x_diff * i, point[1] + y_diff * i)
		nearest_point = find_nearest_point(tf_map, now, width, height, before, n=n)
		if nearest_point != (-1, -1):
			# 연결 하기.
			connect_lines(tf_map, point, nearest_point)
			break

	# 만약 양쪽의 기울기가 크게 차이난다면
	if one_more:
		for i in range(max_itor):
			now = (point[0] + x_diff_else * i, point[1] + y_diff_else * i)
			nearest_point = find_nearest_point(tf_map, now, width, height, before, n=n)
			if nearest_point != (-1, -1):
				# 연결 하기.
				connect_lines(tf_map, point, nearest_point)
				break

def find_nearest_point(tf_map, point, width, height, before, n=5):
	nearest_point = (-1, -1)
	nearest_length = INT_MAX
	search_start_x = point[0] - n if point[0] > n else 0
	search_start_x = int(search_start_x)
	search_start_y = point[1] - n if point[1] + n < width else width
	search_start_y = int(search_start_y)
	search_end_x = point[0] + n if point[0] + n < width else width
	search_end_x = int(search_end_x)
	search_end_y = point[1] + n if point[1] + n < height else height
	search_end_y = int(search_end_y)
	# tf_map에 가까운 곳이 n 거리 이내에 있는가?
	for x in range(search_start_x, search_end_x):
		for y in range(search_start_y, search_end_y):
			if tf_map[y][x] == True:
				length = get_euclidean_distance(point, (x, y))
				if (x, y) not in before and length < n:
					if nearest_length > length:
						nearest_length = length
						nearest_point = (x, y)
				elif x == width - 1 or x == 0 or y == height - 1 or y == 0:
					if nearest_length > length:
						nearest_length = length
						nearest_point = (x, y)
	return nearest_point

def connect_lines(tf_map, start_point, dest_point):
	# 시작점과 목표 지점 사이에 직선을 긋는다.
	start_x = start_point[0]
	start_y = start_point[1]
	dest_x = dest_point[0]
	dest_y = dest_point[1]

	large_x = start_x if start_x > dest_x else dest_x
	small_x = dest_x if start_x > dest_x else start_x
	large_y = start_y if start_y > dest_y else dest_y
	small_y = dest_y if start_y > dest_y else start_y

	if start_x == dest_x:
		for y in range(small_y + 1, large_y):
			tf_map[y][start_x] = True
	elif start_y == dest_y:
		for x in range(small_x + 1, large_x):
			tf_map[start_y][x] = True
	else:
		diff_x = large_x - small_x
		diff_y = large_y - small_y
		if diff_x < diff_y:
			if start_x == small_x:
				before_y = start_y
			else:
				before_y = dest_y
			for x in range(small_x, large_x):
				next_y = before_y + int((dest_y - start_y) / (dest_x - start_x))
				if next_y < before_y:
					temp = next_y
					next_y = before_y
					before_y = temp
				for y in range(before_y, next_y):
					tf_map[y][x] = True
				if int((dest_y - start_y) / (dest_x - start_x)) > 0:
					before_y = next_y
		else:
			if start_y == small_y:
				before_x = start_x
			else:
				before_x = dest_x
			for y in range(small_y, large_y):
				next_x = before_x + int((dest_x - start_x) / (dest_y - start_y))
				if next_x < before_x:
					temp = next_x
					next_x = before_x
					before_y = temp
				for x in range(before_x, next_x):
					tf_map[y][x] = True
				if int((dest_x - start_x) / (dest_y - start_y)) > 0:
					before_x = next_x

def get_euclidean_distance(point1, point2):
	return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def get_average_diff(points):
	x_diff_add = 0
	y_diff_add = 0
	n = len(points)
	for i in range(0, n - 1):
		(x_diff, y_diff) = get_diff(points[i], points[i+1])
		x_diff_add += x_diff
		y_diff_add += y_diff

	return (x_diff_add/n, y_diff_add/n)

def get_diff(point1, point2):
	return (point1[0] - point2[0], point1[1] - point2[1])

def coord_to_image(coordinates, width, height):
	coord_image = [[0 for _ in range(0, width)] for _ in range(0, height)]
	# Change Coordinates into images.
	for coord_list in coordinates:
		for c in coord_list:
			coord_image[c[1]][c[0]] = 255
	coord_image = np.array(coord_image, np.uint8)
	return coord_image

def tf_map_to_image(tf_map, width, height):
	tf_image = [[0 for _ in range(0, width)] for _ in range(0, height)]
	for h in range(height):
		for w in range(width):
			if tf_map[h][w]:
				tf_image[h][w] = 255
	tf_image = np.array(tf_image, np.uint8)
	return tf_image

def show_with_plt(imgs):
	# Add and show Image
	import matplotlib.pyplot as plt
	fig = plt.figure()
	if len(imgs) == 1:
		rows = 1
		cols = 1
	elif len(imgs) == 2:
		rows = 1
		cols = 2
	elif len(imgs) == 3:
		rows = 1
		cols = 3
	elif len(imgs) % 2:
		rows = len(imgs) // 2
		cols = 2
	else:
		rows = 2
		cols = len(imgs) // 2 + 1

	for i in range(0, len(imgs)):
		ax = fig.add_subplot(rows, cols, i + 1)
		ax.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
		ax.axis("off")

	plt.show()

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
		masked_image = get_only_instance_image(args_list[FILE_NAME], mask, height, width)
		# 잘린 이미지를 통해 외곽선을 얻어서 진행.
		contours, heirarchy = get_contours(masked_image)
		coords = contours_to_coord(contours)
		coords = delete_line_threshold(coords, line_n=40)
		cycle_list = []
		noncycle_list = []

		for c in coords:
			cycled, noncycled = divide_cycle(c)
			if len(cycled) != 0:
				cycle_list += cycled
			if len(noncycled) != 0:
				noncycle_list += noncycled
		
		tf_map = make_tf_map(noncycle_list, width, height)
		for nc in noncycle_list:
			border_point = find_border_k_tf_map(tf_map, nc, width, height, n=5, k=2, hard_check=False)
			for b in border_point:
				connect_nearest_point(tf_map, b, width, height, nc)
		tf_image = tf_map_to_image(tf_map, width, height)
		
		coord_image = coord_to_image(noncycle_list, width, height)
		show_with_plt([coord_image, tf_image])
