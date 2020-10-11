import cv2
import numpy as np
import utility

'''
Image 파일들은 전부 height - width 순서.
[
	[p1, p2, p3 .. pn],
	[p1, p2, p3 .. pn]
]
같이 구현 되어서 height( y ) 가 첫번째, width ( x ) 가 두번쨰로 들어가야 한다.

좌표는 전부 (x, y) 순서. width 는 x, height 는 y임을 기억하자.
'''

def get_only_instance_image(input_file, mask, height, width, output_file=None, show=False):
	'''
	파일 이름을 입력받아서 실제로 masking 된 부분의 사진만 output 해 준다.
	input_file	: string, 파일 이름
	masks		: 2차원 list, width / height 크기, True 면 그 자리에 객체가 있는것, 아니면 없음.
	width		: int, 너비
	height		: int, 높이
	output_file	: string, output_file 이름
	'''
	if output_file == None:
		output_file = input_file.split(".")[0] + "_masked" + input_file.split(".")[1]
	original = cv2.imread(input_file)
	masked_image = np.zeros([height, width ,3], dtype=np.uint8)
	mask_num = 0

	for h in range(0, height):
		for w in range(0, width):
				for c in range(0, 3):
					masked_image[h][w][c] = (original[h][w][c] if mask[h][w] else 0)
					if mask[h][w]:
						mask_num += 1

	return masked_image, mask_num

def set_selected_class(divided_class, selected_class, height, width):
	'''
	divided_class 내부에 있는 selected_class list의 원소 class_number 들을 모두 true 로 바꿔준 tf_map을 return 해준다.
	'''
	largest_part_mask = [[False for _ in range(0, width)] for _ in range(0, height)]
	for h in range(0, height):
		for w in range(0, width):
			if divided_class[h][w] in selected_class:
				largest_part_mask[h][w] = True

	return largest_part_mask

def add_n_pixel_mask(divided_class, selected_class, class_border, height, width, n=10):
	'''
	경계를 돌면서 n 픽셀 거리 안에 있는 것들을 현재 Class에 포함시키는 함수.
	'''
	for b in class_border:
		select_outside_pixel(divided_class, selected_class, height, width, b[0], b[1], n)

def select_outside_pixel(divided_class, selected_class, height, width, x, y, n):
	# 주어진 경계에 대해서 만약 거리 안에 있다면 그 class로 변환.
	for x_diff in range(-1*n, n+1, 1):
		for y_diff in range(-1*n, n+1, 1):
			if utility.can_go(x, y, height, width, x_diff=x_diff, y_diff=y_diff):
				if utility.get_pixel_distance((x, y), (x + x_diff, y + y_diff)) <= n:
					divided_class[x + x_diff][y + y_diff] = selected_class

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
			if utility.can_go(now[0], now[1], img_size[0], img_size[1], direction=direction):
				if mask[now[0] + utility.dir_x[direction]][now[1] + utility.dir_y[direction]]:
					que.append((now[0] + utility.dir_x[direction], now[1] + utility.dir_y[direction]))
				else:
					# 근처에 0 Class ( 아무것도 없는 공간 == mask[x][y] 가 Flase ) 가 있다면, 경계선이다.
					zero_boundary = True
		if zero_boundary:
			boundary_coordinate.append((now[0], now[1]))
	return count, boundary_coordinate

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

def get_largest_part(mask, height, width, attach_ratio=0.15):
	'''
		TODO	: attach_ration 이유 정하기
		mask	: list_2D, True False 배열의 mask
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

def connect_nearest_point(tf_map, point, width, height, before, n=5, diff_n=5, max_itor=30, side_diff_k=0.2):
	# Find Nearest Point at given point.
	# Do not search before[-n:] ( if before length is longer than n, else search in before[:])
	nearest_point = (-1, -1)
	p_index = before.index(point)
	one_more = False

	if p_index < diff_n:
		(x_diff, y_diff) = utility.get_average_diff(before[p_index + 1:p_index + diff_n + 1])
	elif p_index > len(before) - diff_n:
		(x_diff, y_diff) = utility.get_average_diff(before[p_index - diff_n:p_index])
	else:
		# 중간에 좌표가 있을 때, 양 옆으로의 좌표 크기 차이가 많이 안나는 경우에만 연결함.
		(x_diff, y_diff) = utility.get_average_diff(before[p_index + 1:p_index + diff_n + 1])
		(x_diff_else, y_diff_else) = utility.get_average_diff(before[p_index - diff_n:p_index])
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
	nearest_length = utility.INT_MAX
	search_start_x = int(point[0] - n) if point[0] > n else 0
	search_start_y = int(point[1] - n) if point[1] + n < width else width
	search_end_x = int(point[0] + n) if point[0] + n < width else width
	search_end_y = int(point[1] + n) if point[1] + n < height else height
	# tf_map에 가까운 곳이 n 거리 이내에 있는가?
	for x in range(search_start_x, search_end_x):
		for y in range(search_start_y, search_end_y):
			if tf_map[y][x] == True:
				length = utility.get_euclidean_distance(point, (x, y))
				if (x, y) not in before and length < n:
					if nearest_length > length:
						nearest_length = length
						nearest_point = (x, y)
				elif x == width - 1 or x == 0 or y == height - 1 or y == 0:
					if nearest_length > length:
						nearest_length = length
						nearest_point = (x, y)
	return nearest_point

