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

# 마스킹된 영역 처리하기
def set_selected_class(divided_class, selected_class, width, height):
	'''
	divided_class 내부에 있는 selected_class list의 원소 class_number 들을 모두 true 로 바꿔준 tf_map을 return 해준다.
	'''
	largest_part_mask = [[False for _ in range(0, width)] for _ in range(0, height)]
	for h in range(0, height):
		for w in range(0, width):
			if divided_class[h][w] in selected_class:
				largest_part_mask[h][w] = True

	return largest_part_mask

def add_n_pixel_mask(divided_class, selected_class, class_border, width, height, n=14):
	'''
	경계를 돌면서 n 픽셀 거리 안에 있는 것들을 현재 Class에 포함시키는 함수.
	'''
	for b in class_border:
		select_outside_pixel(divided_class, selected_class, width, height, b[0], b[1], n)

def select_outside_pixel(divided_class, selected_class, width, height, x, y, n):
	# 주어진 경계에 대해서 만약 거리 안에 있다면 그 class로 변환.
	for x_diff in range(-1*n, n+1, 1):
		for y_diff in range(-1*n, n+1, 1):
			if utility.can_go(x, y, width, height, x_diff=x_diff, y_diff=y_diff):
				if utility.get_pixel_distance((x, y), (x + x_diff, y + y_diff)) <= n:
					divided_class[y + y_diff][x + x_diff] = selected_class

def set_area(divided_class, class_total, change_into):
	for coord in class_total:
		divided_class[coord[1]][coord[0]] = change_into

def set_mask_class(mask, visited, divided_class, class_length, start_index, img_size):
	'''
	mask[h][w] 에서 시작해서 연결된 모든 곳의 좌표에 divided_class list에다가 class_length 값을 대입해 놓는다.
	그 class 의 갯수를 return
	'''
	total_coord, boundary_coordinate, count = bfs(mask, visited, start_index, img_size)
	set_area(divided_class, total_coord, class_length)
	return count, boundary_coordinate

def isTrue(val):
	return val == True

def isFalse(val):
	return val == False

def bfs(mask, visited, start_index, img_size, masking_function=isTrue):
	'''
	mask 되어있는 영역 내에서, divided_class 내부에서 연결된 모든 좌표의 coord와 경계의 coord를 return.
	'''
	count = 1
	que = [(start_index[0], start_index[1])]
	total_coord = []
	boundary_coordinate = []

	# BFS로 Mask 처리하기.
	while que:
		now = que[0]
		del que[0]
		# 방문한 곳은 방문하지 않음.
		if visited[now[1]][now[0]]:
			continue
		
		# Class Dividing 처리.
		visited[now[1]][now[0]] = True
		if masking_function(mask[now[1]][now[0]]):
			total_coord.append(now)
			count += 1
		
		# 경계를 체크하기 위한 Flag
		zero_boundary = False
		for direction in range(0, 4):
			# 해당 방향으로 갈 수 있고, mask 가 칠해진 곳이라면, queue 에 집어넣는다.
			if utility.can_go(now[0], now[1], img_size[0], img_size[1], direction=direction):
				if masking_function(mask[now[1] + utility.dir_y[direction]][now[0] + utility.dir_x[direction]]):
					que.append((now[0] + utility.dir_x[direction], now[1] + utility.dir_y[direction]))
				else:
					# 근처에 0 Class ( 아무것도 없는 공간 == mask[x][y] 가 Flase ) 가 있다면, 경계선이다.
					zero_boundary = True
		if zero_boundary:
			boundary_coordinate.append(now)

	return total_coord, boundary_coordinate, count
    	
def get_divided_class(mask, width, height):
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

	for h in range(height):
		for w in range(width):
			if visited[h][w]:
				continue
			if mask[h][w]:
				# BFS로 True로 되어있는 부분을 탐색.
				class_length += 1
				count, boundary_coordinate = set_mask_class(mask, visited, divided_class, class_length, (w, h), (width, height))
				class_count.append(count)
				class_boundary.append(boundary_coordinate)
	
	return divided_class, class_boundary, class_count, class_length

def get_largest_part(mask, width, height, attach_ratio=0.15):
	'''
		TODO	: attach_ration 이유 정하기
		mask	: list_2D, True False 배열의 mask
		mask의 가장 큰 True 부분을 찾고, 그 부분만 True이고, 나머지는 False로 Output 한다.
	'''
	divided_class, class_boundary, class_count, class_length = get_divided_class(mask, width, height)

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
		add_n_pixel_mask(divided_class, mi, class_boundary[mi-1], width, height)
	
	# 비슷한 것들을 모아서 하나의 Mask로 만들기.
	return set_selected_class(divided_class, max_indx, width, height)

# divided image 만들기.
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
	total_coord, boundary_coordinate, count = bfs(tf_map, visited, start_index, img_size, masking_function=isFalse)
	set_area(divided_class, total_coord, class_length)
	return count, total_coord, boundary_coordinate

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

# 떨어져있는 점 연결하기.
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

def union_find(class_adjac, color_distance, color_threshold):
	'''
	주어진 Class들을 가지고 집합으로 묶어준다.
	'''
	class_length = len(class_adjac)
	union_set = [i for i in range(class_length)]

	for i in range(class_length):
		for j in range(i):
			if class_adjac[i][j] and color_distance[i][j] < color_threshold:
				union_set[j] = parent(i)
	
	# Class 종류를 계산
	class_kind = []
	for i in range(class_length):
		union_set[i] = parent(i)
	
	for i in range(class_length):
		if union_set[i] not in class_kind:
			class_kind.append(union_set[i])
	class_set = [[] for _ in range(len(class_kind))]

	# 각 Class 종류에 따른 list로 묶어서 return.
	for i in range(class_length):
		class_set[union_set[i]].append(i)
	
	return class_set

def parent(union_set, index):
	# Get Parent for given union set.
	now = union_set[index]

	while now != union_set[now]:
		now = union_set[now]

	retValue = now
	now = union_set[index]
	before = index
	
	# 결과로 나온 부모로 모두 변경해준다.
	while now != union_set[now]:
		union_set[before] = retValue
		before = now
		now = union_set[now]

	return retValue

# 좌표와 리스트 탐색
def contours_to_coord(contours):
	'''
	입력받은 외곽선들을 좌표계로 바꾸어서 return 해 준다.
	contours : 2D List. 각 원소들은 외곽선의 sequence 이다.
	'''
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
	'''
	입력받은 리스트 중, line_n 갯수를 넘지 않는 작은 line들은 지워준다.
	'''
	contour_len = list(map(len, contours))

	ret_contours = []

	for i in range(len(contours)):
		if contour_len[i] > line_n:
			ret_contours.append(contours[i])
	return ret_contours

def coord_analysis(coord_list):
	'''
	좌표를 둘러싼 사각형의 양 위 아래, coord 평균을 조사.
	'''
	x_min = utility.INT_MAX
	x_max = 0
	y_min = utility.INT_MAX
	y_max = 0
	coord_add = (0, 0)
	coord_len = len(coord_list)

	for c in coord_list:
		coord_add[0] += c[0]
		coord_add[1] += c[1]
		if c[0] < x_min:
			x_min = c[0]
		if c[1] < y_min:
			y_min = c[1]
		if c[0] > x_max:
			x_max = c[0]
		if c[1] < y_max:
			y_max = c[1]
	return [x_min, x_max, y_min, y_max], (coord_add[0] / coord_len, coord_add[1] / coord_len)

def find_adjac_class_number(divided_class, class_border, width, height):
	# 근처의 인접한 class number를 return.
	ret_class_number = []
	for c in class_border:
		for direction in range(0, 4):
			if utility.can_go(c[0], c[1], width, height, direction=direction):
				if divided_class[c[1] + utility.dir_y[direction]][c[0] + utility.dir_x[direction]] not in ret_class_number:
					ret_class_number.append(divided_class[c[1] + utility.dir_y[direction]][c[0] + utility.dir_x[direction]])
	return ret_class_number

# 외곽선 분해하기
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

def check_border(divided_class, class_border, width, height):
	# Get only class border coordination.
	# 입력된 border가 실제 border인지 아닌지를 판단 해 준다.
	if len(class_border) < 4:
		return class_border
	ret_class_border = []
	for coord in class_border:
		if is_border(divided_class, coord, width, height):
			ret_class_border.append(coord)
	return ret_class_border

def is_border(divided_class, coord, width, height):
	# Check if given coord is outside of area.
	# 단순히 외곽에 0이 있으면 외곽점으로 본다.
	neighbor_list = []
	for direction in range(4):
		if utility.can_go(coord[0], coord[1], width, height, direction=direction):
			neighbor = divided_class[coord[1] + utility.dir_y[direction]][coord[0] + utility.dir_x[direction]]
			if neighbor not in neighbor_list:
				neighbor_list.append(neighbor)
	return len(neighbor_list) > 1

# 근처 Pixel 탐색
def get_around_pixel_list(divided_class, width, height, w, h):
	class_kind = []
	class_number = []
	class_coord = []
	for diff in [(0, -1), (0, 1), (1, -1), (1, 0), (1, 1), (-1, -1), (-1, 0), (-1, 1)]:
		if utility.can_go(w, h, width, height, x_diff=diff[0], y_diff=diff[1]):
			if divided_class[h + diff[1]][w + diff[0]] in class_kind:
				class_number[class_kind.index(divided_class[h + diff[1]][w + diff[0]])] += 1
				class_coord[class_kind.index(divided_class[h + diff[1]][w + diff[0]])].append((w + diff[0], h + diff[1]))
			else:
				class_kind.append(divided_class[h + diff[1]][w + diff[0]])
				class_number.append(0)
				class_coord.append([(w + diff[0], h + diff[1])])
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
