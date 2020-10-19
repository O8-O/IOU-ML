import cv2
import numpy as np
import sys

dir_x = [0, 0, 1, -1]
dir_y = [1, -1, 0, 0]

INT_MAX = sys.maxsize

# Distance 함수들
def get_rgb_distance(pixel1, pixel2):
	'''
	pixel1, 2 : 3차원 list with rgb value ( 10 진법 ).
	단순한 rgb 값의 euclidean distance를 return 한다.
	'''
	distance = 0
	for i in range(3):
		distance += abs(pixel1[i] - pixel2[i]) ** 2

	return distance ** 1/2

def get_cielab_distance(pixel1, pixel2):
	'''
	pixel1, 2 : 3차원 list with rgb value ( 10 진법 ).
	rgb 값의 red mean color distance를 return 한다.
	'''
	npPixel1 = np.array([[pixel1]], dtype='uint8')
	lab1 = cv2.cvtColor(npPixel1, cv2.COLOR_BGR2Lab).tolist()[0][0]
	npPixel2 = np.array([[pixel2]], dtype='uint8')
	lab2 = cv2.cvtColor(npPixel2, cv2.COLOR_BGR2Lab).tolist()[0][0]

	return get_rgb_distance(lab1, lab2)

def get_color_distance_map(class_color, class_length, distance_func=get_cielab_distance):
	'''
	class_color : each class average colors list.
	return : Color distance map for input class_color.
	'''
	color_distance_map = [[0 for _ in range(class_length)] for _ in range(class_length)]

	for i in range(class_length):
		for j in range(i+1, class_length):
			color_distance_map[i][j] = distance_func(class_color[i], class_color[j])
			color_distance_map[j][i] = distance_func(class_color[i], class_color[j])
	
	return color_distance_map

def get_pixel_distance(now, dest):
	return abs(now[0] - dest[0]) + abs(now[1] - dest[1])

def get_each_pixel_distance(now, dest):
	return now[0] - dest[0], now[1] - dest[1]

def get_average_diff(points):
	x_diff_add = 0
	y_diff_add = 0
	n = len(points)
	for i in range(0, n - 1):
		x_diff, y_diff = get_each_pixel_distance(points[i], points[i+1])
		x_diff_add += x_diff
		y_diff_add += y_diff

	return (x_diff_add/n, y_diff_add/n)

def get_euclidean_distance(point1, point2):
	return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 1/2

# Print 함수들
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

def print_image(image, output_file=None, window_name=None):
	if window_name == None:
		window_name = "TEMP_WINDOWS"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	cv2.imshow(window_name, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

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

# image-like 만드는 함수들.
def make_tf_map(coords, width, height, border=True):
	'''
	coords의 좌표들이 true이고, 가장 외곽의 값들이 true 인 height, width 값의 true-false map을 만들어 return.
	'''
	tf_map = [[False for _ in range(width)] for _ in range(height)]
	for coord in coords:
		for c in coord:
			tf_map[c[1]][c[0]] = True
	if border:
		for x in range(width):
			tf_map[0][x] = True
			tf_map[-1][x] = True
		for y in range(height):
			tf_map[y][0] = True
			tf_map[y][-1] = True
	return tf_map

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

def divided_class_into_image(divided_class, class_number, class_color, width, height, printing_class):
	mosiac_image = np.zeros([height, width ,3], dtype=np.uint8)
	for h in range(height):
		for w in range(width):
			if divided_class[h][w] == 0 or divided_class[h][w] not in printing_class:
				mosiac_image[h][w] = [0, 0, 0]
			else:
				mosiac_image[h][w] = class_color[class_number.index(divided_class[h][w])]
	return mosiac_image

def divided_class_into_real_image(divided_class, real_image, width, height, printing_class):
	crop_image = np.zeros([height, width ,3], dtype=np.uint8)
	for h in range(height):
		for w in range(width):
			if divided_class[h][w] == 0 or divided_class[h][w] not in printing_class:
				crop_image[h][w] = [0, 0, 0]
			else:
				crop_image[h][w] = real_image[h][w]
	return crop_image

def get_masked_image(image, coord, width, height):
	'''
	get image`s only coord parts.
	'''
	crop_image = np.zeros([height, width, 3], dtype=np.uint8)
	for h in range(height):
		for w in range(width):
			if (h, w) in coord:
				crop_image[h][w] = image[h][w]
			else:
				crop_image[h][w] = [0, 0, 0]
	return crop_image

def get_class_crop_image(image, coord, width, height):
	x_min = width - 1
	x_max = 0
	y_min = height - 1
	y_max = 0
	
	for c in coord:
		if c[0] < x_min:
			x_min = c[0]
		if c[0] > x_max:
			x_max = c[0]
		
		if c[1] < y_min:
			y_min = c[1]
		if c[1] > y_max:
			y_max = c[1]
	
	crop_width = x_max - x_min
	crop_height = y_max - y_min
	crop_image = np.zeros([crop_height, crop_width, 3], dtype=np.uint8)
	for h in range(y_min, y_max + 1):
		for w in range(x_min, x_max + 1):
			if (h, w) in coord:
				crop_image[h - y_min][w - x_min] = image[h][w]
			else:
				crop_image[h - y_min][w - x_min] = [0, 0, 0]
	return crop_image

# Real Utility.
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

def get_class_with_given_coord(class_total, given_coord):
	ret_class_total = []
	for coord in given_coord:
		for cn in range(len(class_total)):
			if coord in class_total[cn] and coord not in ret_class_total:
				ret_class_total += class_total[cn]
	
	return ret_class_total

def can_go(x, y, width, height, direction=None, x_diff=False, y_diff=False):
	'''
	주어진 범위 밖으로 나가는지 체크
	x , y : 시작 좌표
	width, height: 가로와 세로의 길이
	direction : 방향 index of [동, 서, 남, 북]
	x_diff, y_diff : 만약 특정 길이만큼 이동시, 범위 밖인지 체크하고 싶을 때.
	'''
	if direction == None:        
		x_check = x + x_diff > -1 and x + x_diff < width
		y_check = y + y_diff > -1 and y + y_diff < height
	else:
		x_check = x + dir_x[direction] > -1 and x + dir_x[direction] < width
		y_check = y + dir_y[direction] > -1 and y + dir_y[direction] < height
	return x_check and y_check

if __name__ == "__main__":
	p1 = [205, 203, 202] # #CDCBCA ( 의자 등받이 부분 )
	p2 = [217, 217, 219] # #D9D9DB ( 의자 바닥 부분 )

	class_color = [[205, 203, 202], [90, 92, 105], [217, 217, 219], [172, 166, 153], [194, 181, 160], [92, 63, 43]]
	print(get_color_distance_map(class_color, len(class_color)))
	print(get_color_distance_map(class_color, len(class_color), distance_func=get_rgb_distance))
	