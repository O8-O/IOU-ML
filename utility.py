import cv2
import numpy as np
import sys
import pickle
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil
import config

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

def cut_saturation(color, hsv_thres=100):
	array_color = np.array([[color]], dtype='uint8')
	hsv_color = cv2.cvtColor(array_color, cv2.COLOR_BGR2HSV).tolist()[0][0]
	
	if hsv_color[1] > hsv_thres:
		hsv_color[1] /= int(255 / hsv_thres)
	hsv_color[1] = int(hsv_color[1])
	
	hsv_color = np.array([[hsv_color]], dtype='uint8')
	return cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR).tolist()[0][0]

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

def get_remarkable_color(color_list, color_threshold, convert_rgb=False):
	color_length = len(color_list)
	lab_color_list = [cv2.cvtColor(np.array([[color_list[i]]], dtype="uint8"), cv2.COLOR_BGR2Lab).tolist()[0][0] for i in range(color_length)]

	color_distance = get_color_distance_map(lab_color_list, color_length)
	color_close = [[False for _ in range(color_length)] for _ in range(color_length)]
	
	for i in range(color_length):
		for j in range(color_length):
			if i != j and color_distance[i][j] < color_threshold:
				color_close[i][j] = True
				color_close[j][i] = True

	close_length = []
	for i in range(color_length):
		temp = 0
		for j in range(color_length):
			if i != j and color_close[i][j]:
				temp += 1
		close_length.append(temp)
	
	close_length_sorted = [(-1 * close_length[i], i) for i in range(color_length)]
	close_length_sorted.sort()

	selected_index = [False for _ in range(color_length)]
	result_color = []
	for i in range(color_length):
		if selected_index[i]:
			continue
		now_index = close_length_sorted[i][1]
		selected_index[now_index] = True
		result_color.append(color_list[now_index])
		for j in range(color_length):
			if color_close[now_index][j]:
				selected_index[j] = True
	if convert_rgb:
		result_color = [cv2.cvtColor(np.array([[result_color[i]]], dtype="uint8"), cv2.COLOR_BGR2RGB).tolist()[0][0] for i in range(len(result_color))]
	return result_color

def get_remarkable_color_n(color_list, n, convert_rgb=False):
	color_length = len(color_list)
	lab_color_list = [cv2.cvtColor(np.array([[color_list[i]]], dtype="uint8"), cv2.COLOR_BGR2Lab).tolist()[0][0] for i in range(color_length)]

	color_distance = get_color_distance_map(lab_color_list, color_length)
	color_far_map = [0 for _ in range(color_length)]
	
	for i in range(color_length):
		for j in range(color_length):
			color_far_map[i] += color_distance[i][j]
	
	sorted_color_far = []
	for i in range(color_length):
		sorted_color_far.append((color_far_map[i], i))
	sorted_color_far.sort()

	result_color = []
	for i in range(n):
		result_color.append(color_list[i])
		
	if convert_rgb:
		result_color = [cv2.cvtColor(np.array([[result_color[i]]], dtype="uint8"), cv2.COLOR_BGR2RGB).tolist()[0][0] for i in range(len(result_color))]
	return result_color

# 좌표 변환
def change_coord(coord, ratio=(0.5, 0.5)):
	# coord 좌표를 ratio 비율만큼 변화시킨다.
	return (int(coord[0] * ratio[0]), int(coord[1] * ratio[1]))

def change_coords(coords, ratio=(0.5, 0.5)):
	return [(int(coords[i][0] * ratio[0]), int(coords[i][1] * ratio[1])) for i in range(len(coords))]

def changed_coords2d(coords, ratio=(0.5, 0.5)):
	return [change_coords(coords[i], ratio=ratio) for i in range(len(coords))]

def change_arrcoords(coords, ratio=(0.5, 0.5)):
	resized_coords = []
	for i in range(len(coords)):
		resized_coords.append([])
		for j in range(len(coords[i])):
			# Order of x x y y
			if j < 2:
				resized_coords[i].append(coords[i][j] * ratio[0])
			else:
				resized_coords[i].append(coords[i][j] * ratio[1])
				
	return resized_coords

def get_relative_ratio(width, height, dstWidth, dstHeight):
	return (dstWidth / width, dstHeight / height)

def change_class_total_with_ratio(class_total, ratio=(0.5, 0.5)):
	ret_class_total = []
	for ct in class_total:
		change_ct = change_coord(ct, ratio)
		if change_ct not in ret_class_total:
			ret_class_total.append(change_ct)
	return ret_class_total

def resize_image(image, ratio=(0.5, 0.5)):
	(height, width, _) = image.shape
	retWidth = int(width * ratio[0])
	retHeight = int(height * ratio[1])
	retImage = np.zeros((retHeight, retWidth, 3), dtype=np.uint8)

	for h in range(retHeight):
		for w in range(retWidth):
			retImage[h][w] = get_exist_value(image, h, w, ratio)

	return retImage

def resize_2darr(image, ratio=(0.5, 0.5)):
	(height, width) = image.shape
	retWidth = int(width * ratio[0])
	retHeight = int(height * ratio[1])
	retImage = np.zeros((retHeight, retWidth), dtype=np.uint8)

	for h in range(retHeight):
		for w in range(retWidth):
			retImage[h][w] = get_exist_value(image, h, w, ratio)

	return retImage

def get_exist_value(src, h, w, ratio, func="leftTop"):
	exist_h = int(h * ( 1 / ratio[1]))
	exist_w = int(w * ( 1 / ratio[0]))

	if func=="average":
		exist_hn = int((h + 1) * ( 1 / ratio[1]))
		exist_wn = int((w + 1) * ( 1 / ratio[1]))
		exist_h = int((exist_h + exist_hn)/ 2)
		exist_w = int((exist_w + exist_wn)/ 2)

	return src[exist_h][exist_w]

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
			crop_image[h][w] = [0, 0, 0]
			
	for c in coord:
		crop_image[c[1]][c[0]] = image[c[1]][c[0]]
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

def color_to_image(colors, color_width=100, height=600):
	# Make colors into image which width is color_width
	color_image = np.zeros([height, color_width * len(colors), 3], dtype=np.uint8)
	for i in range(len(colors)):
		for h in range(height):
			for w in range(i * color_width, (i + 1) * color_width):
				color_image[h][w] = colors[i]
	return color_image

def divided_class_into_color_map(divided_class, width, height):
	import random
	color_list = []
	class_list = []
	output_image = np.zeros((height, width, 3), dtype=np.uint8)

	for h in range(height):
		for w in range(width):
			if divided_class[h][w] not in class_list:
				color_list.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
				class_list.append(divided_class[h][w])
			output_image[h][w] = color_list[class_list.index(divided_class[h][w])]
	
	return output_image

def resize_arr(arr, width, height):
	(origin_height, origin_width)= arr.shape
	output_arr = np.zeros((height, width))
	x_ratio = width / origin_width
	y_ratio = height / origin_height
	for h in range(height):
		for w in range(width):
			x = int(w / x_ratio)
			x = 0 if x < 0 else x
			x = origin_width - 1 if x == origin_width else x

			y = int(h / y_ratio)
			y = 0 if y < 0 else y
			y = origin_height - 1 if y == origin_height else y

			output_arr[h][w] = arr[y][x]
	return output_arr

def make_whitemask_image(image, masking_coord):
	ret_mask = np.zeros((image.shape), dtype=np.uint8)
	(height, width, _) = image.shape

	for h in range(height):
		for w in range(width):
			ret_mask[h][w] = [0, 0, 0]

	for coord in masking_coord:
		try:
			ret_mask[coord[1]][coord[0]] = [255, 255, 255] # Fill white mask.
		except:
			pass

	return ret_mask

# Object Detector
def tag_classifier(input_class):
	# Get only our interested feature.
	class_number = [15, 33, 44, 46, 47, 48, 49, 50, 51, 58, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 89]
	class_tag = ["bench", "suitcase", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "table", "chair", "couch", "potted plant", \
		"bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "microwave", "oven", "toaster", "sink", "refrigeratioor", "book", \
		"clock", "vase", "hair drier"]
	if input_class in class_number:
		return class_tag[class_number.index(input_class)]
	else:
		return None

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
	'''
	사용자가 입력한 좌표들로, 그 좌표가 우리가 가지고있는 class_total 좌표계 내에 존재할시. 그 class_total을 모아서 return 해 준다.
	'''
	ret_class_total = []
	for coord in given_coord:
		for cn in range(len(class_total)):
			if coord in class_total[cn] and coord not in ret_class_total:
				ret_class_total += class_total[cn]
				break
	
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

def check_bound(val):
	ret_val = 0
	ret_val = 0 if val < 0 else val
	ret_val = 255 if val > 255 else val
	return ret_val

# File I/O
def init_directory(same_image_dir, nonsame_image_dir):
    # Make basic directories.
    make_dir(same_image_dir)
    make_dir(nonsame_image_dir)

def get_filenames(working_dir):
    # Get files and add dirname front of files.
    files = os.listdir(working_dir)
    return_files = []
    for i in range(len(files)):
        if "." in files[i]:
            return_files.append(working_dir + "/" + files[i])
    return return_files

def load_image(image_path, style_image=False, preserve_aspect_ratio=True):
	"""
	Loads and preprocesses images.
	"""
	img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
	if img.max() > 1.0:
		img = img / 255.
	if len(img.shape) == 3:
		img = tf.stack([img, img, img], axis=-1)

	if style_image:
		img = tf.image.resize(img, (256, 256), preserve_aspect_ratio=preserve_aspect_ratio)
	return img

def make_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

def is_exist(file_name):
	return os.path.exists(file_name)

def save_result(result_list, file_name):
	# Save [divided_class, class_number, class_total, class_border]
	with open(file_name, 'wb') as f:
		pickle.dump(result_list, f)

def load_result(file_name):
	with open(file_name, 'rb') as f:
		retData = pickle.load(f)
	return retData

def read_image(image_file):
	img = cv2.imread(image_file)
	return img

def save_image(image_data, file_name):
	cv2.imwrite(file_name, image_data)

def add_name(input_file, addition, extension=None):
	dot_split = input_file.split(".")
	file_extension = ("." + dot_split[-1]) if extension == None else ("." + extension)
	file_base_name = ""
	if len(dot_split) > 2:
		for i in range(len(dot_split) - 1):
			file_base_name += dot_split[i] + "."
		file_base_name = file_base_name[:-1]
	else:
		file_base_name = input_file.split(".")[0] 
	return file_base_name + addition + file_extension

def get_add_dir(src, add_dir):
	'''
	src 는 디렉토리 + 파일 의 형태
	'''
	dest = ""

	dirs = src.split("/")
	for d in dirs[:-1]:
		dest += d + "/"
	
	dest += add_dir + "/"
	dest += dirs[-1]
	return dest

def move_into(image_file, to):
	shutil.copyfile(image_file, get_add_dir(image_file, to))

def get_od_data(interior_file):
	# coord, str_tag, number_tag, score, rect_files, additional_infor, n_color]
	od_file = add_name(interior_file, "_od", extension="bin")
	if is_exist(od_file):
		return load_result(od_file)
	else:
		return None

def get_segment_data(image):
	# [divided_class, class_number, class_total, class_border]
	sg_file = image.split(".")[0] + ".bin"
	if is_exist(sg_file):
		return load_result(sg_file)
	else:
		return None

def get_base_name(fileName):
	'''
	get file base name did not contain day or etc.
	ex ) /~/~/2020-10-27T15-21-32.684Zinterior7.jpg -> interior7
	ex ) /~/~/interior (73).jpg -> interior (73)
	'''
	if "-" in fileName.split("/")[-1]:
		return fileName.spilt("Z")[-1].split(".")[0]
	else:
		return fileName.split("/")[-1].split(".")[0]

def get_label_files(label_loc="C:/workspace/IOU-Backend/util/IOU-ML/Image/InteriorImage/test/"):
	label_data = []
	for label_folder in ["label0", "label1", "label2", "label3"]:
		files = get_filenames(label_loc + label_folder)
		label_data.append([])
		for f in files:
			label_data[-1].append(f.split("/")[-1])
			
	return label_data

def get_only_jpg_files(get_dir):
	files = get_filenames(get_dir)
	ret_dir = []
	for f in files:
		if ".jpg" in f:
			ret_dir.append(f)
	return ret_dir

def get_userinput_bin(inputFile):
	return config.RESEARCH_BASE_DIR + '/' + add_name(os.path.basename(inputFile), "_userInput", extension="bin")

def get_bin(inputFile):
	return config.RESEARCH_BASE_DIR + '/' + inputFile.split("/")[-1].split(".")[0] + ".bin"

def get_od_bin(inputFile):
	return config.RESEARCH_BASE_DIR + '/' + inputFile.split("/")[-1].split(".")[0] + "_od.bin"

def get_rc_bin(inputFile):
	return config.RESEARCH_BASE_DIR + '/' + inputFile.split("/")[-1].split(".")[0] + "_rc.bin"

def file_basify(inputFile, preferenceImages):
	# Input File : C:\\Users\\KDW\\Desktop\\KOO\\upload\\2020-11-21T06.625Zinterior (70).jpg
	# Test 당시 InputFile Image/example/interior7.jpg, preferenceImage "interior (84).jpg", "interior (40).jpg", "interior (82).jpg"
	if "Zinterior" in inputFile:
		inputBaseFile = inputFile.split("Z")[-1]
		preferenceBaseFile = [preferenceImages[i].split("Z")[-1] for i in range(len(preferenceImages))]
	else:
		inputBaseFile = inputFile
		preferenceBaseFile = preferenceImages
	return inputBaseFile, preferenceBaseFile

def logging(str_data):
	with open("logging.txt", 'a') as f:
		f.write(str_data + "\n")

def lightStringer(lightList):
	#ligthList need to be BGR sequence.
	lightString = "#"

	for i in range(2, -1, -1):
		lightString += str(format(lightList[i], 'x')) 	#BGR Sequence, reverse order.

	return lightString

def save_log_dictionary(inputFile, partChangedImage, changed_log):
	# changeLog 는 [wallColor, floorColor, wfColorImage, utility.lightStringer(baseLight[i]), 
	# changedFurnitureInfor, recommandFurnitureInfor] 형태의 list
	# changedFurnitureInfor 는 [cfLocInfor, cfColorInfor]
	
	return {
		"inputFile" : inputFile,
		"changedFile" : partChangedImage,
		"changedLog" : [make_changed_json(changed_log[i]) for i in range(len(changed_log))]
	}
	
def make_changed_json(changed_log):
	# changed_log[i] 가 입력.
	logging(str(changed_log))
	changedFd = []
	recommendFd = []
	for i in range(len(changed_log[4])):
		changedFd.append({
			"location": changed_log[4][0][i], 
			"color": changed_log[4][1][i]
		})
		recommendFd.append({
			"location": changed_log[4][0][i], 
			"link": changed_log[5][i]
		})
	return {
		"wallColor" : changed_log[0],
		"wallPicture" : changed_log[2],
		"floorColor" : changed_log[1],
		"floorPicture" : changed_log[2],
		"lightColor" : changed_log[3],
		"changedFurniture" : changedFd,
		"recommendFurniture" : recommendFd
	}
