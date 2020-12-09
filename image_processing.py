import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_only_instance_image(input_file, mask, width, height, show=False):
	'''
	파일 이름을 입력받아서 실제로 masking 된 부분의 사진만 output 해 준다.
	input_file	: string, 파일 이름
	masks		: 2차원 list, width / height 크기, True 면 그 자리에 객체가 있는것, 아니면 없음.
	width		: int, 너비
	height		: int, 높이
	output_file	: string, output_file 이름
	'''
	# Input file은 파일의 이름이나, np array로 들어올 수 있다.
	if type(input_file) == type("str"):
		original = cv2.imread(input_file)
	else:
		original = input_file
	masked_image = np.zeros([height, width ,3], dtype=np.uint8)
	mask_num = 0

	for h in range(0, height):
		for w in range(0, width):
				for c in range(0, 3):
					masked_image[h][w][c] = (original[h][w][c] if mask[h][w] else 0)
					if mask[h][w]:
						mask_num += 1

	return masked_image, mask_num

def get_total_instance_image(masks, width, height, base=True):
    	# Set masks total part base.
	mask = np.zeros((height, width))
	for h in range(height):
		for w in range(width):
			mask[h][w] = not base
			for m in masks:
				if m[h][w]:
					mask[h][w] = base
					break
	return mask

def divied_three_part(mask, width, height):
	part_coord = [[], [], [], []]
	divided_class = np.zeros((height, width), dtype=np.uint8)
	for h in range(0, int(height/3)):
		for w in range(width):
			if mask[h][w]:
				divided_class[h][w] = 1
				part_coord[1].append((w, h))
			else:
				part_coord[0].append((w, h))
				
		
	for h in range(int(height/3), int(height/3)*2):
		for w in range(width):
			if mask[h][w]:
				divided_class[h][w] = 2
				part_coord[2].append((w, h))
			else:
				part_coord[0].append((w, h))

	for h in range(int(height/3)*2, int(height/3)*3):
		for w in range(width):
			if mask[h][w]:
				divided_class[h][w] = 3
				part_coord[3].append((w, h))
			else:
				part_coord[0].append((w, h))
	
	return divided_class, part_coord

def get_contours(frame, clipLimit=16.0, tileGridSize=(16, 16), start=190, diff=30):
	'''
	외곽선과 그 외곽선의 그 계층관계를 Return ( contours, heirachy )
	frame = cv2.imread 값.
	'''
	# Converting the image to grayscale.
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Histogram Normalization
	gray_CLAHE = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize).apply(gray)
	gray_filtered = cv2.bilateralFilter(gray_CLAHE, 7, 50, 50)

	# Using the Canny filter to get contours
	edges_high_thresh = cv2.Canny(gray_filtered, start, start + diff)

	# Using Contours, search Contours.
	return cv2.findContours(edges_high_thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

def get_average_color(image, total_class, tc_num):
	'''
	image 의 주어진 tc_num개의 total_class에서의 평균 색을 구한다.
	'''
	aver_r = 0
	aver_g = 0
	aver_b = 0

	for coord in total_class:
		rgb = image[coord[1]][coord[0]]
		aver_r += rgb[0]
		aver_g += rgb[1]
		aver_b += rgb[2]
	
	return [int(aver_r/tc_num), int(aver_g/tc_num), int(aver_b/tc_num)]

def get_class_color(image, class_total, class_count, color_function=get_average_color):
	'''
	class total의 list에 있는 각각의 평균적인 색상을 image에서 구한다.
	color_function 으로 색을 구하는 방식도 지정 가능하다.
	'''
	class_color = []
	for i in range(len(class_total)):
		class_color.append(color_function(image, class_total[i], class_count[i]))
	return class_color

def add_up_image(original_image, add_image, add_coord, width, height):
	output_image = np.zeros(original_image.shape, dtype=np.uint8)
	(he, wi, _) = output_image.shape
	for h in range(he):
		for w in range(wi):
			output_image[h][w] = original_image[h][w]
	
	for coord in add_coord:
		try:
			if coord[0] < wi and coord[1] < he and coord[0] >= 0 and coord[1] >= 0:
				output_image[coord[1]][coord[0]] = add_image[coord[1]][coord[0]]
		except:
			continue
	return output_image

def get_dominant_color(image, clusters=20):
	'''
	get image 2d np.array and get dominant color with clusters number.
	'''
	img = cv2.imread(image)
			
	#reshaping to a list of pixels
	img = img.reshape((img.shape[0] * img.shape[1], 3))
	
	#using k-means to cluster pixels
	kmeans = KMeans(n_clusters = clusters)
	kmeans.fit(img)
	
	#the cluster centers are our dominant colors.
	colors = kmeans.cluster_centers_
	
	return colors.astype(int).tolist()

def add_up_image_to(image, add_image, min_x, max_x, min_y, max_y):
	(add_h, add_w, _) = add_image.shape
	(original_h, original_w, _) = image.shape
	width = max_x - min_x
	height = max_y - min_y

	if add_h * add_w > width * height:
		resize_add_image = cv2.resize(add_image, (width, height), interpolation=cv2.INTER_AREA)
	else:
		resize_add_image = cv2.resize(add_image, (width, height), interpolation=cv2.INTER_CUBIC)
	
	output_image = np.zeros(image.shape, dtype=np.uint8)
	for y in range(original_h):
		for x in range(original_w):
			output_image[y][x] = image[y][x]
	
	for y in range(height):
		for x in range(width):
			try:
				output_image[min_y + y][min_x + x] = resize_add_image[y][x]
			except:
				continue
	
	return output_image

# Styler
def get_mean_and_std(x):
	x_mean, x_std = cv2.meanStdDev(x)
	x_mean = np.hstack(np.around(x_mean, 2))
	x_std = np.hstack(np.around(x_std, 2))
	return x_mean, x_std

def get_gray_scale(color):
	'''
	입력 Color의 GrayScale 된 값을 Return.
	color : BGR Color.
	'''
	gray_color = 0
	gray_parameter = [0.1140, 0.5870, 0.2989]

	for i in range(3):
		gray_color += gray_parameter[i] * color[i]
	
	return gray_color

def blend_color(color1, color2, change_style="median", a=1, b=1):
	'''
	change_style 은 median, gray_scaler 가 있다.
	a 와 b 는 각각 color1 과 color2 의 가중치 비율. a 가 늘어나면 color1 이 늘어나고, 반대는 반대!
	'''
	b_color = np.zeros([3], dtype=np.uint8)
	if change_style == "median":
		# 두 Color의 평균을 사용하는 방법.
		for i in range(3):
			b_color[i] = int((color1[i] * a + color2[i] * b) / (a + b))
		return b_color
	else:
		# Grayscale을 사용하는 방법. 이 방법은 가중치를 사용하지 않는다.
		gray_value = get_gray_scale(color1)
		for i in range(3):
			b_color[i] = int(color2[i] * ( gray_value / 255 ))
	return b_color

def to_gray_scale(image):
	image_arr = cv2.imread(image)
	return cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)

def inpainting(imgFile, maskFile):
	# mask 파일은 지울 부분이 흰색으로 칠해진 원래 이미지와 비슷한 사진.
	img = imgFile
	mask = cv2.imread(maskFile, 0)	# cv2.IMREAD_GRAYSCALE

	next_picture = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
	return next_picture

# Object Detector
def get_rect_image(f, x_min, x_max, y_min, y_max):
	# 지정된 max to min의 사각형 이미지를 얻어낸다.
	width = x_max - x_min
	height = y_max - y_min
	image = cv2.imread(f)
	output_image = np.zeros([height, width, 3], dtype=np.uint8)
	
	for y in range(y_min, y_max):
		for x in range(x_min, x_max):
			output_image[y - y_min][x - x_min] = image[y][x]
	return output_image
