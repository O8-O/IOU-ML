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
	output_image = np.zeros([height, width, 3], dtype=np.uint8)
	for h in range(height):
		for w in range(width):
			output_image[h][w] = original_image[h][w]
	for coord in add_coord:
		output_image[coord[1]][coord[0]] = add_image[coord[1]][coord[0]]
	return output_image

def get_dominant_color(image, clusters=10):
	'''
	get image 2d np.array and get dominant color with clusters number.
	'''
	#convert to rgb from bgr
	img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			
	#reshaping to a list of pixels
	img = img.reshape((img.shape[0] * img.shape[1], 3))
	
	#using k-means to cluster pixels
	kmeans = KMeans(n_clusters = clusters)
	kmeans.fit(img)
	
	#the cluster centers are our dominant colors.
	colors = kmeans.cluster_centers_
	
	return colors.astype(int).tolist()

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
