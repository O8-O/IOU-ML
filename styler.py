from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2

import utility
import segmentation
import image_processing

def load_image(image_path, style_image=False, preserve_aspect_ratio=True):
	"""
	Loads and preprocesses images.
	"""
	# Cache image file locally.
	# Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
	img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
	if img.max() > 1.0:
		img = img / 255.
	if len(img.shape) == 3:
		img = tf.stack([img, img, img], axis=-1)

	if style_image:
		img = tf.image.resize(img, (256, 256), preserve_aspect_ratio=preserve_aspect_ratio)
	return img

def show_n(images, titles=('',)):
	n = len(images)
	image_sizes = [image.shape[1] for image in images]
	w = (image_sizes[0] * 6) // 320
	plt.figure(figsize=(w  * n, w))
	gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
	for i in range(n):
		plt.subplot(gs[i])
		plt.imshow(images[i][0], aspect='equal')
		plt.axis('off')
		plt.title(titles[i] if len(titles) > i else '')
	plt.show()

def set_style(content_image_name, style_image_name):
	content_image = load_image(content_image_name)
	style_image = load_image(style_image_name, style_image=True)
	style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

	# Load TF-Hub module.
	hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
	hub_module = hub.load(hub_handle)
	outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
	return outputs[0]

def set_color_with_color(content_image_name, stlye_color, a=5, b=1, change_style="median"):
	img = cv2.imread(content_image_name)
	(height, width, _) = img.shape
	styled_image = np.zeros(img.shape, dtype=np.uint8)

	for h in range(height):
		for w in range(width):
			styled_image[h][w] = blend_color(img[h][w], stlye_color, a=a, b=b, change_style=change_style)

	return styled_image

def blend_color(color1, color2, change_style="median", a=1, b=1):
	'''
	입력 color는 BGR의 형태.
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

def read_file(input_file, source_file):
	s = cv2.imread(input_file)
	s = cv2.cvtColor(s, cv2.COLOR_BGR2LAB)
	t = cv2.imread(source_file)
	t = cv2.cvtColor(t, cv2.COLOR_BGR2LAB)
	return s, t

def get_mean_and_std(x):
	x_mean, x_std = cv2.meanStdDev(x)
	x_mean = np.hstack(np.around(x_mean, 2))
	x_std = np.hstack(np.around(x_std, 2))
	return x_mean, x_std

def set_color_with_image(input_file, color_file):
	source, target = read_file(input_file, color_file)
	s_mean, s_std = get_mean_and_std(source)
	t_mean, t_std = get_mean_and_std(target)

	# input의 평균과 표준편차를 사용해서 output 색을 조절.
	height, width, channel = source.shape
	for h in range(height):
		for w in range(width):
			for c in range(channel):
				x = source[h, w, c]
				x = ((x - s_mean[c]) * (t_std[c] / s_std[c])) + t_mean[c]

				source[h, w, c] = check_bound(round(x))

	source = cv2.cvtColor(source, cv2.COLOR_LAB2BGR)
	return source

def check_bound(val):
	ret_val = 0
	ret_val = 0 if val < 0 else val
	ret_val = 255 if val > 255 else val
	return ret_val

if __name__ == "__main__":
	content_image_name = "./Image/chair1.jpg"
	style_image_name = "./Image/lether_texture.jpg"
	divided_class, class_total, class_border, class_count, class_length, largest_mask, width, height = segmentation.get_divided_class(content_image_name, "")

	# 일정 크기보다 작은 면적들은 근처에 뭐가 제일 많은지 체크해서 통합시킨다.
	class_number, class_total, class_border, class_count, class_length = \
	segmentation.merge_small_size(divided_class, list(range(1, class_length + 1)), class_total, class_border, class_count, width, height, min_value=120)
	'''
	class_number, class_total, class_border, class_count, class_length, class_color = \
	segmentation.merge_same_color(divided_class, class_number, class_total, class_border, class_count, largest_mask, width, height, sim_score=10)
	'''
	stylized_image = set_style(content_image_name, style_image_name)
	stylized_image_with_list = np.array((stylized_image * 255)[0], np.uint8)
	stylized_image_with_list = cv2.cvtColor(stylized_image_with_list, cv2.COLOR_BGR2RGB)
	'''
	# 등받이만 있는 것 가져오기.
	ret_class_total	= utility.get_class_with_given_coord(class_total, [(503, 64)])
	printing_class = utility.calc_space_with_given_coord(class_number, class_total,[(503, 64)])
	part_change_image = image_processing.add_up_image(largest_mask, stylized_image_with_list, ret_class_total, width, height)
	'''
	'''
	# 색상 변경
	colored_image = set_color_with_color(content_image_name, [255, 157, 65], a=5, b=1, change_style="median")
	'''
	touch_list = [(523, 64), (491, 190), (352, 162), (318, 173), (301, 163), (264, 352), (255, 412), \
		(358, 136), (380, 129), (399, 137), (404, 166), (429, 154), (338, 354), (254, 411), (279, 216), \
		(265, 297), (271, 323), (285, 375), (253, 378), (250, 435), (245, 470), (236, 532), (140, 371), \
		(42, 367), (41, 299), (130, 293), (311, 251), (44, 211), (140, 138), (285, 151), (270, 176), (313, 225), (359, 196), (243, 333), (227, 333)]
	# 등받이만 있는 것 가져오기 - Chair.
	ret_class_total	= utility.get_class_with_given_coord(class_total, [(503, 64)])
	printing_class = utility.calc_space_with_given_coord(class_number, class_total, touch_list)
	part_change_image = image_processing.add_up_image(utility.divided_class_into_real_image(divided_class, largest_mask, width, height, printing_class)\
		, stylized_image_with_list, ret_class_total, width, height)
	utility.print_image(part_change_image)
	
	'''
	# 등받이만 있는 것 가져오기 - SOFA.
	ret_class_total	= utility.get_class_with_given_coord(class_total, [(185, 57), (252, 15), (46, 99)])
	printing_class = utility.calc_space_with_given_coord(class_number, class_total, [(185, 57), (252, 15), (46, 99)])
	part_change_image = image_processing.add_up_image(utility.divided_class_into_real_image(divided_class, largest_mask, width, height, printing_class), \
		colored_image, ret_class_total, width, height)
	utility.print_image(part_change_image)
	'''
	'''
	colored_image = set_color_with_image("../../Image/chair1.jpg", "../../Image/color_style4.jpg")
	cv2.namedWindow("TEMP", cv2.WINDOW_NORMAL)
	cv2.imshow("TEMP", colored_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''