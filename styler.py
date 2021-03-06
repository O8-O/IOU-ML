import numpy as np
from numpy.lib import utils
import tensorflow as tf
from tensorflow.python import util
import tensorflow_hub as hub
import cv2

import utility
import image_processing
import matrix_processing
from utility import print_image

def set_style(content_image_name, style_image_name):
	content_image = utility.load_image(content_image_name)
	style_image = utility.load_image(style_image_name, style_image=True)
	style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

	# Load TF-Hub module.
	hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
	hub_module = hub.load(hub_handle)
	outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
	return outputs[0]

def set_color_with_color(content_image_name, stlye_color, a=5, b=1, change_style="median", light_color=[255, 255, 255], ratio=(1.0, 1.0)):
	# Style Color need to be RGB Color.
	img = cv2.imread(content_image_name)
	img = utility.resize_image(img, ratio=ratio)
	(height, width, _) = img.shape
	styled_image = np.zeros(img.shape, dtype=np.uint8)

	grayscale_input = cv2.imread(content_image_name, cv2.IMREAD_GRAYSCALE)
	grayscale_input = utility.resize_2darr(grayscale_input, ratio=ratio)
	stlye_color = lighter(stlye_color, limit=270)

	for h in range(height):
		for  w in range(width):
			light_power = (grayscale_input[h][w] / 255) ** 3
			for i in range(3):
				color_value = img[h][w][i] - int(light_color[i] * light_power / 2)
				color_value = color_value + int(stlye_color[i] * light_power)
				styled_image[h][w][i] = color_value if color_value < 255 else 255

	return styled_image

def set_color_with_image(input_file, color_file, mask_map, decrease_ratio=(0.1, 0.1)):
	source = utility.read_image(input_file)
	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
	(h, w, _) = source.shape
	source = cv2.resize(source, None, fx=decrease_ratio[0], fy=decrease_ratio[1], interpolation=cv2.INTER_AREA)

	target = utility.read_image(color_file)
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
	(h, w, _) = target.shape
	target = cv2.resize(target, None, fx=decrease_ratio[0], fy=decrease_ratio[1], interpolation=cv2.INTER_AREA)
	
	s_mean, s_std = image_processing.get_mean_and_std(source)
	t_mean, t_std = image_processing.get_mean_and_std(target)

	# input의 평균과 표준편차를 사용해서 output 색을 조절.
	height, width, channel = source.shape
	for h in range(height):
		for w in range(width):
			for c in range(channel):
				x = source[h, w, c]
				x = ((x - s_mean[c]) * (t_std[c] / s_std[c])) + t_mean[c]

				source[h, w, c] = utility.check_bound(round(x))

	original_image = utility.read_image(input_file)
	(h, w, _) = original_image.shape
	original_image = cv2.resize(original_image, None, fx=decrease_ratio[0], fy=decrease_ratio[1], interpolation=cv2.INTER_AREA)
	
	all_class_total = []
	if mask_map == None:
		for h in range(len(original_image)):
			for w in range(len(original_image[0])):
				all_class_total.append((w, h))
	else:
		for h in range(len(mask_map)):
			for w in range(len(mask_map[0])):
				if mask_map[h][w]:
					all_class_total.append((w, h))
				
	source = cv2.cvtColor(source, cv2.COLOR_LAB2BGR)

	part_change_image = image_processing.add_up_image(original_image, source, all_class_total, width, height)
	return part_change_image

def change_dest_color(input_file, output_file, setting_color, divided_class, class_total, touch_list, touch_hint=None, a=5, b=1, change_style="median", save_flag=True, ratio=(1.0, 1.0)):
	colored_image = set_color_with_color(input_file, setting_color, a=a, b=b, change_style=change_style, ratio=ratio)

	if touch_hint == None:
		ret_class_total	= utility.get_class_with_given_coord(class_total, touch_list)
	else:
		ret_class_total = class_total[touch_hint]
	original_image = utility.read_image(input_file)
	original_image = utility.resize_image(original_image, ratio=ratio)
	(height, width, _) = original_image.shape

	# Change ret_class_total`s part with colored image.
	part_change_image = image_processing.add_up_image(original_image, colored_image, ret_class_total, width, height)
	if save_flag:
		utility.save_image(part_change_image, output_file)
	return part_change_image

def change_dest_texture(input_file, output_file, texture_file, divided_class, class_total, touch_list):
	stylized_image = set_style(input_file, texture_file)
	stylized_image = np.array((stylized_image * 255)[0], np.uint8)
	stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_BGR2RGB)
	
	ret_class_total	= utility.get_class_with_given_coord(class_total, touch_list)
	original_image = utility.read_image(input_file)
	(height, width, _) = original_image.shape

	# Change ret_class_total`s part with colored image.
	part_change_image = image_processing.add_up_image(original_image, stylized_image, ret_class_total, width, height)
	utility.save_image(part_change_image, output_file)

def change_area_color(input_file, output_file, setting_color, divided_class, area, a=5, b=1, change_style="median"):
	colored_image = set_color_with_color(input_file, setting_color, a=a, b=b, change_style=change_style)

	original_image = utility.read_image(input_file)
	(height, width, _) = original_image.shape

	# Change ret_class_total`s part with colored image.
	return image_processing.add_up_image(original_image, colored_image, area, width, height)

def change_area_color_multi(input_file, output_file, setting_color, divided_class, area, a=5, b=1, change_style="median"):
	colored_image = []
	for i in range(len(area)):
		colored_image.append(set_color_with_color(input_file, setting_color[i], a=a, b=b, change_style=change_style))
	original_image = utility.read_image(input_file)
	(height, width, _) = original_image.shape
	for i in range(len(area)):
		original_image = image_processing.add_up_image(original_image, colored_image[i], area[i], width, height)

	# Change ret_class_total`s part with colored image.
	return original_image

def change_area_style(input_file, output_file, texture_file, area):
	stylized_image = set_style(input_file, texture_file)
	stylized_image = np.array((stylized_image * 255)[0], np.uint8)
	stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_BGR2RGB)

	original_image = utility.read_image(input_file)
	(height, width, _) = original_image.shape

	# Change ret_class_total`s part with colored image.
	part_change_image = image_processing.add_up_image(original_image, stylized_image, area, width, height)
	utility.save_image(part_change_image, output_file)

def get_similar_color_area(divided_class, class_number, class_total, class_color, dest_color, sim_threshold):
	'''
	주어진 dest_color와 비슷한 색을 가진 class 좌표들을 모아서 return 한다.
	'''
	class_length = len(class_number)
	width = len(divided_class[0])
	height = len(divided_class)

	ret_class_total = []
	for i in range(class_length):
		color_distance = utility.get_cielab_distance(class_color[i], dest_color)
		if class_number[i] != 0 and color_distance < sim_threshold:
			ret_class_total += class_total[i]
	
	return ret_class_total

def get_similar_color_area_adjac(divided_class, class_number, class_total, class_border, class_color, dest_color, sim_threshold):
	'''
	주어진 dest_color와 비슷한 색을 가진 인접 class 좌표들을 모아서 return 한다.
	'''
	class_length = len(class_number)
	class_adjac = [[False for _ in range(class_length)] for _ in range(class_length)]
	height = len(divided_class)
	width = len(divided_class[0])

	for i in range(class_length):
		ret_class_number = matrix_processing.find_adjac_class_number(divided_class, class_border[i], width, height)
		del ret_class_number[ret_class_number.index(class_number[i])]
		if 0 in ret_class_number:
			del ret_class_number[ret_class_number.index(0)]
		for j in range(len(ret_class_number)):
			class_adjac[i][class_number.index(ret_class_number[j])] = True
			class_adjac[class_number.index(ret_class_number[j])][i] = True

	similar_candidate = []
	min_color_distance = utility.INT_MAX
	base_index = 0

	for i in range(class_length):
		color_distance = utility.get_cielab_distance(class_color[i], dest_color)
		if color_distance < sim_threshold:
			similar_candidate.append(i)
		if color_distance < min_color_distance:
			min_color_distance = color_distance
			base_index = i
	
	return_class = [base_index]

	for i in similar_candidate:
		if i == base_index:
			continue
		for j in range(len(return_class)):
			if class_adjac[i][return_class[j]]:
				if i not in return_class:
					return_class.append(i)
	
	return_class_total = []

	for i in range(len(return_class)):
		if class_total[i][0] not in return_class_total:
			return_class_total += class_total[i]
	
	return return_class_total

def turn_off_light(input_file, light_color):
	original_input = cv2.imread(input_file)
	grayscale_input = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
	(height, width, _) = original_input.shape
	output_image = np.zeros((height, width, 3), dtype=np.uint8)

	for h in range(height):
		for  w in range(width):
			light_power = (grayscale_input[h][w] / 255) ** 3
			for i in range(3):
				if original_input[h][w][i] < int(light_color[i] * light_power):
					color_value = original_input[h][w][i] - int(light_color[i] * light_power / 2)
				else:
					color_value = original_input[h][w][i] - int(light_color[i] * light_power)
				output_image[h][w][i] = color_value if color_value > 0 else 0
	
	return output_image

def turn_on_light(input_file, turn_off_picture, light_color):
	grayscale_input = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
	(height, width) = grayscale_input.shape
	output_image = np.zeros((height, width, 3), dtype=np.uint8)
	lighter_color = lighter(light_color)

	for h in range(height):
		for  w in range(width):
			light_power = (grayscale_input[h][w] / 255) ** 3
			for i in range(3):
				color_value = turn_off_picture[h][w][i] + int(lighter_color[i] * light_power)
				output_image[h][w][i] = color_value if color_value < 255 else 255
	
	return output_image

def lighter(color, limit=680):
	add_all = 0
	ret_color = [0, 0, 0]
	for i in range(3):
		ret_color[i] = color[i]
		add_all += ret_color[i]
	
	r_ratio = (255 - color[0]) / 100
	g_ratio = (255 - color[1]) / 100
	b_ratio = (255 - color[2]) / 100

	while add_all < limit:
		ret_color[0] += r_ratio
		ret_color[1] += g_ratio
		ret_color[2] += b_ratio
		add_all = 0
		for i in range(3):
			add_all += ret_color[i]

	return ret_color

def get_light_change(inputFile, baseLight, changeLight):
	turn_off_picture = turn_off_light(inputFile, baseLight)
	turn_on_picture = turn_on_light(inputFile, turn_off_picture, changeLight)
	return turn_on_picture

if __name__ == "__main__":
	file_name = "Image/example/interior1.jpg"
	turn_off_picture = turn_off_light(file_name, [255, 255, 255])
	# turn_on_picture = turn_on_light(file_name, turn_off_picture, [255, 157, 65])
	turn_on_picture = turn_on_light(file_name, turn_off_picture, [178, 220, 240])
	print_image(turn_on_picture)