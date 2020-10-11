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

if __name__ == "__main__":
	content_image_name = "./Image/chair1.jpg"
	style_image_name = "./Image/lether_texture.jpg"
	stylized_image = set_style(content_image_name, style_image_name)
	stylized_image_with_list = np.array((stylized_image * 255)[0], np.uint8)
	stylized_image_with_list = cv2.cvtColor(stylized_image_with_list, cv2.COLOR_BGR2RGB)
	
	divided_class, class_total, class_border, class_count, class_length, largest_mask, width, height = segmentation.get_divided_class("./Image/chair1.jpg", "chair1_masked.jpg")

	# 일정 크기보다 작은 면적들은 근처에 뭐가 제일 많은지 체크해서 통합시킨다.
	class_number, class_total, class_border, class_count, class_length = \
	segmentation.merge_small_size(divided_class, list(range(1, class_length + 1)), class_total, class_border, class_count, width, height, min_value=120)
	
	class_number, class_total, class_border, class_count, class_length, class_color = \
	segmentation.merge_same_color(divided_class, class_number, class_total, class_border, class_count, largest_mask, width, height, sim_score=60)
	
	# 등받이만 있는 것 가져오기.
	ret_class_total	= utility.get_class_with_given_coord(class_total, [(503, 64)])
	printing_class = utility.calc_space_with_given_coord(class_number, class_total,[(503, 64)])
	part_change_image = image_processing.add_up_image(largest_mask, stylized_image_with_list, ret_class_total, width, height)

	utility.print_image(part_change_image)
	