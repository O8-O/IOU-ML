import functools

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2

def load_image(image_path, style_image=False, preserve_aspect_ratio=True):
	"""Loads and preprocesses images."""
	# Cache image file locally.
	# Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
	img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
	if img.max() > 1.0:
		img = img / 255.
	if len(img.shape) == 3:
		img = tf.stack([img, img, img], axis=-1)

	if style_image:
		img = tf.image.resize(img, (256, 256), preserve_aspect_ratio=True)
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
	content_image_name = "../../Image/real_chair.jpg"
	style_image_name = "../../Image/lether_texture.jpg"
	stylized_image = set_style(content_image_name, style_image_name)
	content_image = load_image(content_image_name)
	style_image = load_image(style_image_name, style_image=True)
	print(stylized_image)
	# Visualize input images and the generated stylized image.
	show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])