import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import pathlib

from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import utility
import image_processing

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model(model_name):
	# Get model with model name. If exists at cache, load it.
	base_url = 'http://download.tensorflow.org/models/object_detection/'
	model_file = model_name + '.tar.gz'
	model_dir = tf.keras.utils.get_file( fname=model_name, origin=base_url + model_file, untar=True, cache_dir="./")
	
	model_dir = pathlib.Path(model_dir)/"saved_model"
	model = tf.saved_model.load(str(model_dir))

	return model

def run_inference_for_single_image(model, image):
	image = np.asarray(image)
	# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
	input_tensor = tf.convert_to_tensor(image)
	# The model expects a batch of images, so add an axis with `tf.newaxis`.
	input_tensor = input_tensor[tf.newaxis,...]

	# Run inference
	model_fn = model.signatures['serving_default']
	output_dict = model_fn(input_tensor)

	# All outputs are batches tensors.
	# Convert to numpy arrays, and take index [0] to remove the batch dimension. We're only interested in the first num_detections.
	num_detections = int(output_dict.pop('num_detections'))
	output_dict = {key:value[0, :num_detections].numpy() 
					for key,value in output_dict.items()}
	output_dict['num_detections'] = num_detections

	# detection_classes should be ints.
	output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
	# Handle models with masks:
	if 'detection_masks' in output_dict:
		# Reframe the the bbox mask to the image size.
		detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks( output_dict['detection_masks'], output_dict['detection_boxes'], image.shape[0], image.shape[1])      
		detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
		output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

	return output_dict

def inference(model, image_path):
	# the array based representation of the image will be used later in order to prepare the
	# result image with boxes and labels on it.
	image_np = np.array(Image.open(image_path))
	# Actual detection.
	output_dict = run_inference_for_single_image(model, image_np)
	
	width = image_np.shape[1]
	height = image_np.shape[0]
	# Max, Min, Max, Min 순서. 각각 x 와 y 순서임.
	xy_coord = []
	for coords in output_dict['detection_boxes']:
		# x Min, x Max, y Min, y Max의 순서로 추가.
		xy_coord.append([coords[1] * width, coords[3] * width, coords[0] * height, coords[2] * height])
	
	score_100 = []
	for score in output_dict['detection_scores']:
		score_100.append(score * 100)
	number_tag = output_dict['detection_classes']

	coord, str_tag, score = get_score_cut_result(xy_coord, number_tag, score_100)

	return coord, str_tag, number_tag, score

def get_score_cut_result(coord, tag, score, score_threshold=40):
	return_coord = []
	return_tag = []
	return_score = []
	find_length = len(score)
	for s in range(find_length):
		if score[s] > score_threshold:
			if utility.tag_classifier(tag[s]) != None:
				return_coord.append(coord[s])
				return_tag.append(utility.tag_classifier(tag[s]))
				return_score.append(score[s])

	return return_coord, return_tag, return_score

def visualize_image(output_dict, image, category_index):
	# Image 정보가 오염되니, 중요한 Image는 넣기 전에 조심할것.
	# Visualization of the results of a detection.
	vis_util.visualize_boxes_and_labels_on_image_array(
		image,
		output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'],
		category_index, max_boxes_to_draw=30, min_score_thresh=0.4, use_normalized_coordinates=True, line_thickness=8)

	return image
