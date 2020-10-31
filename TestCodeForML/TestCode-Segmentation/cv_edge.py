# Importing OpenCV
from matplotlib.pyplot import contour
import cv2
import numpy as np


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


def connect_neareast_lines():
	print()


def find_nearest_point(tf_map, point, before, n=10):
	# Find Nearest Point at given point.
	print()


def connect_lines(tf_map, start_point, dest_point):
	start_x = start_point[0]
	start_y = start_point[1]
	dest_x = dest_point[0]
	dest_y = dest_point[1]


def delete_line_threshold(contours, line_n=40):
	contour_len = list(map(len, contours))

	ret_contours = []

	for i in range(0, len(contours)):
		if contour_len[i] > line_n:
			ret_contours.append(contours[i])
	return ret_contours


def contour_to_image(contours, width, height):
	contour_image = [[0 for _ in range(0, width)] for _ in range(0, height)]
	# Change Contours into images.
	for cinstances in contours:
		for c in cinstances:
			contour_image[c[0][1]][c[0][0]] = 255
	contour_image = np.array(contour_image, np.uint8)
	return contour_image


def coord_to_image(coordinates, width, height):
	coord_image = [[0 for _ in range(0, width)] for _ in range(0, height)]
	# Change Coordinates into images.
	for coord_list in coordinates:
		for c in coord_list:
			coord_image[c[1]][c[0]] = 255
	coord_image = np.array(coord_image, np.uint8)
	return coord_image


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
	elif len(imgs) % 2:
		rows = 2
		cols = len(imgs) // 2
	elif len(imgs) % 3:
		rows = 3
		cols = len(imgs) // 2
	else:
		rows = 2
		cols = len(imgs) // 2 + 1

	for i in range(0, len(imgs)):
		ax = fig.add_subplot(rows, cols, i+1)
		ax.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
		ax.axis("off")

	plt.show()

def contours_to_coord(contours):
	coords = []
	for cinstances in contours:
		temp = []
		for c in cinstances:
				now = (c[0][0], c[0][1])
				if now not in temp:
					temp.append(now)
		coords.append(temp)
	return coords


if __name__ == "__main__":
	# Open Image
	frame = cv2.imread('Image/chair1.jpg')
	height, width, _ = frame.shape

	# Converting the image to grayscale.
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Smoothing without removing edges.
	gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)

	# Histogram Normalization
	gray_CLAHE = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(16, 16)).apply(gray)
	gray_filtered = cv2.bilateralFilter(gray_CLAHE, 7, 50, 50)

	# Using the Canny filter to get contours
	diff = 30
	start = 190
	edges_high_thresh = cv2.Canny(gray_filtered, start, start + diff)
	closing_edges = cv2.morphologyEx(edges_high_thresh, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
	edges_filtered = cv2.Canny(closing_edges, 60, 120)

	# Using Contours, search Contours.
	contours, heirarchy = cv2.findContours(edges_high_thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
	coords = contours_to_coord(contours)
	coords = delete_line_threshold(coords, line_n=40)
	cycle_list = []
	noncycle_list = []
	for c in coords:
		cycled, noncycled = divide_cycle(c)
		if len(cycled) != 0:
			cycle_list += cycled
		if len(noncycled) != 0:
			noncycle_list += noncycled
			
	coord_image = coord_to_image(cycle_list, width, height)
	contour_image = contour_to_image(contours, width, height)
	show_with_plt([contour_image, coord_image])