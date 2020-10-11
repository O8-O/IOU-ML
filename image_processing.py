import cv2

def get_contours(frame, start=190, diff=30):
	'''
	외곽선과 그 외곽선의 그 계층관계를 Return ( contours, heirachy )
	frame = cv2.imread 값.
	'''
	# Converting the image to grayscale.
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Histogram Normalization
	gray_CLAHE = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(16, 16)).apply(gray)
	gray_filtered = cv2.bilateralFilter(gray_CLAHE, 7, 50, 50)

	# Using the Canny filter to get contours
	edges_high_thresh = cv2.Canny(gray_filtered, start, start + diff)

	# Using Contours, search Contours.
	return cv2.findContours(edges_high_thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

def contours_to_coord(contours):
	'''
	입력받은 외곽선들을 좌표계로 바꾸어서 return 해 준다.
	contours : 2D List. 각 원소들은 외곽선의 sequence 이다.
	'''
	coords = []
	for cinstances in contours:
		temp = []
		for c in cinstances:
				now = (c[0][0], c[0][1])
				if now not in temp:
					temp.append(now)
		coords.append(temp)
	return coords

def delete_line_threshold(contours, line_n=40):
	'''
	입력받은 외곽선 중, line_n 갯수를 넘지 않는 작은 line들은 지워준다.
	'''
	contour_len = list(map(len, contours))

	ret_contours = []

	for i in range(0, len(contours)):
		if contour_len[i] > line_n:
			ret_contours.append(contours[i])
	return ret_contours
