import numpy as np
import cv2

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

def color_transfer(input_file, color_file):
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

				source[h, w, c] = check_boundart(round(x))

	source = cv2.cvtColor(source, cv2.COLOR_LAB2BGR)
	return source

def check_boundart(val):
	ret_val = 0
	ret_val = 0 if val < 0 else val
	ret_val = 255 if val > 255 else val
	return ret_val

if __name__ == "__main__":
	colored_image = color_transfer("../../Image/chair1.jpg", "../../Image/color_style4.jpg")
	cv2.namedWindow("TEMP", cv2.WINDOW_NORMAL)
	cv2.imshow("TEMP", colored_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()