from tensorflow.python import util
import styler
import segmentation
import image_processing
import utility
import objectDetector
import random
import cv2
import sys
import numpy as np

from utility import coord_to_image

MAX_OUT_IMAGE = 8
MAX_CHANGE_COLOR = 3

def segment(inputFile, outputFile, outputDataFile) :
	'''
	입력받은 파일을 Segmentation 해서 output한다.
	Output 한 결과는 조각난 사진 모음.
	'''
	divided_class, class_number, class_total, class_border, _, _, class_color, _, width, height = \
	segmentation.get_divided_class(inputFile)
	utility.save_result([divided_class, class_number, class_total, class_border], outputDataFile)
	
	dc_image = utility.divided_class_into_image(divided_class, class_number, class_color, width, height, class_number)
	if not outputFile == None:
		utility.save_image(dc_image, outputFile)
	return divided_class, class_number, class_total, class_border

def colorTransferToCoord(inputFile, inputDataFile, outputFileName, destColor, destCoordList) :
	'''
	입력받은 inputFile의 정해진 부분( destCoordList )의 색을 destColor로 변경한다.
	'''
	if utility.is_exist(inputDataFile):
		[divided_class, _, class_total, _] = utility.load_result(inputDataFile)
	else:
		divided_class, _, class_total, _, _, _, _, _, _, _ = \
		segmentation.get_divided_class(inputFile)
	styler.change_dest_color(inputFile, outputFileName, destColor, divided_class, class_total, destCoordList)

def colorTransferToColor(inputFile, inputDataFile, outputFileName, destColor, srcColor) :
	'''
	입력받은 inputFile의 정해진 부분( srcColor와 비슷한 부분 )의 색을 destColor로 변경한다.
	'''
	if utility.is_exist(inputDataFile):
		[divided_class, class_number, class_total, class_border] = \
		utility.load_result(inputDataFile)
		class_count = []
		for ct in class_total:
			class_count.append(len(ct))
	else:
		divided_class, class_number, class_total, class_border, class_count, class_length, class_color, _, _, _ = \
		segmentation.get_divided_class(inputFile)
	
	class_color = image_processing.get_class_color(utility.read_image(inputFile), class_total, class_count)

	destArea = styler.get_similar_color_area(divided_class, class_number, class_total, class_color, srcColor, 240) # Simmilar Color threshold to 200.
	part_change_image = styler.change_area_color(inputFile, outputFileName, destColor, divided_class, destArea)
	utility.save_image(part_change_image, outputFileName)

def colorTransferWithImage(inputFile, inputDataFile, outputFileName, destImage):
	'''
	입력받은 inputFile의 색을 destImage와 비슷하게 변경해서 outputFileName에 저장한다.
	Segmentation이 된다면 자른 부분만 변경.
	'''
	if utility.is_exist(inputDataFile):
		[_, _, class_total, _] = \
		utility.load_result(inputDataFile)
		class_count = []
		for ct in class_total:
			class_count.append(len(ct))
	else:
		_, _, class_total, _, class_count, _, _, _, _, _ = \
		segmentation.get_divided_class(inputFile)
	
	_, _, mask_map, (width, height) = segmentation.get_segmented_image(inputFile)
	changed_image = styler.set_color_with_image(inputFile, destImage, mask_map)
	utility.save_image(changed_image, outputFileName)

def textureTransferToCoord(inputFile, inputDataFile, outputFileName, destTexture, destCoordList)  :
	'''
	입력받은 inputFile의 정해진 부분( destCoordList )의 질감을 destTexture로 변경한다.
	'''
	if utility.is_exist(inputDataFile):
		[divided_class, _, class_total, _] = utility.load_result(inputDataFile)
	else:
		divided_class, _, class_total, _, _, _, _, _, _, _ = \
		segmentation.get_divided_class(inputFile)
	styler.change_dest_texture(inputFile, outputFileName, destTexture, divided_class, class_total, destCoordList)

def textureTransferArea(inputFile, inputDataFile, outputFileName, destTexture, srcColor):
	'''
	입력받은 inputFile의 정해진 부분( srcColor와 비슷한 색 )의 질감을 destTexture로 변경한다.
	'''
	if utility.is_exist(inputDataFile):
		[divided_class, class_number, class_total, _] = \
		utility.load_result(inputDataFile)
		class_count = []
		for ct in class_total:
			class_count.append(len(ct))
	else:
		divided_class, class_number, class_total, _, class_count, _, class_color, _, _, _ = \
		segmentation.get_divided_class(inputFile)
	
	class_color = image_processing.get_class_color(utility.read_image(inputFile), class_total, class_count)

	destArea = styler.get_similar_color_area(divided_class, class_number, class_total, class_color, srcColor, 240) # Simmilar Color threshold to 200.
	styler.change_area_style(inputFile, outputFileName, destTexture, destArea)

def styleTransfer(inputFile, inputDataFile, destFile) :
	'''
	입력받은 inputFile의 색과 질감을 destFile의 색과 질감으로 임의로 변형해준다. 
	'''
	if utility.is_exist(inputDataFile):
		[divided_class, class_number, class_total, class_border] = \
		utility.load_result(inputDataFile)
		class_count = []
		for ct in class_total:
			class_count.append(len(ct))
	else:
		divided_class, class_number, class_total, class_border, class_count, _, class_color, _, _, _ = \
		segmentation.get_divided_class(inputFile)
	
	# Init Variables.
	largest_mask, _, _, (width, height) = segmentation.get_segmented_image(inputFile)
	class_color = image_processing.get_class_color(utility.read_image(inputFile), class_total, class_count)

	file_extension = "." + inputFile.split(".")[1]
	file_base_name = inputFile.split(".")[0]

	segdata = utility.add_name(inputFile, "_segmented")
	utility.save_image(largest_mask, segdata)

	# Get Cutted File`s color.
	input_color = getDominantColor(segdata)
	if len(input_color) < MAX_CHANGE_COLOR:
		input_color *= int(MAX_CHANGE_COLOR // len(input_color)) + 1 

	dest_color = getDominantColor(destFile)
	if len(dest_color) < MAX_CHANGE_COLOR:
		dest_color *= int(MAX_CHANGE_COLOR // len(dest_color)) + 1 
	temp = []
	for i in range(len(dest_color)):
		temp.append(utility.cut_saturation(dest_color[i]))
	dest_color = temp

	for i in range(MAX_OUT_IMAGE):
		next_file_name = file_base_name + "_" + str(i) + file_extension
		now_input_color = random.sample(input_color, MAX_CHANGE_COLOR)
		now_dest_color = random.sample(dest_color, MAX_CHANGE_COLOR)
		destArea = []
		for j in range(MAX_CHANGE_COLOR):
			destArea.append(styler.get_similar_color_area(divided_class, class_number, class_total, class_color, now_input_color[j], 200))
		part_change_image = styler.change_area_color_multi(inputFile, next_file_name, now_dest_color, divided_class, destArea, change_style="grayscale")
		utility.print_image(part_change_image)

def getFurnitureShape(inputFile, inputDataFile, outputFile):
	'''
	입력받은 inputFile과 그 분석 파일 inputDataFile을 통해 grayscale 및 segmentation 된 데이터를 만든다.
	만든 데이터는 outputFile로 ( Grayscale 사진 ) Output.
	'''
	if utility.is_exist(inputDataFile):
		[divided_class, _, class_total, _] = utility.load_result(inputDataFile)
	else:
		segment(inputFile, None, inputDataFile)
		[divided_class, _, class_total, _] = utility.load_result(inputDataFile)

	gray_image = image_processing.to_gray_scale(inputFile)
	utility.print_image(gray_image)
	utility.save_image(gray_image, outputFile)

def objectDetect(inputFile, outputFile) :
	'''
	입력받은 inputFile의 가구를 ObjectDetection한 결과를 outputFile에 저장한다. json 형태로 저장한다.
	현재는 bin file로만 입출력이 가능.
	폴더를 입력하면 outputFile은 무시됨.
	'''
	# Model name 1 mean dataset`s folder 1.
	model_name = '1'
	detection_model = objectDetector.load_model(model_name)
	if "." not in inputFile:
		# File is directory
		files = utility.get_filenames(inputFile)
		for f in files:
			if "." not in f:
				continue

			coord, str_tag, number_tag, score = objectDetector.inference(detection_model, f)
			# Save file name make.
			save_file_name = utility.add_name(f, "_od", extension="bin")
			dirs = save_file_name.split("/")
			save_image_name = ""
			for d in dirs[0:-1]:
				save_image_name += d + "/"
			save_image_name += f.split("/")[-1].split(".")[0] + "/"
			utility.make_dir(save_image_name)
			rect_files = []

			additional_infor = []
			for i in range(len(str_tag)):
				additional_infor.append(-1)
				rect_image = image_processing.get_rect_image(f, int(coord[i][0]), int(coord[i][1]), int(coord[i][2]), int(coord[i][3]))
				rect_image_name = save_image_name + f.split("/")[-1]
				rect_image_name = utility.add_name(rect_image_name, "_" + str(i))
				rect_files.append(rect_image_name)
				utility.save_image(rect_image, rect_image_name)
			utility.save_result([coord, str_tag, number_tag, score, rect_files, additional_infor], save_file_name)
			
	else:
		coord, str_tag, number_tag, score = objectDetector.inference(detection_model, inputFile)
		# Save file name make.
		save_file_name = utility.add_name(inputFile, "_od", extension="bin")
		dirs = save_file_name.split("/")
		save_image_name = ""
		for d in dirs[0:-1]:
			save_image_name += d + "/"
		save_image_name += inputFile.split("/")[-1].split(".")[0] + "/"
		utility.make_dir(save_image_name)
		rect_files = []
		additional_infor = []
		for i in range(len(str_tag)):
			additional_infor.append(-1)
			rect_image = image_processing.get_rect_image(inputFile, int(coord[i][0]), int(coord[i][1]), int(coord[i][2]), int(coord[i][3]))
			rect_image_name = save_image_name + inputFile.split("/")[-1]
			rect_image_name = utility.add_name(rect_image_name, "_" + str(i))
			rect_files.append(rect_image_name)
			utility.save_image(rect_image, rect_image_name)
		utility.save_result([coord, str_tag, number_tag, score, rect_files, additional_infor], outputFile)

def readResultData(outputFile):
	'''
	Object Detection 한 output file을 읽어서 사용 가능한 형태로 return.
	'''
	[coord, str_tag, number_tag, score, filenames] = utility.load_result(outputFile)
	return coord, str_tag

def getDominantColor(inputFile, remarkableThreshold=150):
	colors = image_processing.get_dominant_color(inputFile)
	return utility.get_remarkable_color(colors, remarkableThreshold)

def analysisFurnitureParameter(inputFile, outputFile) :
	'''
	입력받은 inputFile의 가구 Parameter를 저장한다.
	'''

def analysisInteriorParameter(inputFile, outputFile) :
	'''
	입력받은 inputFile의 인테리어 Parameter를 저장한다.
	'''

def option_check(option, parameter_length):
	if len(option) < parameter_length:
		raise Exception("OptionError : OPTIONS_TOO_SMALL");

def change_hex_color_to_bgr(hex_color):
	if len(hex_color) < 6:
		raise Exception("OptionError : HEX_COLOR_TOO_SHORT");
	r = format(hex_color[0:2], 'x')
	g = format(hex_color[2:4], 'x')
	b = format(hex_color[4:], 'x')

	return [b, g, r]

def change_str_to_coord(coord_str):
	'''
	coord_str need to be form with (a,b)
	'''
	if "," not in coord_str and ( "(" == coord_str[0] and ")" == coord_to_image[-1]):
		raise Exception("OptionError : COORD_FORMAT_IS_NOT_FORMATTABLE");
	[a, b] = coord_to_image[1:-1].split(",")
	return (a, b)

def getStyleChangedImage(inputFile, preferenceImages, tempdata="temp"):
	'''
	inputFile에 대한 preferenceImages 를 출력. 
	print 함수로 각 변환한 사진의 이름을 출력하고, 마지막에 몇 장을 줄것인지 출력한다.
	1. Object Detect 결과로 나온 가구들을 Segmentation 화 한다.
	2. 사용자가 좋아한다고 고른 인테리어의 가구 + 사용자가 좋아할것 같은 단독 가구의 색 / 재질 / Segment 를 가져온다.
	3. 원래의 인테리어의 가구들에 적절하게 배치한다.
		3-1. 원래의 인테리어 가구의 재질과 색을 변경한다. ( 모든 sofa, chair 에 대해서 ) -> 40%
		3-2. 원래의 인테리어 가구를 사용자가 좋아할만한 가구로 변경한다. ( 모든 sofa, chair에 대해서 color filter 적용한걸로 ) -> 40%
		3-3. 원래의 인테리어에서 색상 filter만 입혀준다. ( 위의 0.2 부분 )
	'''
	outputFile = utility.get_add_dir(inputFile, tempdata)
	# fav_furniture_list = "Image/InteriorImage/test_furniture/sofa"
	# fav_furniture_list = utility.get_filenames(fav_furniture_list)
	# 기존 Data 출력.
	[coord, str_tag, number_tag, score, rect_files, additional_infor, n_color] = utility.get_od_data(inputFile)
	'''
	segment_data = []
	for f in rect_files:
		segment_data.append(utility.get_segment_data(f))
	fav_furniture_seg_data = []
	for f in fav_furniture_list:
		fav_furniture_seg_data.append(utility.get_segment_data(f))
	'''
	for i in range(MAX_OUT_IMAGE):
		now_index = random.randint(0, len(preferenceImages) - 1)
		saveOutputFile = utility.add_name(outputFile, "_" + str(i))
		if i < MAX_OUT_IMAGE * 0.2:
			original_image = utility.read_image(inputFile)
			decrese_ratio = (1.0, 1.0)
			if original_image.shape[0] * original_image.shape[1] > 600 * 800:
				decrese_ratio = (0.3, 0.3)
			changed_image = styler.set_color_with_image(inputFile, preferenceImages[now_index], mask_map=None)
			utility.save_image(changed_image, saveOutputFile)
		elif i < MAX_OUT_IMAGE * 1.0:
			original_image = utility.read_image(inputFile)
			# 특정 크기 이상이면 decrease ratio를 조절하여 1/3으로..
			decrese_ratio = (1.0, 1.0)
			if original_image.shape[0] * original_image.shape[1] > 600 * 800:
				decrese_ratio = (0.3, 0.3)
				original_image = cv2.resize(original_image, None, fx=decrese_ratio[0], fy=decrese_ratio[1], interpolation=cv2.INTER_AREA)
			for i in range(len(str_tag)):
				if ( str_tag[i] == "sofa" or str_tag[i] == "chair" ):
					styled_furniture = styler.set_color_with_image(rect_files[i], preferenceImages[now_index], None, decrese_ratio)
					original_image = image_processing.add_up_image_to(original_image, styled_furniture, \
						int(coord[i][0] * decrese_ratio[0]), int(coord[i][1] * decrese_ratio[0]), int(coord[i][2] * decrese_ratio[0]), int(coord[i][3] * decrese_ratio[0]))
			utility.save_image(original_image, saveOutputFile)
		else:
			original_image = utility.read_image(inputFile)
			for i in range(len(str_tag)):
				if ( str_tag[i] == "sofa" or str_tag[i] == "chair" ):
					stylized_image = styler.set_style(rect_files[i], preferenceImages[now_index])
					stylized_image = np.array((stylized_image * 255)[0], np.uint8)
					styled_furniture = cv2.cvtColor(stylized_image, cv2.COLOR_BGR2RGB)
					original_image = image_processing.add_up_image_to(original_image, styled_furniture, int(coord[i][0]), int(coord[i][1]), int(coord[i][2]), int(coord[i][3]))
			utility.save_image(original_image, saveOutputFile)
		print(saveOutputFile)
	print(MAX_OUT_IMAGE)

if __name__ == "__main__":
	if len(sys.argv) == 1:
		exit()
	func = sys.argv[1]
	options = sys.argv[2:]
	if func == "segment":
		option_check(options, 3)
		segment(options[0], options[1], options[2])
	elif func == "colorTransferToCoord":
		option_check(options, 5)
		bgr_color = change_hex_color_to_bgr(options[3])
		coord = []
		for str_coord in options[4:]:
			coord.append(change_str_to_coord(str_coord))
		colorTransferToCoord(options[0], options[1], options[2], bgr_color, coord)
	elif func == "colorTransferToColor":
		option_check(options, 5)
		bgr_color_des = change_hex_color_to_bgr(options[3])
		bgr_color_src = change_hex_color_to_bgr(options[4])
		colorTransferToColor(options[0], options[1], options[2], bgr_color_des, bgr_color_src)
	elif func == "colorTransferWithImage":
		option_check(options, 4)
		colorTransferWithImage(options[0], options[1], options[2], options[3])
	elif func == "textureTransferToCoord":
		option_check(options, 5)
		coord = []
		for str_coord in options[4:]:
			coord.append(change_str_to_coord(str_coord))
		textureTransferToCoord(options[0], options[1], options[2], options[3], coord)
	elif func == "textureTransferArea":
		option_check(options, 5)
		bgr_color = change_hex_color_to_bgr(options[4])
		textureTransferArea(options[0], options[1], options[2], options[3], bgr_color)
	elif func == "getFurnitureShape":
		option_check(options, 3)
		getFurnitureShape(options[0], options[1], options[2])
	elif func == "getDominantColor":
		option_check(options, 1)
		getDominantColor(options[0])
	elif func == "styleTransfer":
		option_check(options, 3)
		styleTransfer(options[0], options[1], options[2])
	elif func == "getStyleChangedImage":
		option_check(options, 2)
		# Check Whether Preference.
		label_data = utility.get_label_files()
		label = [0, 0, 0, 0]
		label_file = ["label0", "label1", "label2", "label3"]

		# 어느 라벨에 속하는지 검사.
		for i in range(1, len(options)):
			base_file = options[i].split("/")[-1]
			for i in range(len(label_data)):
				if base_file in label_data[i]:
					label[i] += 1
		
		max_index = 0
		for i in range(len(label)):
			if label[i] > label[max_index]:
				max_index = i

		# 해당 라벨에 속하는 file로 image를 처리.
		getStyleChangedImage(options[0], utility.get_filenames("Image/InteriorImage/test/" + label_file[max_index]))
