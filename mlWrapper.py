from tensorflow.python import util
from tensorflow.python.ops.gen_math_ops import div
from tensorflow.python.ops.math_ops import divide
import styler
import segmentation
import image_processing
import utility
import objectDetctor
import random
import sys

from utility import coord_to_image

MAX_OUT_IMAGE = 5
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

	segdata = file_base_name + "_segmented" + file_extension
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

def objectDect(inputFile, outputFile) :
	'''
	입력받은 inputFile의 가구를 ObjectDetection한 결과를 outputFile에 저장한다. json 형태로 저장한다.
	현재는 bin file로만 입출력이 가능.
	'''
	# Model name 1 mean dataset`s folder 1.
	model_name = '1'
	detection_model = objectDetctor.load_model(model_name)
	coord, str_tag, number_tag, score = objectDetctor.inference(detection_model, inputFile)
	utility.save_result([coord, str_tag, number_tag, score], outputFile)

def readResultData(outputFile):
	'''
	Object Detection 한 output file을 읽어서 사용 가능한 형태로 return.
	'''
	[coord, str_tag, number_tag, score] = utility.load_result(outputFile)
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
		
if __name__ == "__main__":
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
	