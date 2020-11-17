from tensorflow.python import util
import styler
import segmentation
import image_processing
import matrix_processing
import utility
import objectDetector
import random
import cv2
import time
import numpy as np
import config
import imageClassifier
import os

from keras_segmentation.pretrained import pspnet_50_ADE_20K 

MAX_OUT_IMAGE = 8
MAX_CHANGE_COLOR = 3
MAX_WALL_IMAGE = 2
MAX_PART_CHANGE_IMAGE = 4
FILE_INQUEUE = "fileQueue.txt"
FILE_OUTQUEUE = "fileOutQueue.txt"
COLOR_SYSTEM_FILE = "colorSystem.bin"
RESEARCH_BASE_DIR = config.RESEARCH_BASE_DIR
functionList = ["getStyleChangedImage"]
detection_model = None

def segment(inputFile, outputFile, outputDataFile, total=False) :
	'''
	입력받은 파일을 Segmentation 해서 output한다.
	Output 한 결과는 조각난 사진 모음.
	'''
	divided_class, class_number, class_total, class_border, _, _, class_color, largest_mask, width, height = \
	segmentation.get_divided_class(inputFile, total=total)
	utility.save_result([divided_class, class_number, class_total, class_border, largest_mask], outputDataFile)
	
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
		loadData = utility.load_result(inputDataFile)
		if len(loadData) == 5:
			# Newer Version of segmentation.
			[divided_class, class_number, class_total, _, largest_mask] = loadData
		else:
			[divided_class, class_number, class_total, _] = loadData
			largest_mask = None
		class_count = []
		for ct in class_total:
			class_count.append(len(ct))
	else:
		divided_class, class_number, class_total, _, class_count, _, class_color, _, _, _ = \
		segmentation.get_divided_class(inputFile)
		
	# Init Variables. - TODO : Change this part with largest mask.
	# largest_mask, _, _, (width, height) = segmentation.get_segmented_image(inputFile)
	# class_color = image_processing.get_class_color(utility.read_image(inputFile), class_total, class_count)
	img = utility.read_image(inputFile)
	(height, width, _) = img.shape

	file_extension = "." + inputFile.split(".")[1]
	file_base_name = inputFile.split(".")[0]
	
	input_sample = [class_total[i][0] for i in range(len(class_total))]
	if len(input_sample) < MAX_CHANGE_COLOR:
		input_sample *= int(MAX_CHANGE_COLOR // len(input_sample)) + 1 
	dest_color = image_processing.get_dominant_color(destFile, clusters=8)

	for i in range(MAX_OUT_IMAGE):
		next_file_name = file_base_name + "_" + str(i) + file_extension
		now_input_sample = random.sample(input_sample, MAX_CHANGE_COLOR)
		now_dest_color = random.sample(dest_color, MAX_CHANGE_COLOR)
		part_change_image = utility.read_image(inputFile)
		for j in range(MAX_CHANGE_COLOR):
			change_image = styler.change_dest_color(inputFile, next_file_name, now_dest_color[j], divided_class, class_total, [now_input_sample[j]], save_flag=False)
			part_change_image = image_processing.add_up_image(part_change_image, change_image, class_total[input_sample.index(now_input_sample[j])], width, height)
	return part_change_image

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

def getStyleChangedImage_past(inputFile, preferenceImages, tempdata="temp"):
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
	if "\\" in inputFile:
		dirs = inputFile.split("\\")
		inputFile = ""
		for d in dirs[:-1]:
			inputFile += d + "/"
		inputFile += dirs[-1]
	outputFile = utility.get_add_dir(inputFile, tempdata)
	# fav_furniture_list = "Image/InteriorImage/test_furniture/sofa"
	# fav_furniture_list = utility.get_filenames(fav_furniture_list)
	# 기존 Data 출력.
	base_name = inputFile.split("/")[-1].split("Z")[-1]
	researched_files = utility.get_only_jpg_files("C:/workspace/IOU-Backend/util/IOU-ML/Image/InteriorImage/test")
	checked_file = ""	
	for rf in researched_files:
		if base_name in rf:
			checked_file = rf
	
	[coord, str_tag, number_tag, score, rect_files, additional_infor, n_color] = utility.get_od_data(checked_file)
	'''
	segment_data = []
	for f in rect_files:
		segment_data.append(utility.get_segment_data(f))
	fav_furniture_seg_data = []
	for f in fav_furniture_list:
		fav_furniture_seg_data.append(utility.get_segment_data(f))
	'''
	returnImageList = []
	for i in range(MAX_OUT_IMAGE):
		now_index = random.randint(0, len(preferenceImages) - 1)
		saveOutputFile = utility.add_name(outputFile, "_" + str(i))
		if i < MAX_OUT_IMAGE * 0.2:
			original_image = utility.read_image(inputFile)
			decrese_ratio = (1.0, 1.0)
			if original_image.shape[0] * original_image.shape[1] > 1200 * 960:
				decrese_ratio = (0.3, 0.3)
			changed_image = styler.set_color_with_image(inputFile, preferenceImages[now_index], mask_map=None)
			utility.save_image(changed_image, saveOutputFile)
		elif i < MAX_OUT_IMAGE * 1.0:
			original_image = utility.read_image(inputFile)
			# 특정 크기 이상이면 decrease ratio를 조절하여 1/3으로..
			decrese_ratio = (1.0, 1.0)
			if original_image.shape[0] * original_image.shape[1] > 1200 * 960:
				decrese_ratio = (0.3, 0.3)
				original_image = cv2.resize(original_image, None, fx=decrese_ratio[0], fy=decrese_ratio[1], interpolation=cv2.INTER_AREA)
			for i in range(len(str_tag)):
				if ( str_tag[i] == "sofa" or str_tag[i] == "chair" ):
					styled_furniture = styler.set_color_with_image("C:/workspace/IOU-Backend/util/IOU-ML/" + rect_files[i], preferenceImages[now_index], None, decrese_ratio)
					original_image = image_processing.add_up_image_to(original_image, styled_furniture, \
						int(coord[i][0] * decrese_ratio[0]), int(coord[i][1] * decrese_ratio[0]), int(coord[i][2] * decrese_ratio[0]), int(coord[i][3] * decrese_ratio[0]))
			utility.save_image(original_image, saveOutputFile)
		else:
			original_image = utility.read_image(inputFile)
			for i in range(len(str_tag)):
				if ( str_tag[i] == "sofa" or str_tag[i] == "chair" ):
					stylized_image = styler.set_style("C:/workspace/IOU-Backend/util/IOU-ML/" + rect_files[i], preferenceImages[now_index])
					stylized_image = np.array((stylized_image * 255)[0], np.uint8)
					styled_furniture = cv2.cvtColor(stylized_image, cv2.COLOR_BGR2RGB)
					original_image = image_processing.add_up_image_to(original_image, styled_furniture, int(coord[i][0]), int(coord[i][1]), int(coord[i][2]), int(coord[i][3]))
			utility.save_image(original_image, saveOutputFile)
		returnImageList.append(saveOutputFile)
	returnImageList.append(MAX_OUT_IMAGE)
	return returnImageList

def getODandSegment(inputFile, od_model):
	# [coord, str_tag, number_tag, score, rect_files, additional_infor, n_color] = imageClassifier.saveParameter(inputFile, od_model) # Get OD Data
	[coord, str_tag, number_tag, score, rect_files, additional_infor, n_color] = \
	utility.load_result(config.RESEARCH_BASE_DIR + "/" + os.path.basename(utility.get_od_bin(inputFile)))
	for i in range(len(str_tag)):
		if str_tag[i] == "sofa" or str_tag[i] == "chair":
			if utility.is_exist(utility.get_userinput_bin(rect_files[i])):
				rect_data_file = utility.get_userinput_bin(rect_files[i])
			elif utility.is_exist(utility.get_bin(rect_files[i])):
				rect_data_file = utility.get_bin(rect_files[i])
			else:
				rect_data_file = utility.get_bin(rect_files[i])
				segment(rect_files[i], utility.add_name(rect_files[i], "_divided"), rect_data_file)
	return [coord, str_tag, number_tag, score, rect_files, additional_infor, n_color] 

def changeWallFloor(inputFile, outputFile, wall_divided, wall_total, wall_number, i, preferWallColor, preferFloorColor):
	wfOutputFile = utility.add_name(outputFile, "_wfColor" + str(i))
	print(preferWallColor)
	print(preferFloorColor)
	styler.change_dest_color(inputFile, wfOutputFile, preferWallColor[i], \
		wall_divided, wall_total, [wall_total[wall_number.index(segmentation.WALL_CLASS)][0]])
	styler.change_dest_color(wfOutputFile, wfOutputFile, preferFloorColor[i], \
		wall_divided, wall_total, [wall_total[wall_number.index(segmentation.FLOOR_CLASS)][0]])
	return wfOutputFile

def getPartChangedImage(inputFile, outputFile, str_tag, coord, rect_files, selectedPreferenceImage, i, j):
	partChangedOutFile = utility.add_name(outputFile, "_changed_" + str(i) + str(j))
	original_image = utility.read_image(inputFile)

	for k in range(len(str_tag)):
		if ( str_tag[k] == "sofa" or str_tag[k] == "chair" ):
			furniture_file = rect_files[k]
			# 만약 userinput 이 있다면, 그것을 대신 사용.
			if utility.is_exist(utility.get_userinput_bin(furniture_file)):
				furniture_data_file = utility.get_userinput_bin(furniture_file)
			else:
				furniture_data_file = utility.get_bin(furniture_file)
			styled_furniture = styleTransfer(furniture_file, furniture_data_file, selectedPreferenceImage)
			original_image = image_processing.add_up_image_to(original_image, styled_furniture, \
				int(coord[k][0]), int(coord[k][1]), int(coord[k][2]), int(coord[k][3]))

	utility.save_image(original_image, partChangedOutFile)
	return partChangedOutFile

def getStyleChangedImage(inputFile, preferenceImages, od_model, baseLight=[255,255,255], changeLight=[178, 220, 240]):
	'''
	입력 Color는 BGR ( [178, 220, 240] 은 주황불빛 )
	preferenceImages 가 4장만 되어도 충분함.
	'''
	detection_model = pspnet_50_ADE_20K()
	outputFile = utility.get_add_dir(inputFile, "temp")

	# Object Detect & Segmentation
	[coord, str_tag, number_tag, score, rect_files, additional_infor, n_color]  = getODandSegment(inputFile, od_model)
	print("Loading Finished")
	
	# Wall Detection with input image.
	wall_divided = segmentation.detect_wall_floor(inputFile, detection_model)
	wall_total, wall_number = matrix_processing.divided_class_into_class_total(wall_divided)
	print("Wall Divided.")
	
	# Get preference image`s data.
	preferWallColor = []
	preferFloorColor = []
	selectedPreferenceImages = []
	[files, domColors, wallColors, floorColors] = utility.load_result(config.RESEARCH_BASE_FILE)	# Each files` dom color, wall color, floor color will be saved.
	baseNameFiles = [os.path.basename(files[f]) for f in range(len(files))]

	print("Wall Color start.")
	# Select 2 color of above to preferWallColor and preferFloorColor
	for i in range(MAX_WALL_IMAGE):
		indx = random.randint(0, len(preferenceImages) - 1)
		preferImage = preferenceImages[indx]
		loadIndex = baseNameFiles.index(os.path.basename(preferImage))	# We do only compare with base name.
		preferWallColor.append(wallColors[loadIndex])
		preferFloorColor.append(floorColors[loadIndex])
		selectedPreferenceImages.append(files[loadIndex])
	
	print("Wall Colored Selected.")
	# Change wall & floor
	wfColorChangeImage = []
	for i in range(MAX_WALL_IMAGE):
		wfOutputFile = changeWallFloor(inputFile, outputFile, wall_divided, wall_total, wall_number, i, preferWallColor, preferFloorColor)
		wfColorChangeImage.append(wfOutputFile)
	print("Wall Color Changed")
	
	wfColorChangeImage = utility.get_filenames("Image/example/temp")
	# Change Object ( Table and Chair )
	partChangedFiles = []
	
	for i in range(MAX_WALL_IMAGE):
		for j in range(MAX_PART_CHANGE_IMAGE):
			print("now ", i * 2 + j)
			selectedPreferenceImage = selectedPreferenceImages[random.randint(0, len(selectedPreferenceImages) - 1)]
			partChangedOutFile = getPartChangedImage(wfColorChangeImage[i], outputFile, str_tag, coord, rect_files, selectedPreferenceImage, i, j)
			partChangedFiles.append(partChangedOutFile)
	
	print("Part Changed Finished")
	# Add some plant.
	# partChangedFiles = print() # Image number will not be changed.

	partChangedFiles = utility.get_filenames("Image/example/temp")
	# Change Light
	for i in range(MAX_OUT_IMAGE):
		print("Now Proceed : ", i)
		files = utility.add_name(partChangedFiles[i], "_lighter")
		if random.randint(1, MAX_OUT_IMAGE) > 4:
			changed_file = styler.get_light_change(partChangedFiles[i], baseLight, changeLight)
		else:
			changed_file = styler.get_light_change(partChangedFiles[i], baseLight, baseLight)
		utility.save_image(changed_file, files)
	# partChangedFiles 가 결국 바뀐 파일들

def get_color_system(directory):
	'''
	directory 내부에 있는 모든 Color system의 목록을 조사한다. Remarkable Color로 조사한다.
	조사한 결과는 pickle을 통해 directory에 저장해둔다.
	'''
	fileNames = utility.get_filenames(directory)
	colors = []
	baseName = []

	for f in fileNames:
		print(f, " now doing .. " , str(fileNames.index(f)))
		baseName.append(utility.get_base_name(f))
		colors.append(getDominantColor(f))
	
	utility.save_result([baseName, colors], RESEARCH_BASE_DIR + "/" + COLOR_SYSTEM_FILE)

def color_match(inputColor, colors, fileNames, admitable=30):
	result_list = []
	file_result = []
	index = 0
	for color in colors:
		for c in color:
			if utility.get_cielab_distance(inputColor, c) < admitable:
				file_result.append(fileNames[index])
				result_list.append(color)
				break
		index += 1
	return result_list, file_result

def image_color_match(inputImage):
	# 이미지와 어울리는 컬러 List의 List를 return.
	[fileNames, colors] = utility.load_result(RESEARCH_BASE_DIR + "/" + COLOR_SYSTEM_FILE)
	input_colors = getDominantColor(inputImage)
	admitableColors = []
	admitableFiles = []
	while len(admitableColors) == 0:
		admitable = 10
		for input_color in input_colors:
			res_color, files = color_match(input_color, colors, fileNames, admitable)
			index = 0
			for r in res_color:
				if r not in admitableColors:
					admitableColors.append(r)
					admitableFiles.append(files[index])
				index += 1
		admitable += 10
	print(len(colors))
	print(len(admitableColors))
	utility.print_image(utility.color_to_image(input_colors))
	
	return admitableColors, admitableFiles

def checkInput():
	with open(FILE_INQUEUE, 'r') as f:
		lines = f.readline()
	
	# 파일 비우기
	f = open(FILE_INQUEUE, 'w')
	f.close()

	global functionList
	functionIndex = 0
	i = 1
	while i < len(lines):
		while lines[i] not in functionList and i < len(lines):
			i += 1
		output = doJob(lines[functionIndex:i])
		# Write Output Data.
		if output != None:
			with open(FILE_OUTQUEUE, 'a') as f:
				f.write("__DIV__\n")
				for out in output:
					f.write(str(out) + "\n")
		
		functionIndex = i
		
def doJob(argv):			
	func = argv[0]
	options = argv[1:]
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

		if len(options) == 2 and len(options[1]) == 0:
			max_index = 0
		else:
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
		return getStyleChangedImage(options[0], utility.get_filenames("C:/workspace/IOU-Backend/util/IOU-ML/Image/InteriorImage/test/" + label_file[max_index]))

if __name__ == "__main__":
	'''
	test_image_directory = "C:/workspace/IOU-ML/Image/InteriorImage/test_only_image"
	testFile = utility.get_filenames(test_image_directory)
	# get_color_system(test_image_directory)
	print(testFile[0])
	admitableColor, admitableFiles = image_color_match(testFile[0])
	for ad in admitableColor:
		print(admitableFiles[admitableColor.index(ad)])
		utility.print_image(utility.color_to_image(ad))
	'''
	'''
	inputFile = "Image/Interior/interior7/interior7_0.jpg"
	inputDataFile = RESEARCH_BASE_DIR + '/' + utility.add_name(inputFile.split("/")[-1], "_userInput", extension="bin")
	destFile = "Image/example/Sofa/s1.jpg"
	styleTransfer(inputFile, inputDataFile, destFile)
	'''
	#model_name = '1'
	#od_model = objectDetector.load_model(model_name)
	#getStyleChangedImage("Image/example/interior7.jpg", ["interior (35).jpg", "interior (40).jpg", "interior (13).jpg"], od_model)
	getStyleChangedImage("Image/example/interior7.jpg", ["interior (35).jpg", "interior (40).jpg", "interior (13).jpg"], "")

	'''
	files = utility.get_filenames("C:/workspace/IOU-ML/Image/InteriorImage/test_furniture/total")
	index = 0
	for fileName in files:
		print("Now Process ", index, " / " , len(files))
		outputFile = RESEARCH_BASE_DIR + '/' + utility.add_name(fileName.split("/")[-1], "_divided")
		outputDataFile = RESEARCH_BASE_DIR + '/' + utility.add_name(fileName.split("/")[-1], "", extension="bin")
		segment(fileName, outputFile, outputDataFile)
		index += 1
	'''
	'''
	fileName = "Image/example/interior1.jpg"
	outputFile = RESEARCH_BASE_DIR + '/' + utility.add_name(fileName.split("/")[-1], "_divided")
	outputDataFile = RESEARCH_BASE_DIR + '/' + utility.add_name(fileName.split("/")[-1], "", extension="bin")
	segment(fileName, outputFile, outputDataFile, total=True)
	'''
	'''
	# Load ML Module and Read - Do ML Job
	# Model name 1 mean dataset`s folder 1.
	model_name = '1'
	detection_model = objectDetector.load_model(model_name)

	print("Load Module Finished. Now can scheduling.")
	# Scheduler for readFile.
	while True:
		nowTime = time.time()
		# 매 2초 혹은 7초마다 5초마다 검사한다.
		if nowTime % 10 == 2 or nowTime % 10 == 7:
			checkInput()
		else:
			time.sleep(1)	# 1초간 휴식
	'''