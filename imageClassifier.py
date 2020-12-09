import objectDetector
import utility
import config
import image_processing
import os

MAX_COLOR_LENGTH = 5
MAX_ITEM_LENGTH = 15

def normParameter(str_tag, color_list):
	'''
	image의 str tag와 color_list를 받아서 normalized 된 결과를 return
	'''
	parameter = []
	for color in color_list:
		for i in range(3):
			parameter.append(color[i] / 255)

	if "potted plant" in str_tag:
    		parameter.append(1)
	else:
		parameter.append(0)

	total_item = 0
	item = ["vase", "clock", "book", "bowl", "cup"]
	for s in str_tag:
		if s in item:
			total_item += 1
	if total_item > MAX_ITEM_LENGTH:
		parameter.append(1)
	else:
		parameter.append(len(item)/MAX_ITEM_LENGTH)

	return parameter

def saveParameters(fileDir):
	# Model name 1 mean dataset`s folder 1.
	model_name = '1'
	detection_model = objectDetector.load_model(model_name)
	# File is directory
	files = utility.get_filenames(fileDir)
	fileNames = []
	domColors = [] 
	wallColors = []
	floorColors = []

	for f in files:
		if "." not in f:
			continue
		print("Now proceeding ", f , " [ ", files.index(f), " ]")

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

		dom_color = image_processing.get_dominant_color(f)
		n_color = utility.get_remarkable_color_n(dom_color, MAX_COLOR_LENGTH)
		fileNames.append(os.path.basename(f))
		domColors.append(n_color) 
		wallColors.append([])
		floorColors.append([])
		utility.save_result([coord, str_tag, number_tag, score, rect_files, additional_infor, n_color], save_file_name)
		
	utility.save_result([files, domColors, wallColors, floorColors],  config.RESEARCH_BASE_FILE)

def saveParameter(fileName, detection_model):
	coord, str_tag, number_tag, score = objectDetector.inference(detection_model, fileName)

	# Save file name make.
	save_file_name = config.RESEARCH_BASE_DIR + "/" + os.path.basename(utility.get_od_bin(fileName))
	dirs = save_file_name.split("/")

	save_image_name = ""
	for d in dirs[0:-1]:
		save_image_name += d + "/"
	save_image_name += fileName.split("/")[-1].split(".")[0] + "/"

	utility.make_dir(save_image_name)

	rect_files = []
	additional_infor = []

	for i in range(len(str_tag)):
		additional_infor.append(-1)
		rect_image = image_processing.get_rect_image(fileName, int(coord[i][0]), int(coord[i][1]), int(coord[i][2]), int(coord[i][3]))
		rect_image_name = save_image_name + fileName.split("/")[-1]
		rect_image_name = utility.add_name(rect_image_name, "_" + str(i))
		rect_files.append(rect_image_name)
		utility.save_image(rect_image, rect_image_name)

	dom_color = image_processing.get_dominant_color(fileName)
	n_color = utility.get_remarkable_color_n(dom_color, MAX_COLOR_LENGTH)
	utility.save_result([coord, str_tag, number_tag, score, rect_files, additional_infor, n_color], save_file_name)
	return [coord, str_tag, number_tag, score, rect_files, additional_infor, n_color]

def readParameter(fileDir):
	# File is directory
	files = utility.get_filenames(fileDir)
	
	bin_files = []
	for f in files:
		if ".bin" in f:
			bin_files.append(f)

	total_parameter = []
	total_files = []

	for bf in bin_files:
		[_, str_tag, _, _, _, _, n_color] = utility.load_result(bf)
		parameter = normParameter(str_tag, n_color)
		image_file = bf[:-7] + ".jpg"
		total_parameter.append(parameter)
		total_files.append(image_file)
	return total_parameter, total_files

def classifyAndMove(total_parameter, total_files, labelDir):
	labels = objectDetector.classifier(total_parameter, cluster_number=4)
	print(labels)
	for i in range(len(labels)):
		utility.move_into(total_files[i], labelDir[labels[i]])

import segmentation
def segment(inputFile, outputFile, outputDataFile, total=True):
	divided_class, class_number, class_total, class_border, _, _, class_color, largest_mask, width, height = \
	segmentation.get_divided_class(inputFile, total=total)
	utility.save_result([divided_class, class_number, class_total, class_border, largest_mask], outputDataFile)
	
	dc_image = utility.divided_class_into_image(divided_class, class_number, class_color, width, height, class_number)
	if not outputFile == None:
		utility.save_image(dc_image, outputFile)

def re_segmentation(fileDir, resegIndex):
	fileNames = utility.get_filenames(fileDir)
	for fIndex in range(len(fileNames)):
		f = fileNames[fIndex]
		odf = utility.get_od_bin(f)
		[coord, str_tag, number_tag, score, rect_files, additional_infor, n_color] = utility.load_result(odf)

		for i in resegIndex[fIndex]:
			rect_data_file = utility.get_bin(rect_files[i])
			print(rect_data_file, " will be re-generated. : ", str_tag[i])
			segment(rect_files[i], utility.add_name(rect_files[i], "_divided"), rect_data_file)

