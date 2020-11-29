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

def addWallFloorParameter():
	[files, domColors, wallColors, floorColors] = utility.load_result(config.RESEARCH_BASE_FILE)
	wallFloorColorList = [["interior (84).jpg", [210, 207, 202], [143, 167, 197]],
	["interior (8).jpg",	[170, 172, 180],	[211, 213, 224]],
	["interior (70).jpg",	[16, 63, 241],	[190, 208, 216]],
	["interior (79).jpg",	[99, 111, 117],	[208, 219, 221]],
	["interior (80).jpg",	[158, 155, 150],	[154, 157, 165]],
	["interior (82).jpg",	[241, 238, 232],	[70, 98, 128]],
	["interior (83).jpg",	[162, 209, 226],	[189, 177, 169]],
	["interior (56).jpg",	[177, 191, 208],	[68, 66, 75]],
	["interior (40).jpg",	[160, 146, 140],	[192, 184, 207]],
	["interior (35).jpg",	[239, 238, 242],	[70, 100, 160]],
	["interior (34).jpg",	[198, 193, 194],	[75, 67, 68]],
	["interior (30).jpg",	[223, 224, 228],	[40, 65, 109]],
	["interior (29).jpg",	[255, 255, 255],	[255, 255, 255]],
	["interior (28).jpg",	[67, 81, 139],	[69, 91, 119]],
	["interior (13).jpg",	[53, 166, 198],	[48, 69, 77]],
	["interior (1).jpg",	[220, 219, 221],	[57, 90, 135]]]
	wallFiles = []
	for wfFile in wallFloorColorList:
		wallFiles.append(wfFile[0])
	for f in files:
		indx = wallFiles.index(os.path.basename(f))
		wallColors[indx] = wallFloorColorList[indx][1]
		floorColors[indx] = wallFloorColorList[indx][2]
	utility.save_result([files, domColors, wallColors, floorColors], config.RESEARCH_BASE_FILE)

def makeSofaUrl(sofa):
	return config.RESEARCH_BASE_DIR + "/furniture/sofa/" + sofa

def makeTableUrl(table):
	return config.RESEARCH_BASE_DIR + "/furniture/table/" + table

def addRecommandFiles():
	inputFileList = ["interior (84).jpg", "interior (8).jpg", "interior (70).jpg", "interior (79).jpg",\
		"interior (80).jpg", "interior (82).jpg", "interior (83).jpg", "interior (56).jpg", \
		"interior (40).jpg", "interior (35).jpg", "interior (34).jpg", "interior (30).jpg", \
		"interior (29).jpg", "interior (28).jpg", "interior (13).jpg",  "interior (1).jpg", ]
	recomandList = [
		[makeSofaUrl("s33.jpg"), makeSofaUrl("s25.jpg"), makeSofaUrl("s80.jpg"), makeSofaUrl("s21.jpg"), makeSofaUrl("s156.jpg"), makeTableUrl("34.jpg"), makeTableUrl("13.jpg")],
		[ ],
		[makeSofaUrl("s13.jpg"), makeSofaUrl("s17.jpg"), makeSofaUrl("s137.jpg"), makeSofaUrl("s136.jpg"), makeSofaUrl("s149.jpg"),\
			makeTableUrl("59.jpg"), makeTableUrl("10.jpg"), makeTableUrl("43.jpg"), makeTableUrl("44.jpg")],
		[makeSofaUrl("s22.jpg"), makeSofaUrl("s24.jpg"), makeSofaUrl("s33.jpg"), makeSofaUrl("s41.jpg"), makeSofaUrl("s42.jpg"),\
			makeTableUrl("18.jpg"), makeTableUrl("42.jpg"), makeTableUrl("45.jpg")],
		[makeSofaUrl("s73.jpg"), makeSofaUrl("s60.jpg"), makeSofaUrl("s58.jpg"), makeSofaUrl("s59.jpg"), makeSofaUrl("s132.jpg"), makeSofaUrl("s156.jpg"),\
			makeTableUrl("82.jpg"), makeTableUrl("83.jpg"), makeTableUrl("3.jpg")],
		[makeSofaUrl("s151.jpg"), makeSofaUrl("s152.jpg"), makeSofaUrl("s156.jpg"),\
			makeTableUrl("8.jpg"), makeTableUrl("45.jpg"), makeTableUrl("58.jpg")],
		[makeSofaUrl("s136.jpg"), makeSofaUrl("s137.jpg"), makeSofaUrl("s143.jpg"), makeSofaUrl("s144.jpg"), makeSofaUrl("s59.jpg"), makeSofaUrl("s69.jpg"),\
			makeTableUrl("9.jpg"), makeTableUrl("5.jpg"), makeTableUrl("16.jpg")],
		[makeSofaUrl("s41.jpg"), makeSofaUrl("s38.jpg"), makeSofaUrl("s150.jpg"), makeSofaUrl("s151.jpg"), makeSofaUrl("s13.jpg"),\
			makeTableUrl("5.jpg"), makeTableUrl("15.jpg"), makeTableUrl("51.jpg")],
		[makeTableUrl("52.jpg"), makeTableUrl("41.jpg"), makeTableUrl("64.jpg"), makeTableUrl("70.jpg"), makeTableUrl("66.jpg"), makeTableUrl("65.jpg")],
		[makeSofaUrl("s13.jpg"), makeSofaUrl("55.jpg"), makeSofaUrl("s48.jpg"), makeSofaUrl("s49.jpg"), makeSofaUrl("s12.jpg"),\
			makeTableUrl("59.jpg"), makeTableUrl("10.jpg"), makeTableUrl("43.jpg"), makeTableUrl("44.jpg")],
		[makeSofaUrl("s65.jpg"), makeSofaUrl("s73.jpg"), makeSofaUrl("s74.jpg"), makeSofaUrl("s149.jpg"), makeSofaUrl("s142.jpg"), makeSofaUrl("s66.jpg"),\
			makeTableUrl("89.jpg"), makeTableUrl("67.jpg"), makeTableUrl("55.jpg"), makeTableUrl("56.jpg")],
		[makeTableUrl("50.jpg"), makeTableUrl("66.jpg"), makeTableUrl("71.jpg"), makeTableUrl("88.jpg")],
		[makeSofaUrl("s133.jpg"), makeSofaUrl("s135.jpg"), makeSofaUrl("s139.jpg"), makeSofaUrl("s149.jpg"),\
			makeTableUrl("43.jpg"), makeTableUrl("37.jpg"), makeTableUrl("59.jpg"), makeTableUrl("67.jpg")],
		[makeSofaUrl("s22.jpg"), makeSofaUrl("s26.jpg"), makeSofaUrl("s29.jpg"), makeSofaUrl("s11.jpg"), makeSofaUrl("s18.jpg"), makeSofaUrl("s19.jpg"),\
			makeTableUrl("4.jpg"), makeTableUrl("8.jpg"), makeTableUrl("5.jpg"), makeTableUrl("1.jpg")],
		[makeSofaUrl("s25.jpg"), makeSofaUrl("s26.jpg"), makeSofaUrl("s35.jpg"), makeSofaUrl("s36.jpg"), makeSofaUrl("s41.jpg"), makeSofaUrl("s43.jpg"),\
			makeTableUrl("23.jpg"), makeTableUrl("58.jpg"), makeTableUrl("88.jpg")],
		[makeSofaUrl("s18.jpg"), makeSofaUrl("s19.jpg"), makeSofaUrl("s16.jpg"), makeSofaUrl("s151.jpg"), makeSofaUrl("s154.jpg"), makeSofaUrl("s150.jpg"),\
			makeTableUrl("35.jpg"), makeTableUrl("2.jpg"), makeTableUrl("42.jpg")],
	]
	utility.save_result([inputFileList, recomandList], config.RECOMMAND_BASE_FILE)

if __name__ == "__main__":
	'''
	fileDir = "Image/Interior"
	labelDir = ["label0", "label1", "label2", "label3"]
	
	for ld in labelDir:
		utility.make_dir(fileDir + "/" + ld)
	'''

	#saveParameters("C:/workspace/IOU-ML/Image/InteriorImage/represent")
	# addWallFloorParameter()
	addRecommandFiles()
	[basePreferenceFiles, recommandFile] = utility.load_result(config.RECOMMAND_BASE_FILE)
	#parameter, files = readParameter(fileDir)
	#classifyAndMove(parameter, files, labelDir)
