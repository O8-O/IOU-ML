from tensorflow.python.ops.gen_math_ops import div
from tensorflow.python.ops.math_ops import divide
import styler
import segmentation
import image_processing
import utility
import objectDetctor

def segment(inputFile, outputFile, outputDataFile) :
	'''
	입력받은 파일을 Segmentation 해서 output한다.
	Output 한 결과는 조각난 사진 모음.
	'''
	divided_class, class_number, class_total, class_border, _, _, class_color, _, width, height = \
	segmentation.get_divided_class(inputFile)
	utility.save_result([divided_class, class_number, class_total, class_border], outputDataFile)
	
	dc_image = utility.divided_class_into_image(divided_class, class_number, class_color, width, height, class_number)
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
	styler.change_area_color(inputFile, outputFileName, destColor, divided_class, destArea)

def textureTransfer(inputFile, inputDataFile, outputFileName, destTexture, destCoordList)  :
	'''
	입력받은 inputFile의 정해진 부분( destCoordList )의 질감을 destTexture로 변경한다.
	'''

def styleTransfer(inputFile, inputDataFile, destFile) :
	'''
	입력받은 inputFile의 색과 질감을 destFile의 색과 질감으로 임의로 변형해준다. 
	'''

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

def analysisFurnitureParameter(inputFile, outputFile) :
	'''
	입력받은 inputFile의 가구 Parameter를 저장한다.
	'''

def analysisInteriorParameter(inputFile, outputFile) :
	'''
	입력받은 inputFile의 인테리어 Parameter를 저장한다.
	'''

if __name__ == "__main__":
	fileName = "Image/chair1.jpg"
	fileCheckName = "Image/chair1.bin"
	color_one_point = "Image/chair1-onePoint.jpg"
	color_multi_point = "Image/chair1-multiPoint.jpg"
	outputFile = "Image/chair1-divided.jpg"
	# segment(fileName, outputFile, fileCheckName)
	# colorTransferToCoord(fileName, fileCheckName, color_one_point, [255, 157, 65], [(503, 64)])
	colorTransferToColor(fileName, fileCheckName, color_multi_point, [255, 157, 65], [207, 205, 200])
