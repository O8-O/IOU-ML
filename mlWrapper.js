const { PythonShell } = require("python-shell");

function runPythonCode(options, next) {
	PythonShell.run("pytest.py", options, (err, data) => {
		next(err, data);
	});
}

class MlWrapper {
	constructor() {

	}

	segmentation(inputFile, outputFile, outputDataFile, res, rej) {
		//입력받은 파일을 Segmentation 해서 output한다. Output 한 결과는 조각난 사진 모음.
		var options = ["segment", inputFile, outputFile, outputDataFile];
		return new Promise((res, rej) => {
			runPythonCode(options, (err, data) => {
				console.log(err, data);
			})
		});
	}

	colorTransferToCoord(inputFile, inputDataFile, outputFileName, destColor, destCoordList) {

	}

	colorTransferToColor(inputFile, inputDataFile, destColor, srcColor) {

	}

	textureTransfer(inputFile, inputDataFile, destTexture, destCoordList) {

	}

	styleTransfer(inputFile, inputDataFile, destFile) {

	}

	objectDect(inputFile, outputFile) {

	}

	analysisFurnitureParameter(inputFile, outputFile) {

	}

	analysisInteriorParameter(inputFile, outputFile) {
		
	}
}



var fileName = "Image/chair1.jpg"
var fileCheckName = "Image/chair1.bin"
var grayscale = "Image/chair1-gray.jpg"
var color_one_point = "Image/chair1-onePoint.jpg"
var color_multi_point = "Image/chair1-multiPoint.jpg"
var outputFile = "Image/chair1-divided.jpg"
var color_dest_image = "Image/interior2.jpg"
var color_change_with_image = "Image/chair1-image.jpg"
var texture_file = "Image/lether_texture.jpg"
var texture_one_point = "Image/Chair-texture-onePoint.jpg"
var texture_multi_point = "Image/Chair-texture-multiPoint.jpg"
var style_transfer_image = "Image/styles.jpg"

ml = new MlWrapper();
ml.segmentation(fileName, outputFile, fileCheckName,
	()=> {
		console.log("Success!")
	},
	()=> {
		console.log("Fail!")
	}
)