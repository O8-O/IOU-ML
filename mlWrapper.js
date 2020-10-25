const { PythonShell } = require("python-shell");

class MlWrapper {
	constructor() { }

	runner(options) {
		return new Promise((res, rej) => {
			PythonShell.run("mlWrapper.py", options, (err, data) => {
				if(err != null)	 rej(err);
				else  res(data);
			})
		});
	}

	segmentation(inputFile, outputFile, outputDataFile) {
		//입력받은 파일을 Segmentation 해서 output한다. Output 한 결과는 조각난 사진 모음.
		var options = {args : ["segment", inputFile, outputFile, outputDataFile]};
		return this.runner(options);
	}

	colorTransferToCoord(inputFile, inputDataFile, outputFileName, destColor, destCoordList) {
		// 입력받은 inputFile의 정해진 부분( destCoordList )의 색을 destColor로 변경한다.
		var options = {args : ["colorTransferToCoord", inputFile, inputDataFile, outputFileName, destColor, destCoordList]};
		return this.runner(options);
	}

	colorTransferToColor(inputFile, inputDataFile, destColor, srcColor) {
		// 입력받은 inputFile의 정해진 부분( srcColor와 비슷한 부분 )의 색을 destColor로 변경한다.
		var options = {args : ["colorTransferToColor", inputFile, inputDataFile, destColor, srcColor]};
		return this.runner(options);
	}

	colorTransferWithImage(inputFile, inputDataFile, outputFileName, destImage) {
		/*
		입력받은 inputFile의 색을 destImage와 비슷하게 변경해서 outputFileName에 저장한다.
		Segmentation이 된다면 자른 부분만 변경.
		*/
		var options = {args : ["colorTransferWithImage", inputFile, inputDataFile, outputFileName, destImage]};
		return this.runner(options);
	}

	textureTransferToCoord(inputFile, inputDataFile, outputFileName, destTexture, destCoordList) {
		// 입력받은 inputFile의 정해진 부분( destCoordList )의 질감을 destTexture로 변경한다.
		var options = {args : ["textureTransfer", inputFile, inputDataFile, outputFileName, destTexture, destCoordList]};
		return this.runner(options);
	}

	textureTransferArea(inputFile, inputDataFile, outputFileName, destTexture, srcColor) {
		// 입력받은 inputFile의 정해진 부분( srcColor와 비슷한 색 )의 질감을 destTexture로 변경한다.
		var options = {args : ["textureTransfer", inputFile, inputDataFile, outputFileName, destTexture, srcColor]};
		return this.runner(options);
	}

	styleTransfer(inputFile, inputDataFile, destFile) {
		// 입력받은 inputFile의 색과 질감을 destFile의 색과 질감으로 임의로 변형해준다. 
		var options = {args : ["styleTransfer", inputFile, inputDataFile, destFile]};
		return this.runner(options);
	}

	objectDect(inputFile, outputFile) {
		// 입력받은 inputFile의 가구를 ObjectDetection한 결과를 outputFile에 저장한다. json 형태로 저장한다. 현재는 bin file로만 입출력이 가능.
		var options = {args : ["objectDect", inputFile, outputFile, outputDataFile]};
		return this.runner(options);
	}

	analysisFurnitureParameter(inputFile, outputFile) {
		// Not Implemented.
		var options = {args : ["analysisFurnitureParameter", inputFile, outputFile, outputDataFile]};
		return this.runner(options);
	}

	analysisInteriorParameter(inputFile, outputFile) {
		// Not Implemented.
		var options = {args : ["analysisInteriorParameter", inputFile, outputFile, outputDataFile]};
		return this.runner(options);
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
ml.segmentation(fileName, outputFile, fileCheckName).then(
	(data)=> {
		console.log("Success!")
	},
	(err)=> {
		console.log("Fail!")
	}
);