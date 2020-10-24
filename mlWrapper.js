const { PythonShell } = require("python-shell");

let options = {
	scriptPath: "path/to/my/scripts",
	args: ["value1", "value2", "value3"]
};

PythonShell.run("my_script.py", options, function(err, data) {
	if (err) throw err;
	console.log(data);
});

class MlWrapper {
	segmentation(inputFile, outputFile, outputDataFile) {

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