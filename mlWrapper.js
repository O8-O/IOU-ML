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
	segmentation(inputFile, outputFile) {

	}

	colorTransferToCoord(inputFile, destColor, setCoordList) {

	}

	colorTransferToColor(inputFile, destColor, srcColor) {

	}

	textureTransfer(inputFile, destTexture, setCoordList) {

	}

	styleTransfer(inputFile, destFile) {

	}

	objectDect(inputFile, outputFile) {

	}

	analysisFurnitureParameter(inputFile, outputFile) {

	}

	analysisInteriorParameter(inputFile, outputFile) {
		
	}
}