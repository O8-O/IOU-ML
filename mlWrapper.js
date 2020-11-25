const { PythonShell } = require("python-shell");
const fs = require('fs');

ML_DATA = "C:/MLDAT/"
FILE_INQUEUE = ML_DATA + "fileQueue.txt";
FILE_OUTQUEUE = ML_DATA + "fileOutQueue.txt";

module.exports =  class MlWrapper {
	constructor() { }

	requestServiceStart(requestImage, preferenceImage, preferenceLight) {
		// Only can using getStyleChangedImage.
		var reqFunction = "getStyleChangedImage";
		var reqString = reqFunction + "\n" + requestImage + "\n";
		if(preferenceLight == null) reqString += "255 255 255\n";
		else reqString += String(preferenceLight[0]) + " " + String(preferenceLight[1]) + " " + String(preferenceLight[2]) + "\n"
		for(var i = 0 ; i < preferenceImage.length; i++) {
			reqString += preferenceImage + "\n"
		}
		fs.writeFile(fileName, reqString, () => {});
	}

	checkServiceEnd() {
		return new Promise((res, rej) => {
			fs.readFile(FILE_OUTQUEUE, (err, data) => {
				if(err) rej(err);
				else {
					var result = data.split("\n");
					if(result.length == 0) { rej(); }
					else {
						var changedList = [];
						for(var i = 1; i < result.length; i++) {
							changedList.push(result[i]);
						}
						fs.writeFile(fileName, "", () => {});
						res(changedList);
					}
					
				}
			});
		});
	}
}

