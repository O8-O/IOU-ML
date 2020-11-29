const fs = require('fs');

ML_DATA = "C:/MLDATA/"
FILE_INQUEUE = ML_DATA + "fileQueue.txt";
FILE_OUTQUEUE = ML_DATA + "fileOutQueue.txt";

module.exports =  class MlWrapper {
	constructor() { }

	requestServiceStart(requestImage, preferenceImage, preferenceLight) {
		// Only can using getStyleChangedImage.
		console.log("Request Start.");
		var reqFunction = "getStyleChangedImage";	
		var reqString = reqFunction + "\n" + requestImage + "\n";
		if(preferenceLight == null) reqString += "255 255 255\n";
		else reqString += String(parseInt(preferenceLight[1] + preferenceLight[2], 16)) + " " + String(parseInt(preferenceLight[3] + preferenceLight[4], 16)) + " " + String(parseInt(preferenceLight[5] + preferenceLight[6], 16)) + "\n"
		for(var i = 0 ; i < preferenceImage.length; i++) {
			reqString += preferenceImage[i] + "\n"
		}
		fs.writeFile(FILE_INQUEUE, reqString, () => {});
	}

	checkServiceEnd() {
		return new Promise((res, rej) => {
			fs.readFile(FILE_OUTQUEUE, (err, data) => {
				if(err) rej(err);
				var result = String(data).split("\n");
				if(result.length == 0 || result[0].length == 0) { res([]); }
				else {
					var resultData = result[0].replace(/\//g, "\\\\");
					resultData = resultData.replace(/'/g, "\"");
					var changedData = JSON.parse(resultData);
					var changedList = [];
					changedList.push({changedFile: changedData.inputFile});

					for(var i = 0; i < changedData.changedFile.length; i++) {
						changedList.push({
							changedFile: changedData.changedFile[i],
							changedJson: changedData.changedLog[i]
						})
					}

					console.log(changedList);
					fs.writeFile(FILE_OUTQUEUE, "", () => {});
					res(changedList);
				}
			});
		});
	}
}

/*
// 사용 예시
const mlWrapper = require("./mlWrapper");

var mlCaller = new mlWrapper();
// If request, requestImage, preferenceImageList and preferenceLight. Image link need to be full-path. ( Not relative )
// Light might be RGB order.
mlCaller.requestServiceStart("example/file/to/link.jpg", ["prefer1.jpg", "prefer2.jpg", "prefer3.jpg"], [192, 234, 170]);

// ... or if no light color specification,
mlCaller.requestServiceStart("example/file/to/link.jpg", ["prefer1.jpg", "prefer2.jpg", "prefer3.jpg"], null);

// and you can check job end like this.
mlCaller.checkServiceEnd()
.then((changedList) => {
	if(changedList.length == 0) {
		// Job not end
	}
	else {
		console.log("hmm?");
	}
})
.catch((err) => {
	console.log(err);
});

changedList[0].changedFile = "원본 파일";
changedList[1].changedFile = "C:/바뀐/파일/이름1.jpg";
changedList[1].changedJson = "바뀐 json 정보";
changedList[1].changedJson = {
	wallColor : [233, 242, 172],
	wallPicture : "C:/무엇을/통해/색이/나왔는가.jpg",
	floorColor : [233, 242, 172],
	floorPicture : "C:/무엇을/통해/색이/나왔는가.jpg",
	lightColor : [255, 255, 255],
	changedFurniture : [
		{start : [234, 457], color : [233, 242, 172]},
		{start : [1023, 678], color : [233, 242, 172]},
	],
	recommandFurniture : [
		{start : [234, 457], pictureList : ["C:/recommand/file/path/filename1.jpg", "C:/recommand/file/path/filename2.jpg", "C:/recommand/file/path/filename3.jpg"]},
		{start : [234, 457], pictureList : ["C:/recommand/file/path/filename1.jpg", "C:/recommand/file/path/filename2.jpg", "C:/recommand/file/path/filename3.jpg"]},
	],
	recommandMore : [
		"C:/recommand/file/path/filename1.jpg", "C:/recommand/file/path/filename2.jpg", "C:/recommand/file/path/filename3.jpg"
	]
};
// 2 ~ 8 까지 반복
// Job end.	changedList is full-path file list which is modified.
 */