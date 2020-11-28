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
					if(result.length == 0) { res([]); }
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
	}
})
.catch((err) => {
	console.log(err);
});