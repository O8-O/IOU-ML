import mlWrapper
import utility
import sys
import config
import random

# To use, 2번째 인자에 리스트들을 담아서 넘겨주면 된다.
if __name__ == "__main__":
    [selectedPreferenceImages, wfColorChangeImage, outputFile, str_tag, coord, rect_files, i, j, ratio] = utility.load_result(sys.argv[1])
    selectedPreferenceImage = selectedPreferenceImages[random.randint(0, len(selectedPreferenceImages) - 1)]
    partChangedOutFile = mlWrapper.getPartChangedImage(wfColorChangeImage[i], outputFile, str_tag, coord, rect_files, selectedPreferenceImage, i, j, ratio=ratio)
    print()
    print(partChangedOutFile)