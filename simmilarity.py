import subprocess
import sys
import os
import shutil

DEBUG = False

def init_directory(same_image_dir, nonsame_image_dir):
    # Make basic directories.
    make_dir(same_image_dir)
    make_dir(nonsame_image_dir)

def get_filenames(working_dir):
    # Get files and add dirname front of files.
    files = os.listdir(working_dir)
    files = [working_dir + "/" + files[i] for i in range(len(files))]
    return files

def make_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

def get_not_same_file_name(nonsame_dir, i, j, file_path):
    return nonsame_dir + "/same_" + str(i) + "_" + str(j) + "_" + file_path.split("/")[-1]


if __name__ == "__main__":
    working_dir = "./Image/Images"
    same_dir = "./Image/Same"
    nonsame_dir = "./Image/NotSame"
    
    init_directory(same_dir, nonsame_dir)
    files = get_filenames(working_dir)
    file_amount = len(files)

    simmilarity_list = []
    simmilarity_TF = [True for _ in range(file_amount)]

    for i in range(0, file_amount):
        for j in range(i + 1, file_amount):
            score = subprocess.check_output(['pyssim', files[i], files[j]], encoding="utf8")
            if float(score) > 0.9:
                simmilarity_TF[i] = False
                simmilarity_TF[j] = False
                simmilarity_list.append((i, j))
            if DEBUG:
                print("For " + files[i] + " and " + files[j])
                print(score.rstrip())
    
    # Copy to nonsame directory
    for i in range(0, file_amount):
        if simmilarity_TF[i]:
            shutil.copy2(files[i], nonsame_dir)
    
    # Copy to same directory
    for i in range(0, file_amount):
        if not simmilarity_TF[i]:
            for j in range(0, len(simmilarity_list)):
                if simmilarity_list[j][0] == i:
                    shutil.copy2(files[simmilarity_list[j][0]], get_not_same_file_name(same_dir, i, j, files[simmilarity_list[j][0]]))
                    shutil.copy2(files[simmilarity_list[j][1]], get_not_same_file_name(same_dir, i, j, files[simmilarity_list[j][1]]))