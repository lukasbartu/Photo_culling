__author__ = 'Lukáš Bartůněk'

import os
import natsort
import  pathlib

def prepare_paths(pth,abs_p):
    if abs_p:
        abs_pth = pth
    else:
        abs_pth = os.getcwd() + pth  # absolute file of image directory
    folder_name = abs_pth.split("/")[-1]  # name of most nested folder for better naming of resulting files
    sim_path = "image_similarities_" + folder_name + ".json"  # file to save result of precalculating similarities
    q_path = "image_quality_" + folder_name + ".json"  # file to save result of quality evaluation
    content_path = "image_content_" + folder_name + ".json" # file to save result of image content evaluation
    return abs_pth, sim_path, q_path, content_path

def prepare_img_list(pth):
    # assuming the files are numbered based on the sequence they have been taken in
    img_list = []  # list of image file names to process
    global_path = pathlib.Path(pth)
    list(global_path.rglob("*.jpg"))
    for p in list(global_path.rglob("*.jpg")):
        img_list.append(str(p))

    #
    # for path in os.scandir(pth):
    #     if path.is_file():
    #         if path.name.endswith(".jpg"):
    #             img_list += [path.name]
    img_num = len(img_list)
    img_list = natsort.natsorted(img_list)
    return img_list,img_num
