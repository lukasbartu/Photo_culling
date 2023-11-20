__author__ = 'Lukáš Bartůněk'

import os
import natsort
import pathlib


def prepare_paths(pth, abs_p=False):
    if abs_p:
        abs_pth = pth
    else:
        abs_pth = os.getcwd() + pth  # absolute file of image directory
    folder_name = abs_pth.split("/")[-1]  # name of most nested folder for better naming of resulting files
    sim_path = "data/image_similarity_" + folder_name + ".json"  # file to save result of precalculating similarities
    q_path = "data/image_quality_" + folder_name + ".json"  # file to save result of quality evaluation
    content_path = "data/image_content_" + folder_name + ".json"  # file to save result of image content evaluation
    return abs_pth, sim_path, q_path, content_path


def prepare_img_list(pth):
    # assuming the files are numbered based on the sequence they have been taken in
    img_list = []  # list of image file names to process
    global_path = pathlib.Path(pth)
    list(global_path.rglob("*.jpg"))
    for p in list(global_path.rglob("*.jpg")):
        img_list.append(str(p))
    img_list = natsort.natsorted(img_list)
    return img_list, len(img_list)


def remove_folder_name(lst, f):
    copied_list = list.copy(lst)
    for i, l in enumerate(copied_list):
        copied_list[i] = l.replace(str(f + "/"), "")
    return copied_list


def get_class_weights(results):
    true_samples = 0
    false_samples = 0
    for result in results:
        if result == 1:
            true_samples += 1
        else:
            false_samples += 1
    class_weights = [ (1 / true_samples) * (len(results) / 2.0) , (1 / false_samples) * (len(results) / 2.0)]
    return class_weights

def get_sim_window(s_lst):
    first_id =  s_lst[0]["first_id"]
    n = 0

    for l in s_lst:
        if l["first_id"] == first_id:
            n+=1
    return n