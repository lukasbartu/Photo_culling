__author__ = 'Lukáš Bartůněk'

import os
import natsort
import pathlib
import shutil
import json
import operator


def prepare_paths(pth):
    folder_name = pth.split("/")[-1]  # name of most nested folder for better naming of resulting files
    sim_path = "data/image_similarity_" + folder_name + ".json"  # file to save result of precalculating similarities
    q_path = "data/image_quality_" + folder_name + ".json"  # file to save result of quality evaluation
    content_path = "data/image_content_" + folder_name + ".json"  # file to save result of image content evaluation
    return sim_path, q_path, content_path


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
    class_weights = [(1 / true_samples) * (len(results) / 2.0), (1 / false_samples) * (len(results) / 2.0)]
    return class_weights


def load_trained():
    with open('data/recommended_parameters.json') as json_file:
        data = json.load(json_file)
        q_t, s_t, t_a_ratio, s_c_ratio = data
    return float(q_t), float(s_t), float(t_a_ratio), float(s_c_ratio)


def get_sim_window(s_lst):
    first_id = s_lst[0]["first_id"]
    n = 0
    for s_l in s_lst:
        if s_l["first_id"] == first_id:
            n += 1
    return n


def copy_images(summary, folder, dest_folder):
    folder_name = folder.split("/")[-1]
    save_folder = dest_folder + "/Selected summary " + folder_name
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
    for img in summary:
        shutil.copy2(os.path.join(folder, img), save_folder)


def save_list(summary, folder, dest_folder):
    folder_name = folder.split("/")[-1]
    save_folder = dest_folder + "/Selected summary " + folder_name + ".txt"
    with open(save_folder, "w") as file:
        for img in summary:
            file.write(img)
            file.write(", ")


def format_data_sim(s_file, q_file):
    with open(q_file) as f:
        q_data = json.load(f)
    q_list = sorted(q_data, key=operator.itemgetter("id"))
    with open(s_file) as f:
        s_data = json.load(f)
    s_list = sorted(s_data, key=operator.itemgetter("first_id"))

    nbrs = get_sim_window(s_list)
    last_id = - 1
    data_sim = []
    for s in s_list:
        if not s["first_id"] == last_id:
            data_sim.append([])
            spaces = nbrs - s["first_id"]
            while spaces > 0:
                data_sim[s["first_id"]].append([0, 0, 0, 0])
                spaces -= 1
            if not last_id == -1:
                while len(data_sim[last_id]) < nbrs * 2:
                    data_sim[last_id].append([0, 0, 0, 0])
            last_id = s["first_id"]
        second_img_score = [0, 0]
        for q in q_list:
            if q["id"] == s["second_id"]:
                second_img_score = [q["aesthetic_quality"], q["technical_quality"]]
        data_sim[last_id].append(
            [second_img_score[0], second_img_score[1], s["feature_similarity_score"], s["content_similarity_score"]])
    while len(data_sim[last_id]) < nbrs * 2:
        data_sim[last_id].append([0, 0, 0, 0])
    return data_sim, nbrs
