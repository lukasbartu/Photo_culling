__author__ = 'Lukáš Bartůněk'

import json
import operator
import natsort


def calculate_img_score(q_pth, t_a_ratio):
    image_overall_scores = []
    with open(q_pth) as json_file:
        q_data = json.load(json_file)
    q_list = sorted(q_data, key=operator.itemgetter("id"))
    for i, temp in enumerate(q_list):
        quality_score = temp["aesthetic_quality"] * (1-t_a_ratio) + temp["technical_quality"] * t_a_ratio
        image_overall_scores += [{"id": i,
                                  "img": temp["img"],
                                  "score": quality_score}]
    return image_overall_scores


def update_scores(sim_data, image_scores, s_t, img, s_c_ratio):
    for i in range(img["id"]-20, img["id"]+21):
        if sim_data[i]["first_id"] != img["id"]:
            continue
        if (s_c_ratio * sim_data[i]["feature_similarity_score"] + (1-s_c_ratio) * sim_data[i]["content_similarity_score"]) >= s_t:
            i = sim_data[i]["second_id"]
            image_scores[i].update({"score": 0.1})
    return image_scores


def select_summary(sim_pth, q_pth, num, s_t, t_a_ratio, s_c_ratio, q_cutoff, size=10, size_based=False):
    if size_based:
        select_num = int(num*(size/100))
    else:
        select_num = num
    image_scores = calculate_img_score(q_pth, t_a_ratio)
    top_list = []
    selected = 0
    with open(sim_pth) as json_file:
        sim_data = json.load(json_file)
    while selected < select_num:
        if selected != 0:
            image_scores = update_scores(sim_data, image_scores, s_t, added_img, s_c_ratio)
        sorted_imgs = sorted(image_scores, key=operator.itemgetter("score"), reverse=True)
        added_img = sorted_imgs[0]
        if not size_based and (added_img["score"] < q_cutoff):
            break
        elif size_based and added_img["score"] < 0:
            break
        image_scores[added_img["id"]].update({"score": -1})
        top_list.append(added_img["img"])
        selected += 1
        top_list = natsort.natsorted(top_list)
    return top_list