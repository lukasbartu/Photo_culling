__author__ = 'Lukáš Bartůněk'

import json
import operator
import natsort


def generate_preset_values(q, s, tar, scr, q_path, s_path):
    tar_preset = (tar*25)/100
    scr_preset = (scr*25)/100

    qualities = calculate_img_score(q_path, tar_preset)
    qualities = sorted(qualities, key=operator.itemgetter("score"))
    q_range = qualities[-1]["score"] - qualities[0]["score"]
    q_preset = (q_range/4)*q + qualities[0]["score"]

    similarities = calculate_img_similarities(s_path, scr_preset)
    similarities = sorted(similarities, key=operator.itemgetter("score"))
    s_range = similarities[-1]["score"] - similarities[0]["score"]
    s_preset = (s_range/4)*s + similarities[0]["score"]

    return q_preset, s_preset, tar_preset, scr_preset


def calculate_img_similarities(s_pth, s_c_ratio):
    image_overall_similarities = []
    with open(s_pth) as json_file:
        s_data = json.load(json_file)
    for i, temp in enumerate(s_data):
        similarity = temp["feature_similarity_score"] * (1 - s_c_ratio) + temp["content_similarity_score"] * s_c_ratio
        image_overall_similarities += [{"pair": i,
                                        "score": similarity}]
    return image_overall_similarities


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
    for i in range(len(sim_data)):
        if sim_data[i]["first_id"] != img["id"]:
            continue
        if (s_c_ratio * sim_data[i]["feature_similarity_score"] +
           (1-s_c_ratio) * sim_data[i]["content_similarity_score"]) >= s_t:
            i = sim_data[i]["second_id"]
            image_scores[i].update({"score": 0})
    return image_scores


def select_summary(sim_pth, q_pth, num, s_t, t_a_ratio, s_c_ratio, q_cutoff, size_based, size=10):
    select_num = int(num*(size/100))
    image_scores = calculate_img_score(q_pth, t_a_ratio)
    top_list = []
    added_img = None
    with open(sim_pth) as json_file:
        sim_data = json.load(json_file)
    while True:
        if len(top_list) != 0:
            image_scores = update_scores(sim_data, image_scores, s_t, added_img, s_c_ratio)
        added_img = sorted(image_scores, key=operator.itemgetter("score"), reverse=True)[0]
        if not size_based and added_img["score"] <= q_cutoff:
            break
        elif size_based and (added_img["score"] <= 0 or len(top_list) >= select_num):
            break
        image_scores[added_img["id"]].update({"score": -1})
        top_list.append(added_img["img"])
        top_list = natsort.natsorted(top_list)
    return top_list
