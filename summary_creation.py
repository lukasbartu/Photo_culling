__author__ = 'Lukáš Bartůněk'

import json
import operator
import natsort

def calculate_img_score(q_pth,t_a_ratio):
    image_overall_scores = []
    with open(q_pth) as json_file:
        q_data = json.load(json_file)
    q_list = sorted(q_data, key=operator.itemgetter("id"))
    for i, temp in enumerate(q_list):
        quality_score = temp["aesthetic_quality"] * (1-t_a_ratio) +  temp["technical_quality"] * t_a_ratio
        image_overall_scores += [{"id": i,
                                  "img": temp["img"],
                                  "score": quality_score}]
    return image_overall_scores

def update_scores(sim_data,image_scores,s_t,img,img_num,selection):
    if selection:
        sim_penalty = 0.8
    else:
        sim_penalty = 0
    in_window = False
    for temp in sim_data:
        if img == temp["first_img"]:
            in_window = True
            if temp["feature_similarity_score"] >= s_t or temp["content_similarity_score"] >= s_t:
                i = temp["second_id"]
                image_scores[i].update({"score": image_scores[i]["score"] * sim_penalty})
                img_num -= 1
        elif in_window: # after updating all neighbours
            break
    return image_scores, img_num

def select_summary(sim_pth,q_pth,num,s_t,t_a_r,q_cutoff,percent=10,selection=False):
    if selection:
        select_num = int(num*(percent/100))
    else:
        select_num = num
    image_scores = calculate_img_score(q_pth, t_a_r)
    top_list = []
    selected = 0
    with open(sim_pth) as json_file:
        sim_data = json.load(json_file)
    while selected < select_num:
        if selected != 0:
            image_scores, num = update_scores(sim_data, image_scores, s_t, added_img,num,selection)
        sorted_imgs = sorted(image_scores, key=operator.itemgetter("score"), reverse=True)
        temp = sorted_imgs[0]
        if not selection and (temp["score"] < q_cutoff or selected==num):
            break
        added_img = temp["img"]
        added_id = temp["id"]
        image_scores[added_id].update({"score": 0})
        top_list.append(added_img)
        selected +=1
        top_list = natsort.natsorted(top_list)
    return top_list