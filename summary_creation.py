__author__ = 'Lukáš Bartůněk'

import json, os
import webbrowser
import operator
import numpy as np
from content_assessment import get_content_score

def calculate_img_score(q_pth,c_pth,c_q_ratio):
    image_overall_scores = []
    with open(q_pth) as json_file:
        q_data = json.load(json_file)
    with open(c_pth) as json_file:
        c_data = json.load(json_file)
    c_list = sorted(c_data,key=operator.itemgetter("id"))
    for i, temp in enumerate(c_list):
        content = np.array(temp["content"])
        content_score = get_content_score(content)
        image_overall_scores+=[{"id": i,
                                "img": temp["img"],
                                "score": content_score}]
    q_list = sorted(q_data,key=operator.itemgetter("id"))
    for i, temp in enumerate(q_list):
        score = image_overall_scores[i]["score"] * (1-c_q_ratio) + temp["quality_mean"] * c_q_ratio
        image_overall_scores[i].update({"score": score})
    return image_overall_scores

def update_scores(sim_data,image_scores,q_t,img):
    sim_penalty = 0.8
    in_window = False
    for temp in sim_data:
        if img == temp["first_img"]:
            in_window = True
            if temp["similarity_score"] >= q_t:
                i = temp["second_id"]
                image_scores[i].update({"score": image_scores[i]["score"] * sim_penalty})
        elif in_window: # after updating all neighbours
            break
    return image_scores


# TODO - update simil
def select_summary(sim_pth,q_pth,c_pth,percent,num,s_t,dir_pth,c_q_r):
    select_num = int(num*(percent/100))
    image_scores = calculate_img_score(q_pth, c_pth, c_q_r)
    top_list = []
    selected = 0
    with open(sim_pth) as json_file:
        sim_data = json.load(json_file)
    while selected <select_num:
        if selected != 0:
            image_scores = update_scores(sim_data, image_scores, s_t, added_img)
        sorted_imgs = sorted(image_scores, key=operator.itemgetter("score"), reverse=True)
        temp = sorted_imgs[0]
        added_img = temp["img"]
        added_id = temp["id"]
        image_scores[added_id].update({"score": 0})
        top_list.append(added_img)
        selected +=1
    for t in top_list:
        webbrowser.open(os.path.join(dir_pth,t))
    return top_list