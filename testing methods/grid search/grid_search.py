__author__ = 'Lukáš Bartůněk'

from summary_creation import select_summary
from utils import remove_folder_name
import numpy as np
from scipy.optimize import brute
from scipy import optimize
import matplotlib.pyplot as plt
import json


def num_common_elements(list1, list2):
    result = []
    for element in list1:
        name = element.split("/")[-1]
        if name in list2:
            result.append(element)
    return len(result)


with open('res_lists.json') as json_file:
    res_lists = json.load(json_file)

r_lst = []
for r in res_lists:
    r_lst += r

s_file = "image_similarity_originals.json"
q_file = "image_quality_originals.json"

q_range = [0, 101, 10, 100]
s_range = [0, 101, 10, 100]
tar_range = [0, 101, 10, 100]
scr_range = [0, 101, 10, 100]
q = 50
t = 50
s = 50
c = 50
best_f1 = 0
best_p = [0, 0, 0, 0]

for i in range(5):
    qt = np.arange(q_range[0], q_range[1], q_range[2])
    st = np.arange(s_range[0], s_range[1], s_range[2])
    tar = np.arange(tar_range[0], tar_range[1], tar_range[2])
    scr = np.arange(scr_range[0], scr_range[1], scr_range[2])
    for t in tar:
        summary = select_summary(sim_pth=s_file, q_pth=q_file, num=3000,
                                 s_t=s, t_a_ratio=t, q_cutoff=q)

        true_positive = num_common_elements(summary, r_lst)
        false_positive = len(summary) - true_positive
        false_negative = len(r_lst) - true_positive
        f1 = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
        print("Q", q, "S", s, "T", t, "F1", f1)
    t = best_t
    for q in qt:
        summary = select_summary(sim_pth=s_file, q_pth=q_file, num=3000,
                                 s_t=s, t_a_ratio=t, q_cutoff=q, s_c_ratio=c)

        true_positive = num_common_elements(summary, r_lst)
        false_positive = len(summary) - true_positive
        false_negative = len(r_lst) - true_positive
        f1 = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
        if f1 > best_f1:
            best_f1 = f1
            best_q = q
        print("Q", q, "S", s, "T", t, "C", c, "F1", f1)
    q = best_q
    for c in tar:
        summary = select_summary(sim_pth=s_file, q_pth=q_file, num=3000,
                                 s_t=s, t_a_ratio=t, q_cutoff=q, s_c_ratio=c)

        true_positive = num_common_elements(summary, r_lst)
        false_positive = len(summary) - true_positive
        false_negative = len(r_lst) - true_positive
        f1 = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
        if f1 > best_f1:
            best_f1 = f1
            best_c = c
        print("Q", q, "S", s, "T", t, "C", c, "F1", f1)
    c = best_c
    for s in st:
        summary = select_summary(sim_pth=s_file, q_pth=q_file, num=3000,
                                 s_t=s, t_a_ratio=t, q_cutoff=q, s_c_ratio=c)

        true_positive = num_common_elements(summary, r_lst)
        false_positive = len(summary) - true_positive
        false_negative = len(r_lst) - true_positive
        f1 = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
        if f1 > best_f1:
            best_f1 = f1
            best_s = s
        print("Q", q, "S", s, "T", t, "C", c, "F1", f1)
    s = best_s

    best_p = [best_q, best_s, best_t, best_c]
    print("BEST PARS:", best_p)
    size = [q_range[3], s_range[3], tar_range[3]]
    q_range =   [max(best_p[0] - size[0]/10, 0), min(best_p[0] + size[0]/10, 100), size[0] / 100, min(best_p[0] + size[0]/10, 100) - max(best_p[0] - size[0]/10, 0)]
    s_range =   [max(best_p[1] - size[1]/10, 0), min(best_p[1] + size[1]/10, 100), size[1] / 100, min(best_p[1] + size[1]/10, 100) - max(best_p[1] - size[1]/10, 0)]
    tar_range = [max(best_p[2] - size[2]/10, 0), min(best_p[2] + size[2]/10, 100), size[2] / 100, min(best_p[2] + size[2]/10, 100) - max(best_p[2] - size[2]/10, 0)]
    scr_range = [max(best_p[3] - size[3]/10, 0), min(best_p[3] + size[3]/10, 100), size[3] / 100, min(best_p[3] + size[3]/10, 100) - max(best_p[3] - size[3]/10, 0)]
    print("!!!RANGES!!!")
    print(q_range)
    print(s_range)
    print(tar_range)
