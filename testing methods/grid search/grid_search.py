__author__ = 'Lukáš Bartůněk'

from summary_creation import select_summary
import numpy as np
import json


def num_common_elements(list1, list2):
    result = []
    for element in list1:
        name = element.split("/")[-1]
        if name in list2:
            result.append(element)
    return len(result)


with open('../logistic regression grad search/res_lists.json') as json_file:
    res_lists = json.load(json_file)

r_lst = []
for r in res_lists:
    r_lst += r

s_file = "image_similarity_originals.json"
q_file = "image_quality_originals.json"

q_range = [0, 100, 50, 100]
s_range = [0, 100, 50, 100]
tar_range = [0, 100, 20, 100]
scr_range = [0, 100, 20, 100]
best_f1 = 0
good_f1 = 0
best_p = [50, 50, 50, 50]

possible_good_q = [50]
possible_good_s = [50]
possible_good_t = [50]
possible_good_c = [50]

for i in range(5):
    qt = np.arange(q_range[0], q_range[1], q_range[2])
    st = np.arange(s_range[0], s_range[1], s_range[2])
    tar = np.arange(tar_range[0], tar_range[1], tar_range[2])
    scr = np.arange(scr_range[0], scr_range[1], scr_range[2])
    print('TESTING C')
    good_f1 = 0
    possible_good_c = []
    for i, c in enumerate(scr):
        if i == 0:
            continue
        for s in possible_good_s:
            for q in possible_good_q:
                for t in possible_good_t:
                    summary = select_summary(sim_pth=s_file, q_pth=q_file, num=6000,
                                             s_t=s, t_a_ratio=t, q_cutoff=q, s_c_ratio=c)
                    true_positive = num_common_elements(summary, r_lst)
                    false_positive = len(summary) - true_positive
                    false_negative = len(r_lst) - true_positive
                    f1 = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
                    if f1 > good_f1:
                        good_f1 = f1
                        possible_good_c = [c]
                    elif f1 == good_f1 and c not in possible_good_c:
                        possible_good_c.append(c)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_p = [q, s, t, c]
                    print("Q", q, "S", s, "T", t, "C", c, "F1", f1)
    print('TESTING T')
    good_f1 = 0
    possible_good_t = []
    for i, t in enumerate(tar):
        if i == 0:
            continue
        for s in possible_good_s:
            for q in possible_good_q:
                for c in possible_good_c:
                    summary = select_summary(sim_pth=s_file, q_pth=q_file, num=6000,
                                             s_t=s, t_a_ratio=t, q_cutoff=q, s_c_ratio=c)

                    true_positive = num_common_elements(summary, r_lst)
                    false_positive = len(summary) - true_positive
                    false_negative = len(r_lst) - true_positive
                    f1 = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
                    if f1 > good_f1:
                        good_f1 = f1
                        possible_good_t = [t]
                    elif f1 == good_f1 and t not in possible_good_t:
                        possible_good_t.append(t)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_p = [q, s, t, c]
                    print("Q", q, "S", s, "T", t, "C", c, "F1", f1)
    print('TESTING Q')
    good_f1 = 0
    possible_good_q = []
    for i, q in enumerate(qt):
        if i == 0:
            continue
        for s in possible_good_s:
            for t in possible_good_t:
                for c in possible_good_c:
                    summary = select_summary(sim_pth=s_file, q_pth=q_file, num=6000,
                                             s_t=s, t_a_ratio=t, q_cutoff=q, s_c_ratio=c)
                    true_positive = num_common_elements(summary, r_lst)
                    false_positive = len(summary) - true_positive
                    false_negative = len(r_lst) - true_positive
                    f1 = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
                    if f1 > good_f1:
                        good_f1 = f1
                        possible_good_q = [q]
                    elif f1 == good_f1 and q not in possible_good_q:
                        possible_good_q.append(q)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_p = [q, s, t, c]
                    print("Q", q, "S", s, "T", t, "C", c, "F1", f1)
    print('TESTING S')
    good_f1 = 0
    possible_good_s = []
    for i, s in enumerate(st):
        if i == 0:
            continue
        for q in possible_good_q:
            for t in possible_good_t:
                for c in possible_good_c:
                    summary = select_summary(sim_pth=s_file, q_pth=q_file, num=6000,
                                             s_t=s, t_a_ratio=t, q_cutoff=q, s_c_ratio=c)
                    true_positive = num_common_elements(summary, r_lst)
                    false_positive = len(summary) - true_positive
                    false_negative = len(r_lst) - true_positive
                    f1 = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
                    if f1 > good_f1:
                        good_f1 = f1
                        possible_good_s = [s]
                    elif f1 == good_f1 and s not in possible_good_s:
                        possible_good_s.append(s)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_p = [q, s, t, c]
                    print("Q", q, "S", s, "T", t, "C", c, "F1", f1)



    print("BEST PARS:", best_p, "BEST F1", best_f1)
    size = [q_range[3], s_range[3], tar_range[3], scr_range[3]]
    q_range = [max(best_p[0] - size[0] / 4, 0), min(best_p[0] + size[0] / 4, 100), size[0] / 10,
               min(best_p[0] + size[0] / 4, 100) - max(best_p[0] - size[0] / 4, 0)]
    s_range = [max(best_p[1] - size[1] / 4, 0), min(best_p[1] + size[1] / 4, 100), size[1] / 10,
               min(best_p[1] + size[1] / 4, 100) - max(best_p[1] - size[1] / 4, 0)]
    tar_range = [max(best_p[2] - size[2] / 4, 0), min(best_p[2] + size[2] / 4, 100), size[2] / 20,
                 min(best_p[2] + size[2] / 4, 100) - max(best_p[2] - size[2] / 4, 0)]
    scr_range = [max(best_p[3] - size[3] / 4, 0), min(best_p[3] + size[3] / 4, 100), size[3] / 20,
                 min(best_p[3] + size[3] / 4, 100) - max(best_p[3] - size[3] / 4, 0)]
    print("!!!RANGES!!!")
    print(q_range)
    print(s_range)
    print(tar_range)
    print(scr_range)
    possible_good_q = [best_p[0]]
    possible_good_s = [best_p[1]]
    possible_good_t = [best_p[2]]
    possible_good_c = [best_p[3]]

