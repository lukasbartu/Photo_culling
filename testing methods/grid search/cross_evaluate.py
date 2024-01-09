__author__ = 'Lukáš Bartůněk'

import json
import summary_creation


def num_common_elements(list1, list2):
    result = []
    for element in list1:
        name = element.split("/")[-1]
        if name in list2:
            result.append(element)
    return len(result)


names = ["Sossusvlei", "WalvisBay", "Etosha", "Chobe", "VictoriaFalls"]
summary = [0, 0, 0, 0, 0]

with open('res_lists.json') as json_file:
    res_lists = json.load(json_file)

for i, name in enumerate(names):
    q_file = "image_quality_" + names[i] + ".json"
    s_file = "image_similarity_" + names[i] + ".json"

    summary[i] = summary_creation.select_summary(sim_pth=s_file, q_pth=q_file, num=5000,
                                                 s_t=10, t_a_ratio=5, q_cutoff=55, s_c_ratio=5,
                                                 size_based=False)

true_positive = []
false_positive = []
false_negative = []

for i in range(len(summary)):
    true_positive.append(num_common_elements(summary[i], res_lists[i]))
    false_positive.append(len(summary[i]) - true_positive[i])
    false_negative.append(len(res_lists[i]) - true_positive[i])

f1 = (2 * sum(true_positive)) / (2 * sum(true_positive) + sum(false_positive) + sum(false_negative))
print("F1:", f1)
