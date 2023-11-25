__author__ = 'Lukáš Bartůněk'

import os
import json

names = ["Sossusvlei", "WalvisBay", "Etosha", "Chobe", "VictoriaFalls", "Lisbon"]
s_files = [0, 0, 0, 0, 0, 0]
q_files = [0, 0, 0, 0, 0, 0]
for i, n in enumerate(names):
    s_files[i] = "image_similarity_" + n + ".json"
    q_files[i] = "image_quality_" + n + ".json"

s = []
q = []

overall =0
for i, f in enumerate(q_files):
    with open(s_files[i]) as json_file:
        s_list = json.load(json_file)
        for l in s_list:
            l["first_id"] += overall
            l["second_id"] += overall
        s += s_list
    with open(q_files[i]) as json_file:
        q_list = json.load(json_file)
        for l in q_list:
            l["id"] += overall
        overall += len(q_list)
        q += q_list


print(len(s))
print(len(q))

with open(os.path.join(os.getcwd(), "image_similarity_originals.json"), "w") as write_file:
    json.dump(s, write_file, indent=2)

with open(os.path.join(os.getcwd(), "image_quality_originals.json"), "w") as write_file:
    json.dump(q, write_file, indent=2)
