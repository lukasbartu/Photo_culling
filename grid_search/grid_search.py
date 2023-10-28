__author__ = 'Lukáš Bartůněk'

from summary_creation import select_summary
from utils import prepare_paths,prepare_img_list
import numpy as np
from scipy.optimize import brute
from scipy import optimize
import matplotlib.pyplot as plt

def num_common_elements(list1, list2):
    result = []
    for element in list1:
        name = element.split("/")[-1]
        if name in list2:
            result.append(element)
    return len(result)

def evaluate_parameters(params,*args):
    sim_path, q_path, img_num, res_lists = args
    s_t, q_t = params
    t_a_r = 0
    summary = select_summary(sim_pth=sim_path, q_pth=q_path, num=img_num,
                             s_t=s_t, t_a_r=t_a_r, q_cutoff=q_t)
    f1_list = [0,0,0,0]
    for i,res_list in enumerate(res_lists):
        true_positive = num_common_elements(summary, res_list)
        false_positive = len(summary) - true_positive
        false_negative = len(res_list) - true_positive

        if not true_positive == 0:
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)

            f1_list[i] = 2 * (precision * recall) / (precision + recall)
        else:
            f1_list[i] = 0

    f1 = sum(f1_list)/len(f1_list)  # average f1 for training summaries
    return -f1

params = (slice(0, 21, 2), slice(20,81,6))

abs_pth,sim_path,q_path,c_path = prepare_paths("/images/kybic_photos/originals",abs_p=False)
img_list,img_num = prepare_img_list(abs_pth)

res_list1, _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_1_Sossusvlei")
res_list2, _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_2_WalvisBay")
res_list3, _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_3_Etosha")
res_list4, _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_4_Chobe")
res_lists = [res_list1,res_list2,res_list3,res_list4]

for lst in res_lists:
    for i, res in enumerate(lst):
        lst[i] = res.split("/")[-1]

a,f,g,j = brute(func=evaluate_parameters,ranges=params,full_output = True,finish=None,workers=-1,
                args=["image_similarities_originals.json","image_quality_originals.json",img_num,res_lists])

a = np.asarray(a)
f = np.asarray(f)
g = np.asarray(g)
j = np.asarray(j)

with open('fourth_run.txt', 'w') as file:
    file.write(str(a))
    file.write('\n\n')
    file.write(str(f))
    file.write('\n\n')
    file.write(str(g))
    file.write('\n\n')
    file.write(str(j))

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')

# Set axes label
ax.set_xlabel('s_t', labelpad=20)
ax.set_ylabel('q_t', labelpad=20)
ax.set_zlabel('f1', labelpad=20)

surf = ax.plot_surface(g[0],g[1],-j, cmap = plt.cm.cividis)

plt.show()
