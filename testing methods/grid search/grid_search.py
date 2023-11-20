__author__ = 'Lukáš Bartůněk'

from summary_creation import select_summary
from utils import prepare_img_list
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
    sim_paths, q_paths, img_nums, res_lists = args
    res_list1, _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_1_Sossusvlei")
    res_list2, _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_2_WalvisBay")
    res_list3, _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_3_Etosha")
    res_list4, _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_5_VictoriaFalls")
    res_lists = [res_list1, res_list2, res_list3, res_list4]
    s_t,q_t,t_a_r= params
    percentage = 0
    summary = [0,0,0,0]
    f1_list = [0,0,0,0]
    for i,s in enumerate(summary):
        temp = select_summary(sim_pth=sim_paths[i], q_pth=q_paths[i], num=img_nums[i],
                             s_t=s_t, t_a_r=t_a_r, q_cutoff=q_t,selection=False,percent=percentage)

        true_positive = num_common_elements(temp, res_lists[i])
        false_positive = len(temp) - true_positive
        false_negative = len(res_lists[i]) - true_positive

        if not true_positive == 0:
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)

            f1_list[i] = (2 * precision * recall) / (precision + recall)
        else:
            f1_list[i] = 0

    f1 = sum(f1_list)/len(f1_list)  # average f1 for training summaries
    return -f1

params = (slice(0, 101, 10), slice(0, 101, 10), slice(0, 101, 10))

sim_paths = ["data/image_similarity_Sossusvlei.json","data/image_similarity_WalvisBay.json","data/image_similarity_Etosha.json","data/image_similarity_VictoriaFalls.json"]
q_paths = ["data/image_quality_Sossusvlei.json","data/image_quality_WalvisBay.json","data/image_quality_Etosha.json","data/image_quality_VictoriaFalls.json"]
img_lists = [0,0,0,0]
img_nums = [0,0,0,0]

img_lists[0],img_nums[0] = prepare_img_list("/home/lukas/Bakalářka/photo_culling/images/kybic_photos/originals/Sossusvlei")
img_lists[1],img_nums[1] = prepare_img_list("/home/lukas/Bakalářka/photo_culling/images/kybic_photos/originals/WalvisBay")
img_lists[2],img_nums[2] = prepare_img_list("/home/lukas/Bakalářka/photo_culling/images/kybic_photos/originals/Etosha")
img_lists[3],img_nums[3] = prepare_img_list("/home/lukas/Bakalářka/photo_culling/images/kybic_photos/originals/VictoriaFalls")

res_list1, _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_1_Sossusvlei")
res_list2, _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_2_WalvisBay")
res_list3, _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_3_Etosha")
res_list4, _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_5_VictoriaFalls")
res_lists = [res_list1,res_list2,res_list3,res_list4]

for lst in res_lists:
    for i, res in enumerate(lst):
        lst[i] = res.split("/")[-1]

a,f,g,j = brute(func=evaluate_parameters,ranges=params,full_output = True,workers=-1, finish=optimize.fmin,
                args=tuple([sim_paths,q_paths,img_nums,res_lists]))

a = np.asarray(a)
f = np.asarray(f)
g = np.asarray(g)
j = np.asarray(j)

with open('../../grid search results/4D_run.txt', 'w') as file:
    file.write(str(a))
    file.write('\n\n')
    file.write(str(-f))
    file.write('\n\n')
    file.write(str(g))
    file.write('\n\n')
    file.write(str(-j))

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection='3d')

# Set axes label
ax.set_xlabel('S_T', labelpad=20)
ax.set_ylabel('Q_T', labelpad=20)
ax.set_zlabel('T_A_R', labelpad=20)

img = ax.scatter(g[0], g[1], g[2], c=-j, cmap=plt.cm.cividis)
fig.colorbar(img)

#surf = ax.plot_surface(g[0],g[1],-j, cmap = plt.cm.cividis)

plt.show()
