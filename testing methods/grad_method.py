__author__ = 'Lukáš Bartůněk'

import torch
from utils import prepare_img_list
import json
import operator
import matplotlib.pyplot as plt

img_lists = [0, 0, 0, 0, 0]
res_lists = [0, 0, 0, 0, 0]
names = ["Sossusvlei", "WalvisBay", "Etosha", "Chobe", "VictoriaFalls"]

img_lists[0], _ = prepare_img_list("images/kybic_photos/originals/Sossusvlei")
img_lists[1], _ = prepare_img_list("images/kybic_photos/originals/WalvisBay")
img_lists[2], _ = prepare_img_list("images/kybic_photos/originals/Etosha")
img_lists[3], _ = prepare_img_list("images/kybic_photos/originals/Chobe")
img_lists[4], _ = prepare_img_list("images/kybic_photos/originals/VictoriaFalls")

res_lists[0], _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_1_Sossusvlei")
res_lists[1], _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_2_WalvisBay")
res_lists[2], _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_3_Etosha")
res_lists[3], _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_4_Chobe")
res_lists[4], _ = prepare_img_list("images/kybic_photos/corrected/281_namibie/Namibia2019_5_VictoriaFalls")


for i, img_list in enumerate(img_lists):
    for j, img in enumerate(img_list):
        img_list[j] = img.split("/")[-1]

for i, img_list in enumerate(res_lists):
    for j, img in enumerate(img_list):
        img_list[j] = img.split("/")[-1]

train_data = []
train_data_sim = []
train_results = []
validate_data = []
validate_data_sim = []
validate_results = []
test_data = []
test_data_sim = []
test_results = []

for i, img_list in enumerate(img_lists):
    q_file = "data/image_quality_" + names[i] + ".json"
    s_file = "data/image_similarity_" + names[i] + ".json"
    sim_scores = []
    with open(q_file) as f:
        q_data = json.load(f)
    q_list = sorted(q_data, key=operator.itemgetter("id"))

    with open(s_file) as f:
        s_data = json.load(f)
    s_list = sorted(s_data, key=operator.itemgetter("first_id"))

    last_id = -1
    sim_scores = []
    for s in s_list:
        if not s["first_id"] == last_id:
            sim_scores.append([])
            if s["first_id"] < 5:
                sim_scores[s["first_id"]].append([0, 0, 0, 0])
            if s["first_id"] < 4:
                sim_scores[s["first_id"]].append([0, 0, 0, 0])
            if s["first_id"] < 3:
                sim_scores[s["first_id"]].append([0, 0, 0, 0])
            if s["first_id"] < 2:
                sim_scores[s["first_id"]].append([0, 0, 0, 0])
            if s["first_id"] < 1:
                sim_scores[s["first_id"]].append([0, 0, 0, 0])
            if not last_id == -1:
                while len(sim_scores[last_id]) < 10:
                    sim_scores[last_id].append([0, 0, 0, 0])
            last_id = s["first_id"]
        for q in q_list:
            if q["id"] == s["second_id"]:
                second_img_score = [q["aesthetic_quality"], q["technical_quality"]]
        sim_scores[last_id].append(
            [second_img_score[0], second_img_score[1], s["feature_similarity_score"], s["content_similarity_score"]])
    while len(sim_scores[last_id]) < 10:
        sim_scores[last_id].append([0, 0, 0, 0])

    for j, img in enumerate(img_list):
        if img in res_lists[i]:
            if i == 4:
                test_data.append((q_list[j]["aesthetic_quality"], q_list[j]["technical_quality"]))
                test_data_sim.append(sim_scores[j])
                test_results.append(1)
            elif i == 1:
                validate_data.append((q_list[j]["aesthetic_quality"], q_list[j]["technical_quality"]))
                validate_data_sim.append(sim_scores[j])
                validate_results.append(1)
            else:
                train_data.append((q_list[j]["aesthetic_quality"], q_list[j]["technical_quality"]))
                train_data_sim.append(sim_scores[j])
                train_results.append(1)
        else:
            if i == 4:
                test_data.append((q_list[j]["aesthetic_quality"], q_list[j]["technical_quality"]))
                test_data_sim.append(sim_scores[j])
                test_results.append(0)
            elif i == 1:
                validate_data.append((q_list[j]["aesthetic_quality"], q_list[j]["technical_quality"]))
                validate_data_sim.append(sim_scores[j])
                validate_results.append(0)
            else:
                train_data.append((q_list[j]["aesthetic_quality"], q_list[j]["technical_quality"]))
                train_data_sim.append(sim_scores[j])
                train_results.append(0)


train_data = torch.asarray(train_data).T
train_data_sim = torch.transpose(torch.asarray(train_data_sim), 0, 2)
train_results = torch.asarray(train_results)

validate_data = torch.asarray(validate_data).T
validate_data_sim = torch.transpose(torch.asarray(validate_data_sim), 0, 2)
validate_results = torch.asarray(validate_results)

test_data = torch.asarray(test_data).T
test_data_sim = torch.transpose(torch.asarray(test_data_sim), 0, 2)
test_results = torch.asarray(test_results)

true_samples = 0
false_samples = 0
for result in train_results:
    if result == 1:
        true_samples += 1
    else:
        false_samples += 1
class_weights_train = [len(train_results) / true_samples, len(train_results) / false_samples]


def forward(x, s, w):
    ui = (x[0] * (1 - (w[0]/100)) + x[1] * (w[0]/100))
    uj = (s[0] * (1 - (w[0]/100)) + s[1] * (w[0]/100))
    si = (s[2] * (1 - (w[2]/100)) + s[3] * (w[2]/100))
    temp1 = torch.sigmoid(w[3] - si)
    temp2 = torch.sigmoid(ui - uj)  # Ui > Uj ?
    temp4 = 1 - (1-((1 - temp1) * temp2)) * (1-temp1)
    temp5 = torch.prod(temp4, dim=0)

    p = torch.sigmoid(ui - w[1]) * temp5
    return p


def loss_fun(y, y_pred, c_w):
    eps = 1e-20
    return -torch.sum(c_w[0] * (y * (torch.log(y_pred + eps))) +
                      (c_w[1] * (1 - y) * (torch.log(1 - y_pred + eps)))) / len(y)


# t_a_r, q_t, f_c_r, s_t
weights = torch.tensor([40, 65, 60, 20], requires_grad=True, dtype=torch.double)

loss_BGD = []

change = 0
momentum = 0.9
lr = torch.asarray([0.1, 0.1, 0.5, 0.5])
best_loss = 1e10
best_weights = []
for i in range(5000):
    pred = forward(train_data, train_data_sim, weights)
    loss = loss_fun(train_results, pred, class_weights_train)
    loss_BGD.append(loss.item())
    loss.backward()

    pred_validate = forward(validate_data, validate_data_sim, weights)
    loss_validate = loss_fun(train_results, pred, class_weights_train)

    print(i, loss.item(), loss_validate.item(),
          [weights[0].item(), weights[1].item(), weights[2].item(), weights[3].item()])

    if loss_validate.item() < best_loss:
        best_loss = loss_validate.item()
        best_weights = weights
        best_epoch = i

    with torch.no_grad():
        weights -= momentum * change
        new_change = momentum * change + lr * weights.grad
        weights -= new_change
        change = new_change
        weights[0] = min(weights[0], 100)
        weights[1] = min(weights[1], 100)
        weights[2] = min(weights[2], 100)
        weights[3] = min(weights[3], 100)
        weights[0] = max(weights[0], 0)
        weights[1] = max(weights[1], 0)
        weights[2] = max(weights[2], 0)
        weights[3] = max(weights[3], 0)
    weights.grad.zero_()


print("TRAIN LOSS:", best_loss, best_epoch,
      [best_weights[0].item(), best_weights[1].item(), best_weights[2].item(), best_weights[3].item()])

plt.plot(loss_BGD, label="Batch Gradient Descent")
plt.xlabel('Epoch')
plt.ylabel('Cost/Total loss')
plt.legend()
plt.show()

# test_data = torch.asarray([50, 50])
# test_data_sim = torch.asarray([
#     [[100, 100, 100,   100], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
#     [[  0,   0, 100,   100], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
#     [[100, 100, 100,     0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
#     [[  0,   0, 100,     0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
#     [[100, 100,   0,   100], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
#     [[  0,   0,   0,   100], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
#     [[100, 100,   0,     0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
#     [[  0,   0,   0,     0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
#     ])
# test_data_sim = torch.transpose(torch.asarray(test_data_sim), 0, 2)
# best_weights = torch.asarray([50,50,50,50])


true_samples = 0
false_samples = 0
for result in test_results:
    if result == 1:
        true_samples+=1
    else:
        false_samples+=1
class_weights_test = [len(test_results) / true_samples,len(test_results) / false_samples]

pred = forward(test_data, test_data_sim, best_weights)
loss = loss_fun(test_results, pred, class_weights_test)

print("TEST LOSS:", loss.item())

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
true_num = 0
false_num = 0
with torch.no_grad():

    for i, p in enumerate(pred):
        if p >= 0.5 and test_results[i].item() == 1:
            true_positive += 1
        elif p < 0.5 and test_results[i].item() == 0:
            true_negative += 1
        elif p >= 0.5 and test_results[i].item() == 0:
            false_positive += 1
        else:
            false_negative += 1

    if true_positive == 0:
        f1 = 0
    else:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = (2 * precision * recall) / (precision + recall)

print("F1 Score:", f1*100)
