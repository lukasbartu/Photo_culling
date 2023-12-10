__author__ = 'Lukáš Bartůněk'

import torch
import json
import operator
import matplotlib.pyplot as plt
import numpy as np

names = ["Sossusvlei", "WalvisBay", "Etosha", "Chobe", "VictoriaFalls"]

with open('img_lists.json') as json_file:
    img_lists = json.load(json_file)

with open('res_lists.json') as json_file:
    res_lists = json.load(json_file)

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
    q_file = "image_quality_" + names[i] + ".json"
    s_file = "image_similarity_" + names[i] + ".json"
    with open(q_file) as f:
        q_data = json.load(f)
    q_list = sorted(q_data, key=operator.itemgetter("id"))

    with open(s_file) as f:
        s_data = json.load(f)
    s_list = sorted(s_data, key=operator.itemgetter("first_id"))

    max_nbrs = 20
    last_id = - 1
    data_sim = []
    for s in s_list:
        if not s["first_id"] == last_id:
            data_sim.append([])
            spaces = max_nbrs - s["first_id"]
            while spaces > 0:
                data_sim[s["first_id"]].append([0, 0, 0, 0])
                spaces -= 1
            if not last_id == -1:
                while len(data_sim[last_id]) < max_nbrs * 2:
                    data_sim[last_id].append([0, 0, 0, 0])
            last_id = s["first_id"]
        second_img_score = [0, 0]
        for q in q_list:
            if q["id"] == s["second_id"]:
                second_img_score = [q["aesthetic_quality"], q["technical_quality"]]
        data_sim[last_id].append(
            [second_img_score[0], second_img_score[1], s["feature_similarity_score"], s["content_similarity_score"]])
    while len(data_sim[last_id]) < max_nbrs * 2:
        data_sim[last_id].append([0, 0, 0, 0])

    for j, img in enumerate(img_list):
        if img in res_lists[i]:
            if i == 5:
                test_data.append((q_list[j]["aesthetic_quality"], q_list[j]["technical_quality"]))
                test_data_sim.append(data_sim[j])
                test_results.append(1)
            elif i == 4:
                validate_data.append((q_list[j]["aesthetic_quality"], q_list[j]["technical_quality"]))
                validate_data_sim.append(data_sim[j])
                validate_results.append(1)
            else:
                train_data.append((q_list[j]["aesthetic_quality"], q_list[j]["technical_quality"]))
                train_data_sim.append(data_sim[j])
                train_results.append(1)
        else:
            if i == 5:
                test_data.append((q_list[j]["aesthetic_quality"], q_list[j]["technical_quality"]))
                test_data_sim.append(data_sim[j])
                test_results.append(0)
            elif i == 4:
                validate_data.append((q_list[j]["aesthetic_quality"], q_list[j]["technical_quality"]))
                validate_data_sim.append(data_sim[j])
                validate_results.append(0)
            else:
                train_data.append((q_list[j]["aesthetic_quality"], q_list[j]["technical_quality"]))
                train_data_sim.append(data_sim[j])
                train_results.append(0)

train_data = torch.asarray(train_data).T
train_data_sim = torch.transpose(torch.asarray(train_data_sim), 0, 2)
train_results = torch.asarray(train_results)

validate_data = torch.asarray(validate_data).T
validate_data_sim = torch.transpose(torch.asarray(validate_data_sim), 0, 2)
validate_results = torch.asarray(validate_results)

# test_data = torch.asarray(test_data).T
# test_data_sim = torch.transpose(torch.asarray(test_data_sim), 0, 2)
# test_results = torch.asarray(test_results)

true_samples = 0
false_samples = 0
for result in train_results:
    if result == 1:
        true_samples += 1
    else:
        false_samples += 1
class_weights_train = [len(train_results) / (true_samples * 2), len(train_results) / (false_samples * 2)]

true_samples = 0
false_samples = 0
for result in validate_results:
    if result == 1:
        true_samples += 1
    else:
        false_samples += 1
class_weights_validate = [len(validate_results) / (true_samples * 2), len(validate_results) / (false_samples * 2)]


def forward(x, s, w):
    ui = (x[0] * (1 - (w[0] / 100)) + x[1] * (w[0] / 100))
    uj = (s[0] * (1 - (w[0] / 100)) + s[1] * (w[0] / 100))
    sij = (s[2] * (1 - (w[2] / 100)) + s[3] * (w[2] / 100))

    p = torch.sigmoid(ui - w[1]) * torch.prod(1 - torch.sigmoid(sij - w[3]) * torch.sigmoid(uj - ui), dim=0)
    return p


def loss_fun(y, y_pred, c_w):
    eps = 1e-20
    return -torch.sum(c_w[0] * (y * (torch.log(y_pred + eps))) +
                      (c_w[1] * (1 - y) * (torch.log(1 - y_pred + eps)))) / len(y)


# t_a_r, q_t, s_c_r, s_t
weights = torch.tensor([5, 55, 90, 20], requires_grad=True, dtype=torch.double)

loss_BGD = []
loss_val = []

change = 0
momentum = 0.9
lr = torch.asarray([0.01, 0.01, 0.01, 0.01])
best_loss = 1e10
# best_weights = torch.tensor([5.0190735553839785, 54.9600518080562, 90.79135620039723, 15.250126458043551])
best_epoch = 0
eps = 1e-20
for i in range(10000):
    pred = forward(train_data, train_data_sim, weights)
    loss = loss_fun(train_results, pred, class_weights_train)
    loss_BGD.append(loss.item())
    loss.backward()

    pred_validate = forward(validate_data, validate_data_sim, weights)
    loss_validate = loss_fun(validate_results, pred_validate, class_weights_validate)
    loss_val.append(loss_validate.item())

    print(i, loss.item(), loss_validate.item(),
          [weights[0].item(), weights[1].item(), weights[2].item(), weights[3].item()])

    if loss_validate.item() < best_loss and (best_loss - loss_validate.item()) > 0.00001:
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

plt.plot(loss_BGD, label="Train loss")
plt.plot(loss_val, label="Validation loss")
plt.axvline(x=best_epoch, color="tab:red", ls="--", label="Best validation loss")
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.legend()
plt.title("Batch gradient descent")
plt.savefig("Log_reg.pdf", format="pdf", bbox_inches="tight")
plt.show()

x_axis = np.arange(best_epoch-200,best_epoch+201,1)
plt.plot(x_axis,loss_val[best_epoch-200:best_epoch+201], label="Validation loss", color="tab:orange")
plt.axvline(x=best_epoch, color="tab:red", ls="--", label="Best validation loss")
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.legend()
plt.title("Validation loss close-up")
plt.savefig("Log_reg_close-up.pdf", format="pdf", bbox_inches="tight")
plt.show()

# true_samples = 0
# false_samples = 0
# for result in test_results:
#     if result == 1:
#         true_samples += 1
#     else:
#         false_samples += 1
# class_weights_test = [len(test_results) / (true_samples * 2), len(test_results) / (false_samples * 2)]
#
# pred = forward(test_data, test_data_sim, best_weights)
# loss = loss_fun(test_results, pred, class_weights_test)
#
# print("TEST LOSS:", loss.item())
#
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
true_num = 0
false_num = 0
class_threshold = 0.5
positive = 0
negative = 0
# with torch.no_grad():
#     for i, p in enumerate(pred):
#         if p >= class_threshold and test_results[i].item() == 1:
#             positive += 1
#             true_positive += 1
#         elif p < class_threshold and test_results[i].item() == 0:
#             negative += 1
#             true_negative += 1
#         elif p >= class_threshold and test_results[i].item() == 0:
#             positive += 1
#             false_positive += 1
#         else:
#             negative += 1
#             false_negative += 1
#
#     if true_positive == 0:
#         f1 = 0
#     else:
#         precision = true_positive / (true_positive + false_positive)
#         recall = true_positive / (true_positive + false_negative)
#         f1 = (2 * precision * recall) / (precision + recall)
#
# print("TEST F1 Score:", f1)

cross_validation = torch.tensor(np.concatenate((train_data, validate_data), axis=1))
cross_validation_sim = torch.tensor(np.concatenate((train_data_sim, validate_data_sim), axis=2))
cross_validation_results = torch.tensor(np.concatenate((train_results, validate_results)))

pred = forward(cross_validation, cross_validation_sim, best_weights)

with torch.no_grad():
    for i, p in enumerate(pred):
        if p >= class_threshold and cross_validation_results[i].item() == 1:
            positive += 1
            true_positive += 1
        elif p < class_threshold and cross_validation_results[i].item() == 0:
            negative += 1
            true_negative += 1
        elif p >= class_threshold and cross_validation_results[i].item() == 0:
            positive += 1
            false_positive += 1
        else:
            negative += 1
            false_negative += 1

    if true_positive == 0:
        f1 = 0
    else:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = (2 * precision * recall) / (precision + recall)

print("CROSS-VALIDATION F1 Score:", f1)
