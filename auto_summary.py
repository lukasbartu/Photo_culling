__author__ = 'Lukáš Bartůněk'

import torch
import json
import operator


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


def load_weights():
    with open("data/auto_summary_weights.json", "r") as read_file:
        weights = json.load(read_file)
    return torch.asarray(weights, requires_grad=True)


def save_weights(weights):
    with open("data/auto_summary_weights.json", "w") as write_file:
        json.dump(weights, write_file, indent=2)


def format_data(s_file,q_file,nbrs):
    with open(q_file) as f:
        q_data = json.load(f)
    q_list = sorted(q_data, key=operator.itemgetter("id"))
    with open(s_file) as f:
        s_data = json.load(f)
    s_list = sorted(s_data, key=operator.itemgetter("first_id"))

    last_id = - 1
    data_sim = []
    for s in s_list:
        if not s["first_id"] == last_id:
            data_sim.append([])
            spaces = nbrs - s["first_id"]
            while spaces > 0:
                data_sim[s["first_id"]].append([0, 0, 0, 0])
                spaces -= 1
            if not last_id == -1:
                while len(data_sim[last_id]) < nbrs*2:
                    data_sim[last_id].append([0, 0, 0, 0])
            last_id = s["first_id"]
        second_img_score = [0,0]
        for q in q_list:
            if q["id"] == s["second_id"]:
                second_img_score = [q["aesthetic_quality"], q["technical_quality"]]
        data_sim[last_id].append(
            [second_img_score[0], second_img_score[1], s["feature_similarity_score"], s["content_similarity_score"]])
    while len(data_sim[last_id]) < nbrs*2:
        data_sim[last_id].append([0, 0, 0, 0])

    data_q = []
    for q in q_list:
        data_q.append((q["aesthetic_quality"], q["technical_quality"]))

    data_q = torch.asarray(data_q, requires_grad=True).T
    data_sim = torch.asarray(data_sim, requires_grad=True)
    data_sim = torch.transpose(torch.asarray(data_sim), 0, 2)

    return data_q, data_sim


def auto_select_summary(img_list, s_file, q_file, nbrs):
    data_quality, data_similarity = format_data(s_file, q_file, nbrs)
    weights = load_weights()
    pred = forward(data_quality, data_similarity, weights)
    summary = []
    for i, p in enumerate(pred):
        if p >= 0.5:
            summary.append(img_list[i])
    return summary


def update_parameters(summary, img_list, s_file, q_file, nbrs):
    data_quality, data_similarity = format_data(s_file, q_file, nbrs)
    weights = load_weights()
    old_weights = weights.tolist()

    results = []
    for i,img in enumerate(img_list):
        if img in summary:
            results.append(1)
        else:
            results.append(0)
    results = torch.asarray(results)
    class_weights = get_class_weights(results)

    change = 0
    momentum = 0.9
    lr = torch.asarray([0.1, 0.1, 0.5, 0.5])
    best_loss = 1e10
    best_weights = []
    for i in range(1000):
        pred = forward(data_quality, data_similarity, weights)
        loss = loss_fun(results, pred, class_weights)
        loss.backward()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = weights

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

    new_weights = (0.2 * best_weights) + (0.8 * torch.asarray(old_weights))
    save_weights(new_weights.tolist())


def get_class_weights(results):
    true_samples = 0
    false_samples = 0
    for result in results:
        if result == 1:
            true_samples += 1
        else:
            false_samples += 1
    class_weights = [len(results) / true_samples, len(results) / false_samples]
    return class_weights
