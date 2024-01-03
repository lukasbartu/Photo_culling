__author__ = 'Lukáš Bartůněk'

import torch
import json
import operator
import numpy as np
from utils import get_class_weights, format_data_sim


def forward(x, s, w):
    ui = (x[0] * (1 - (w[0]/100)) + x[1] * (w[0]/100))
    uj = (s[0] * (1 - (w[0]/100)) + s[1] * (w[0]/100))
    sij = (s[2] * (1 - (w[2]/100)) + s[3] * (w[2]/100))

    p = torch.sigmoid(ui - w[1]) * torch.prod(1 - torch.sigmoid(sij - w[3]) * torch.sigmoid(uj - ui), dim=0)
    return p


def loss_fun(y, y_pred, c_w):
    eps = 1e-20
    return -torch.sum(c_w[0] * (y * (torch.log(y_pred + eps))) +
                      (c_w[1] * (1 - y) * (torch.log(1 - y_pred + eps)))) / len(y)


def load_weights():
    with open("data/logical_approximation.json", "r") as read_file:
        weights = json.load(read_file)
    return torch.asarray(weights, requires_grad=True)


def save_weights(weights):
    with open("data/logical_approximation.json", "w") as write_file:
        json.dump(weights, write_file, indent=2)


def format_data(s_file, q_file):
    with open(q_file) as f:
        q_data = json.load(f)
    q_list = sorted(q_data, key=operator.itemgetter("id"))

    data_sim, nbrs = format_data_sim(s_file, q_file)

    pad = (20 - nbrs)
    data_sim = np.pad(array=np.asarray(data_sim), pad_width=np.asarray([(0, 0), (pad, pad), (0, 0)]))

    data_q = []
    for q in q_list:
        data_q.append((q["aesthetic_quality"], q["technical_quality"]))

    data_q = torch.asarray(data_q, requires_grad=True).T
    data_sim = torch.asarray(data_sim, requires_grad=True)
    data_sim = torch.transpose(torch.asarray(data_sim), 0, 2)

    return data_q, data_sim


def summary(lst, s_file, q_file,  output_size,  size_based):
    data_quality, data_similarity = format_data(s_file, q_file)
    weights = load_weights()  # [t_a_r, q_t, s_c_r, s_t]
    pred = forward(data_quality, data_similarity, weights)

    s = []
    threshold = 0.5
    n = 0
    with torch.no_grad():
        if size_based:
            while len(s) != output_size:
                if n == 5000:
                    break
                n += 1
                s = []
                for i, p in enumerate(pred):
                    if p >= threshold:
                        s.append(lst[i])
                if len(s) > output_size:
                    weights[1] = weights[1] * 1.11
                elif len(s) < output_size:
                    weights[1] = weights[1] * 0.9
                pred = forward(data_quality, data_similarity, weights)
        else:
            for i, p in enumerate(pred):
                if p >= threshold:
                    s.append(lst[i])
    return s


def update_parameters(s, lst, s_file, q_file):
    data_quality, data_similarity = format_data(s_file, q_file)
    weights = load_weights()
    old_weights = weights.tolist()
    best_weights = []

    results = []
    for i, img in enumerate(lst):
        if img in s:
            results.append(1)
        else:
            results.append(0)
    results = torch.asarray(results)
    class_weights = get_class_weights(results)

    change = 0
    momentum = 0.8
    lr = torch.asarray([0.1, 0.1, 0.1, 0.1])
    best_loss = 1e10
    for i in range(5000):
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

    new_weights = (0.8 * best_weights) + (0.2 * torch.asarray(old_weights))
    save_weights(new_weights.tolist())


def reset_model():
    with open("data/logical_approximation_default.json", "r") as read_file:
        weights = json.load(read_file)
    with open("data/logical_approximation.json", "w") as write_file:
        json.dump(weights, write_file, indent=2)
