__author__ = 'Lukáš Bartůněk'

from utils import prepare_img_list, prepare_paths, get_class_weights
import json
import operator
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from keras.callbacks import ModelCheckpoint


def format_data(s_file, q_file, nbrs):
    with open(q_file) as f:
        q_data = json.load(f)
    q_list = sorted(q_data, key=operator.itemgetter("id"))
    with open(s_file) as f:
        s_data = json.load(f)
    s_list = sorted(s_data, key=operator.itemgetter("first_id"))

    last_id = -1
    data_sim = []
    for s in s_list:
        if not s["first_id"] == last_id:
            data_sim.append([])
            spaces = nbrs - s["first_id"]
            while spaces > 0:
                data_sim[s["first_id"]].append([0, 0, 0, 0])
                spaces -= 1
            if not last_id == -1:
                while len(data_sim[last_id]) < nbrs * 2:
                    data_sim[last_id].append([0, 0, 0, 0])
            last_id = s["first_id"]
        second_img_score = [0, 0]
        for q in q_list:
            if q["id"] == s["second_id"]:
                second_img_score = [q["aesthetic_quality"], q["technical_quality"]]
        data_sim[last_id].append(
            [second_img_score[0], second_img_score[1], s["feature_similarity_score"], s["content_similarity_score"]])
    while len(data_sim[last_id]) < nbrs * 2:
        data_sim[last_id].append([0, 0, 0, 0])

    data = []
    for i, q in enumerate(q_list):
        block = [q["aesthetic_quality"], q["technical_quality"]]
        for k, temp in enumerate(data_sim[i]):
            for t in temp:
                block.append(t)
        data.append(block)

    pad = (20 - nbrs)*4

    data = np.asarray(data)
    data = np.pad(data, ((0, 0), (pad, pad)))
    data =  data.astype('float32')
    return data


def summary(lst, s_file, q_file, nbrs):
    data = format_data(s_file, q_file, nbrs)

    model = keras.models.load_model("data/best_nn_model.keras")

    pred = model.predict(data, verbose=False)

    s = []
    for i, p in enumerate(pred):
        if p >= 0.5:
            s.append(lst[i])
    return s


def update_model(s, lst, s_file, q_file, nbrs):
    data = format_data(s_file, q_file, nbrs)

    results = []
    for i, img in enumerate(lst):
        if img in s:
            results.append([1])
        else:
            results.append([0])
    results = np.asarray(results)
    results = results.astype("float32")

    model = keras.models.load_model("data/best_nn_model.keras")

    if len(s) == 0:
        return
    class_weights = get_class_weights(results)
    best_checkpoint_path = "best_checkpoint_nn"
    save_best_model = ModelCheckpoint(best_checkpoint_path, monitor='f1_score',
                                      save_best_only=True, save_weights_only=True, mode="max")

    model.fit(data, results, epochs=100, class_weight={0: class_weights[1], 1: class_weights[0]},
              callbacks=[save_best_model], verbose=False)

    model.save("data/best_nn_model.keras")


path = "/images/Ples/fotokoutek"

abs_path, s_pth, q_pth, _ = prepare_paths(path)
img_list, _ = prepare_img_list(abs_path)

s = summary(img_list, s_pth, q_pth, 20)

update_model(s, img_list, s_pth, q_pth, 20)

