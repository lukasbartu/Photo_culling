__author__ = 'Lukáš Bartůněk'

from utils import get_class_weights, format_data_sim
import json
import operator
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras


def format_data(s_file, q_file):
    with open(q_file) as f:
        q_data = json.load(f)
    q_list = sorted(q_data, key=operator.itemgetter("id"))

    data_sim, nbrs = format_data_sim(s_file, q_file)

    data = []
    for i, q in enumerate(q_list):
        block = [q["aesthetic_quality"], q["technical_quality"]]
        for k, temp in enumerate(data_sim[i]):
            for t in temp:
                block.append(t)
        data.append(block)

    pad = (20 - nbrs)*4
    data = np.pad(array=np.asarray(data), pad_width=np.asarray([(0, 0), (pad, pad)]))
    data = data.astype('float32')
    return data


def summary(lst, s_file, q_file):
    data = format_data(s_file, q_file)

    model = keras.models.load_model("data/best_nn_model.keras")

    pred = model.predict(data, verbose=False)

    s = []
    for i, p in enumerate(pred):
        if p >= 0.5:
            s.append(lst[i])
    return s


def update_model(s, lst, s_file, q_file):
    data = format_data(s_file, q_file)

    results = []
    for i, img in enumerate(lst):
        if img in s:
            results.append([1])
        else:
            results.append([0])
    results = np.asarray(results)
    results = results.astype("float32")

    model = keras.models.load_model('data/best_nn_model.keras')

    if len(s) == 0:
        return
    class_weights = get_class_weights(results)

    model.fit(data, results, epochs=100, class_weight={0: class_weights[1], 1: class_weights[0]},
              verbose=False, workers=-1)

    model.save('data/best_nn_model.keras')

