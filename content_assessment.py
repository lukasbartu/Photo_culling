__author__ = 'Lukáš Bartůněk'

import os
import json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import preprocessing
from keras.applications.efficientnet_v2 import preprocess_input, EfficientNetV2B1


def pred_result(img, model):
    t = preprocessing.image.img_to_array(img)
    t = np.expand_dims(t, axis=0)
    t = preprocess_input(t)
    f = model.predict(t, verbose=0, batch_size=8)
    f = f.tolist()
    return f[0]


def calculate_content(lst, result_pth):
    if os.path.exists(result_pth):
        return
    class_model = EfficientNetV2B1(weights='data/efficientnetv2-b1.h5')
    content_list = []
    for i, img in enumerate(lst):
        temp = preprocessing.image.load_img(img, color_mode='rgb', target_size=(240, 240))
        res = pred_result(temp, class_model)
        content_list += [{"id": i,
                        "img": lst[i],
                        "content": res}]
    with open(os.path.join(os.getcwd(), result_pth), "w") as write_file:
        json.dump(content_list, write_file, indent=2)
