__author__ = 'Lukáš Bartůněk'

import os,json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import preprocessing
import numpy as np
from keras.applications.vgg16 import preprocess_input,VGG16

class_model = VGG16(weights='imagenet')

def pred_result(img):
    t = preprocessing.image.img_to_array(img)
    t = np.expand_dims(t, axis=0)
    t = preprocess_input(t)
    f = class_model.predict(t)
    f = f.tolist()
    return f[0]

def calculate_content(pth,lst,result_pth):
    if os.path.exists(result_pth):
        return
    content_list = []
    for i, img in enumerate(lst):
        temp = preprocessing.image.load_img(os.path.join(pth,img),color_mode='rgb', target_size=(224, 224))
        res = pred_result(temp)
        content_list+=[{"id": i,
                        "img": lst[i],
                        "content": res}]
    with open(os.path.join(os.getcwd(), result_pth), "w") as write_file:
        json.dump(content_list, write_file, indent=2)
