__author__ = 'Lukáš Bartůněk'

def pred_result(img):
    t = preprocessing.image.img_to_array(img)
    t = np.expand_dims(t, axis=0)
    t = preprocess_input(t)
    f = class_model.predict(t)
    f = f[0]
    f = f.tolist()
    return f

# TODO - NOT USED YET
def calculate_image_content(pth,lst,result_pth):
    content_list = []
    for i, img in enumerate(lst):
        temp = preprocessing.image.load_img(os.path.join(pth,img),color_mode='rgb', target_size=(224, 224))
        res = pred_result(temp)
        content_list+=[{"first_id": i,
                        "img": lst[i],
                        "content": res}]
    with open(os.path.join(os.getcwd(), result_pth), "w") as write_file:
        json.dump(content_list, write_file, indent=2)