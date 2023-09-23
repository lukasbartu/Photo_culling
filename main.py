__author__ = 'Lukáš Bartůněk'

# Select a subset of photographs from current directory to represent the sequence.
# Considers both technical and aesthetic quality while trying to eliminate duplicity.
#
# SIFT similarity is edited code taken from https://github.com/adumrewal/SIFTImageSimilarity/blob/master/SIFTSimilarityInteractive.ipynb
# NIMA image quality is edited code taken from https://github.com/yunxiaoshi/Neural-IMage-Assessment
# argument parser code taken from https://gitlab.fel.cvut.cz/kybicjan/fotoselektor
#
# DONE - Familiarize yourself with existing tools and approaches in this domain
#           and suggest a suitable architecture for the photo selection tool.
# DONE - Test existing approaches for evaluating image quality, image similarity and  TODO - image content.
# TODO - Evaluate several methods for combining extracted image information for photo selection.
# DONE - Implement an easy-to-use software tool for photo  selection, allowing the user to set basic parameters of the process.
# TODO - Experimentally evaluate the performance of the developed software by comparing it with other existing tools,
#           by comparing it with human performance, or by evaluating user satisfaction.
# TODO - [Optional] Add the ability to read and write image metadata for easier integration with standard tools.
# TODO - [Optional] Create a graphical user interface allowing to interactively examine and alter the selection process.
# TODO - [Optional] Improve the selection process by learning from a (possibly large) dataset of manually selected photographs.

import os
import time
import argparse
import cv2
import json
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from PIL import Image
import webbrowser
import operator
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.applications.vgg16 import preprocess_input,decode_predictions,VGG16
from keras import preprocessing

class NIMA(nn.Module):

    """Neural IMage Assessment model by Google"""
    def __init__(self, base_model, num_classes=10):
        super(NIMA, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        out_f = self.features(x)
        out = out_f.view(out_f.size(0), -1)
        out = self.classifier(out)
        return out_f,out

def prepare_paths(pth,abs_p):
    if abs_p:
        abs_pth = pth
    else:
        abs_pth = os.getcwd() + pth  # absolute file of image directory
    folder_name = abs_pth.split("/")[-1]  # name of most nested folder for better naming of resulting files
    sim_path = "image_similarities_" + folder_name + ".json"  # file to save result of precalculating similarities
    q_path = "image_quality_" + folder_name + ".json"  # file to save result of quality evaluation
    content_path = "image_content_" + folder_name + ".json" # file to save result of image content evaluation
    return abs_pth, sim_path, q_path, content_path

def prepare_img_list(pth):
    # assuming the files are numbered based on the sequence they have been taken in
    img_list = []  # list of image file names to process
    for path in os.scandir(pth):
        if path.is_file():
            if path.name.endswith(".jpg"):
                img_list += [path.name]
    img_num = len(img_list)
    img_list.sort()
    return img_list,img_num

def compute_SIFT(image):
    return sift.detectAndCompute(image, None)

def image_resize(image):
    max_d = 1024
    height,width,channel = image.shape
    aspect_ratio = width/height
    if aspect_ratio < 1:
        new_size = (int(max_d*aspect_ratio),max_d)
    else:
        new_size = (max_d,int(max_d/aspect_ratio))
    image = cv2.resize(image,new_size)
    return image

def calculate_matches(des1, des2):
    matches = bf.knnMatch(des1, des2, k=2)
    top_results1 = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            top_results1.append([m])

    matches = bf.knnMatch(des2, des1, k=2)
    top_results2 = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            top_results2.append([m])

    top_results = []
    for match1 in top_results1:
        match1_query_index = match1[0].queryIdx
        match1_train_index = match1[0].trainIdx

        for match2 in top_results2:
            match2_query_index = match2[0].queryIdx
            match2_train_index = match2[0].trainIdx

            if (match1_query_index == match2_train_index) and (match1_train_index == match2_query_index):
                top_results.append(match1)
    return top_results

def calculate_score(matches,keypoint1,keypoint2):
    return 100 * (matches/min(keypoint1,keypoint2))

def calculate_similarities(pth,lst,result_pth,num,nbrs):
    features={} # keypoints and descriptors
    sim_list = [] # list to store results
    for i in range(num):
        img = image_resize(cv2.imread(os.path.join(pth,lst[i])))
        keypoints, descriptors = compute_SIFT(img)
        features[i] = (keypoints,descriptors)
    for i in range(num):
        keypoints_i, descriptors_i = features[i]
        for j in range(max(0,i-nbrs),min(num,i+nbrs+1)):
            if i==j:
                continue
            keypoints_j, descriptors_j = features[j]
            matches = calculate_matches(descriptors_i, descriptors_j)
            score = calculate_score(len(matches), len(keypoints_i), len(keypoints_j))
            sim_list+=[{"first_id": i,
                        "first_img": lst[i],
                        "second_img": lst[j],
                        "similarity_score": score}]
    with open(os.path.join(os.getcwd(), result_pth), "w") as write_file:
        json.dump(sim_list, write_file, indent=2)

def prepare_model(model_pth):
    base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model = NIMA(base_model)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), model_pth), map_location=torch.device('cpu')))
    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device

# TODO - separate technical and aesthetics quality assessment
def calculate_qualities(pth, lst, result_pth, model,device):
    q_list = [] # list to store results
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mean, std = 0.0, 0.0
    i = 0
    for img in lst:
        im = Image.open(os.path.join(pth, str(img))).convert('RGB')
        imt = test_transform(im)
        imt = imt.unsqueeze(dim=0)
        imt = imt.to(device)
        with torch.no_grad():
            out_f, out_class = model(imt)
        out_class = out_class.view(10, 1)
        for j, e in enumerate(out_class, 1):
            mean += j * e
        for k, e in enumerate(out_class, 1):
            std += e * (k - mean) ** 2
        std = std ** 0.5
        q_list += [{"first_img": lst[i],
                    "quality_mean": float(mean),
                    "quality_std": float(std)}]
        i += 1
        mean, std = 0.0, 0.0
    with open(os.path.join(os.getcwd(), result_pth), "w") as write_file:
        json.dump(q_list, write_file, indent=2)

# TODO - implement actual selection, for now selects top mean_quality photos without similar photos occurring
def select_summary(sim_pth,q_pth,percent,num,q_t,dir_pth):
    select_num = int(num*(percent/100))
    with open(sim_pth) as json_file:
        sim_data = json.load(json_file)
    with open(q_pth) as json_file:
        q_data = json.load(json_file)
    q_sorted_lst = sorted(q_data,key=operator.itemgetter("quality_mean"))
    q_sorted_lst.reverse()
    q_top_list = []
    i = 0
    for temp1 in q_sorted_lst:
        if i >= select_num:
            break
        img = temp1["first_img"]
        in_top_list = False
        for temp2 in sim_data:
            if img == temp2["first_img"] and temp2["second_img"] in q_top_list and temp2["similarity_score"] > q_t:
                in_top_list = True
                break
        if not in_top_list:
            q_top_list.append(img)
            i+=1
    for t in q_top_list:
        webbrowser.open(os.path.join(dir_pth,t))
    return q_top_list

def simple_calculate_q(abs_pth,img_list,q_path,model_path):
    tic = time.perf_counter()
    print("Preparing model")
    mdl, dvc = prepare_model(model_pth=model_path)
    print("Successfully loaded model")
    calculate_qualities(pth=abs_pth, lst=img_list, result_pth=q_path, model=mdl, device=dvc)
    print("Quality Calculated")
    toc = time.perf_counter()
    print(f"Process took: {toc - tic:0.2f} s")

def simple_calculate_s(abs_pth,img_list,sim_path,img_num,num_nbrs):
    tic = time.perf_counter()
    print("Images have been loaded")
    calculate_similarities(pth=abs_pth, lst=img_list, result_pth=sim_path, num=img_num, nbrs=num_nbrs)
    print("Similarities calculated")
    toc = time.perf_counter()
    print(f"Process took: {toc - tic:0.2f} s")

def simple_create_sum(sim_path,q_path,percent,img_num,q_t,abs_pth):
    tic = time.perf_counter()
    print("Selecting summary of photos")
    summary = select_summary(sim_pth=sim_path, q_pth=q_path, percent=percent, num=img_num, q_t=q_t,
                   dir_pth=abs_pth)
    print("Summary:",summary)
    toc = time.perf_counter()
    print(f"Process took: {toc - tic:0.2f} s")

class_model = VGG16(weights='imagenet') # TODO - find more suitable CNN with more classes
sift = cv2.SIFT_create(1000) # SIFT algorithm with number of keypoints
bf = cv2.BFMatcher() # keypoint matcher

def main(arg_list):
    abs_pth,sim_path,q_path,c_path = prepare_paths(arg_list.directory,abs_p=False)
    img_list,img_num = prepare_img_list(abs_pth)

    if arg_list.calculate_quality:
        simple_calculate_q(abs_pth, img_list, q_path, arg_list.model_path)
    if arg_list.calculate_similarity:
        simple_calculate_s(abs_pth, img_list, sim_path, img_num, arg_list.number_of_neighbors)
    if arg_list.select_photos:
        simple_create_sum(sim_path, q_path, arg_list.percentage, img_num, arg_list.similarity_threshold, abs_pth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a subset of photographs from current directory. Considers both technical and aesthetic quality as well as similarity between photographs.",
                                     epilog='Lukáš Bartůněk, 2023')
    parser.add_argument('-q','--calculate_quality',help='precalculate aesthetic and technical quality',action='store_true')
    parser.add_argument('-s','--calculate_similarity',help='precalculate similarity between images',action='store_true')
    parser.add_argument('-e','--select_photos',help='select a subset of images based on the precalculated quality and similarities',action='store_true')
    parser.add_argument('-n','--number_of_neighbors',help='how many images before and after to consider for similarity calculation',type=int,default=5)
    parser.add_argument('-t','--technical_weight',help='how much weight (0-100) to give to the technical quality. The remainder is aesthetic quality',type=int,default=50)
    parser.add_argument('-p','--percentage',help='how many percent (0-100) of the original images to select in the first round.',type=int,default=10)
    parser.add_argument('-q_t','--similarity_threshold',help='threshold on SIFT similarity to consider images as similar to be pruned.',type=int,default=10)
    parser.add_argument('-model','--model_path',help='path to model for quality assessment',default= "model.pth" )
    parser.add_argument( '-dir','--directory',help='directory containing photographs',default= "/images/Ples/fotokoutek" )
    args=parser.parse_args()
    main(args)