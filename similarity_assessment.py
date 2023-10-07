__author__ = 'Lukáš Bartůněk'

import cv2,os,json

sift = cv2.SIFT_create(1000) # SIFT algorithm with number of keypoints
bf = cv2.BFMatcher() # keypoint matcher

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
    if os.path.exists(result_pth):
        return
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
                        "second_id": j,
                        "first_img": lst[i],
                        "second_img": lst[j],
                        "similarity_score": score}]
    with open(os.path.join(os.getcwd(), result_pth), "w") as write_file:
        json.dump(sim_list, write_file, indent=2)
