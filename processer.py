from __future__ import print_function
import os,time,cv2, sys, math
import numpy as np
from utils import utils, helpers

def arr2key(arr):
    return np.dot(arr, [65536,256,1])
dataset = "CamVid"
processed_dataset = "CamVid_people"

class_names_list, label_values = helpers.get_label_info(os.path.join(dataset, "class_dict.csv"))
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=dataset)

people_images = []
#directory = train_output_names
directory = test_output_names
#directory = val_output_names
n = len(directory)

for k in range(n):
    print(k,"/",len(directory))
    output_image = utils.load_image(directory[k])
    for i in range(len(output_image)):
        for j in range(len(output_image[0])):
            #print(i,j)
            #print(output_image[i][j])
            #print(arr2key(output_image[i][j]), [64,64,0])
            if arr2key(output_image[i][j]) == 4210688:
                output_image[i][j] = np.array([255,255,255])
            else:
                output_image[i][j] = np.array([0,0,0])

    filename = directory[k].replace("CamVid","CamVid_people")
    print(filename)
    cv2.imwrite(filename, output_image)
    



