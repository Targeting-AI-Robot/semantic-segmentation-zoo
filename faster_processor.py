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
print(train_output_names[0])

people_images = []
#file_names = train_output_names
#file_names = test_output_names
file_names = val_output_names
n = len(file_names)

filter = np.zeros((720,960,3))
for i in range(720):
    for j in range(960):
        filter[i][j] = np.array([65536,256,1])

for k in range(n):
    print(k,"/",len(file_names))
    output_image = utils.load_image(file_names[k])
    det = np.sum(np.multiply(filter, output_image), axis=-1)
    output_image[:,:] = np.array([0,0,0])
    output_image[det == 4210688] = np.array([255,255,255])
    output_image[det == 32960] = np.array([255,255,255])
    output_image[det == 12615744] = np.array([255,255,255])

    filename = file_names[k].replace("CamVid","CamVid_people")
    print(filename)
    cv2.imwrite(filename, output_image)
    



