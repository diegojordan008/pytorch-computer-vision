
###############################
######## VISUALZIATION ######## 

# The aim of this pyhon file is to be able to interact and show picturs and classes of dataset
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
sys.path.append('..')
from load_dataset import custom_tiny
from load_dataset import utils


# Get back 1 folder in the working directory
c = os.path.dirname(os.getcwd())
os.chdir(c)


# Instanciate: 
emotions = ["Fear", "Happiness", "Love", "Sadness", "Violence"]

for data_set in (["train", "test", "val"]): 
    data = custom_tiny.TinyData(data_set)

    # get the len 
    data_len = len(data)

    # Choose 1 random number from all: 
    random_pic = random.randint(1, data_len)

    # Get image and label 
    img = data[random_pic]["data"]
    print("random_pic: ", random_pic)
    lb_array = np.array(data[random_pic]["label"])
    lb = np.where(lb_array==1)[0][0]
    emotion = emotions[lb]
    print(lb_array)
    plt.imshow( img.permute(1, 2, 0) )
    plt.title("Dataset: " + data_set + "  -  emotion: " + emotion)
    plt.show()

