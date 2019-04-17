
import os
import sys
# import cv2
# import random
# import copy
# import json
# import colorsys
# import collections
import math
import re
import string
import pandas as pd
import numpy as np
# from scipy import interpolate
# from interval import Interval
# from sklearn.metrics import auc
# from skimage.measure import find_contours
import matplotlib.pyplot as plt
# from matplotlib import patches,  lines, rcParams
# from matplotlib.patches import Polygon


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

linestyles = ['-', '--', ':', '-.', '-', '--']
font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 20,
        }
font_legend = {'family': 'Times New Roman',
               'weight': 'bold',
               'size': 15,
               }
colors = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
          (0.0, 1.0, 0.0), (1.0, 0.0, 1.0),
          (1.0, 1.0, 0.0), (0.0, 0.0, 0.0)]


def plot_training_curve(data, figsize=(16, 16), save=False, save_path=None):
    plt.figure(figsize=figsize)
    plt.subplots()
    xnum = int(data['iters'] / data['interval']) * data['epochs']
    plt.plot(np.arange(1, 1 + xnum),
             np.array(data['loss']), ls='-', c='#CD0000', lw=1.5, label="Training")
    # plt.plot(data['epoch'], data['val_loss'], ls='--',
    #          c='#66CD00', lw=1.5, label="Validation")
    plt.xlim(0, xnum)
    plt.xlabel('Epochs', font)
    plt.ylabel('Loss', font)

    ax = plt.gca()
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # ax.spines['top'].set_color('none')
    # ax.spines['right'].set_color('none')
    # ax.yaxis.grid(True, which='major')

    plt.legend(loc="upper right", prop=font_legend,
               edgecolor='None', frameon=False,
               labelspacing=0.2)
    if save:
        assert save_path, "Path to save must be provided!"
        plt.margins(0, 0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(" Save done! ")
    plt.show()


def strip_comma(text):
    temp = []
    for c in text:
        if c not in string.punctuation:
            temp.append(c)
    new_text = ''.join(temp)
    # print(new_text)
    return new_text


def read_log(log):
    lnum = 0
    data = {"name": None, "epochs": None, "iters": None,
            "loss": [], "lr": [], "bbox": [], "mask": [],
            "memory": [], "interval": []}
    with open(log, 'r+') as log_read:
        while True:
            lnum += 1
            line = log_read.readline()
            if not line:
                break
            llist = line.split()
            label = llist[5]
            if label == "Start":
                data["name"] = llist[-1].split("/")[-1]
            elif label == "workflow:":
                data["epochs"] = int(llist[-2])
            # print(line)
            else:
                if lnum == 3:
                    step = re.split('\D{1}', llist[6])
                    # print(step)
                    data["interval"], data["iters"] = int(step[3]), int(step[4])
                iters_per = math.ceil(data["iters"] / data["interval"])
                # print(llist)
                if (lnum - 2) % iters_per == 0:
                    data["bbox"] += llist[22:28]
                    data["mask"] += llist[-6:]
                else:
                    data["lr"].append(float(strip_comma(llist[8])) / 100000)
                    data["loss"].append(float(llist[-1]))
                    data["memory"].append(int(strip_comma(llist[16])))
            # if lnum >= 10:
            # break

    data["bbox"] = [float(strip_comma(x)) / 1000 for x in data["bbox"]]
    data["mask"] = [float(strip_comma(x)) / 1000 for x in data["mask"]]
    assert len(data["lr"]) == len(data["loss"]), "Wrong length!"
    assert len(data["bbox"]) == len(data["mask"]) == 6 * int(data["epochs"])
    # print("name: ", data["name"])
    # print("epochs: ", data["epochs"])
    # print("iters: ", data["iters"])
    # print("lr: ", data["lr"])
    # print("loss: ", data["loss"])
    # print("iters/per: ", iters_per)
    # print("bbox aps: ", data["bbox"], len(data["bbox"]))
    # print("mask aps: ", data["mask"], len(data["mask"]))
    return data

