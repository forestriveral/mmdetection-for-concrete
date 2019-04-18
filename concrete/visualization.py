
import os
import re
import sys
import math
import string
import pandas as pd
import numpy as np
import mmcv
from mmdet import datasets
from mmdet.core import eval_map
# from scipy import interpolate
# from interval import Interval
# from sklearn.metrics import auc
# from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines, rcParams
# from matplotlib.patches import Polygon


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

linestyles = ['-', '--', ':', '-.', '--', '-']
font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 20,
        }
font_legend = {'family': 'Times New Roman',
               'weight': 'bold',
               'size': 15,
               }
colors = [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0),
          (0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
          (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)]


def plot_training_curve(data, figsize=(16, 16), save=False, save_path=None,
                        plot="loss"):
    plt.figure(figsize=figsize)
    plt.subplots()
    xnum = int(data['iters'] / data['interval']) * data['epochs']
    x_axis = np.arange(1, 1 + xnum)
    if plot == "loss":
        plt.plot(x_axis, np.array(data['loss']),
                 ls='-', c='#CD0000', lw=2, label="Training")
        plt.plot(x_axis, np.array(data['val']), ls='--',
                 c='#66CD00', lw=2, label="Validation")
        plt.xlim(0, xnum)
        plt.xlabel('Epochs', font)
        plt.ylabel('Loss', font)

        plt.legend(loc="upper right", prop=font_legend,
                   edgecolor='None', frameon=False,
                   labelspacing=0.2)
    if plot == "lr":
        plt.plot(x_axis, np.array(data['lr'])*10**3,
                 ls='-', c='#CD0000', lw=2)
        # plt.plot(data['epoch'], data['val_loss'], ls='--',
        #          c='#66CD00', lw=1.5, label="Validation")
        plt.xlim(0, xnum)
        plt.xlabel('Epochs', font)
        plt.ylabel('Learning rate(×10$^-3$)', font)
    if plot == "memory":
        plt.plot(x_axis, np.array(data['memory']) * 10 ** -3,
                 ls='-', c='#CD0000', lw=2)
        plt.xlim(0, xnum)
        plt.xlabel('Epochs', font)
        plt.ylabel('Using memory(×10$^3$)', font)

    # if plot in ["bbox", "mask"]:
    #     labels = [None, "50", "75", "s", "m", "l"]
    #     assert set(labels) > set(ap_types), "Invalid evaluation type!"
    #     for ap_type in ap_types:
    #         aps = []
    #         ind = labels.index(ap_type)
    #         if not ap_type:
    #             ap_type = " ".join([plot, "50-95"])
    #         else:
    #             ap_type = " ".join([plot, ap_type])
    #         # print(ind, data[plot])
    #         for i in range(data['epochs']):
    #             aps.append(data[plot][ind + i * 6])
    #         plt.plot(np.arange(1, 1 + data['epochs']), aps, ls=linestyles.pop(),
    #                  c=colors.pop(), lw=1.5, label=ap_type)
    #
    #     plt.xlim(0, data['epochs'])
    #     plt.xlabel('Epochs', font)
    #     plt.ylabel('Mean average precision', font)
    #     plt.legend(loc="lower right", prop=font_legend,
    #                edgecolor='None', frameon=False,
    #                labelspacing=0.2)

    ax = plt.gca()
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    # ax.spines['top'].set_color('none')
    # ax.spines['right'].set_color('none')
    # ax.yaxis.grid(True, which='major')

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
            "loss": [], "lr": [], "val": [], "memory": [],
            "interval": []}
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
                if label == "Epoch(val)":
                    # data["bbox"] += llist[22:28]
                    # data["mask"] += llist[-6:]
                    data["val"].append(float(llist[-1]))
                else:
                    data["lr"].append(float(strip_comma(llist[8])) / 100000)
                    data["loss"].append(float(llist[-1]))
                    data["memory"].append(int(strip_comma(llist[16])))
            # if lnum >= 10:
            # break

    # data["bbox"] = [float(strip_comma(x)) / 1000 for x in data["bbox"]]
    # data["mask"] = [float(strip_comma(x)) / 1000 for x in data["mask"]]
    assert len(data["lr"]) == len(data["loss"]) == len(data["memory"]), "Wrong length!"
    # assert len(data["bbox"]) == len(data["mask"]) == 6 * int(data["epochs"])
    # print("name: ", data["name"])
    # print("epochs: ", data["epochs"])
    # print("iters: ", data["iters"])
    # print("lr: ", data["lr"])
    # print("loss: ", data["loss"])
    # print("iters/per: ", iters_per)
    # print("bbox aps: ", data["bbox"], len(data["bbox"]))
    # print("mask aps: ", data["mask"], len(data["mask"]))
    return data


def plot_pr_curve():
    pass

