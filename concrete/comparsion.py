import os
import sys
import numpy as np
import pandas as pd
import pickle
import concrete.visualization as visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import rcParams
from concrete.visualization import \
    linestyles, font, font_legend, colors, colours

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library


def read_logs(logs, obj="loss", save=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    x_axis = np.arange(1, 85)
    for i, log in enumerate(logs):
        label = log.split("/")[-2].split("_")[-1]
        data = visualization.read_log(log)
        plt.plot(x_axis, np.array(data[obj]),
                 ls=linestyles[i], c=colours[i], lw=2, label=label)

        plt.xlim(0, 84)
        plt.xlabel('Epochs', font)
        plt.ylabel('Loss', font)
        ax.xaxis.set_major_locator(MultipleLocator(7 * 2))
        x_label = 2 * np.arange(0, (12 / 2) + 1)
        x_label = visualization.convert_to_str(x_label)
        ax.set_xticklabels(x_label)
        plt.legend(loc="upper right", prop=font_legend,
                   edgecolor='None', frameon=False,
                   labelspacing=0.2)

    ax = plt.gca()
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(20) for label in labels]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(" Save done![{}]".format(save))
    plt.show()


def read_coco_eval(evals, save=None):
    cocos = []
    for e in evals:
        f = os.path.join(e, "coco_val.csv")
        # print(f)
        coco = pd.read_csv(f)
        if coco.columns.size != 8:
            added_heads = ['segm_mAP', 'segm_50', 'segm_75']
            for h in added_heads:
                coco[h] = None
        cocos.append(coco)
    result = pd.concat(cocos)
    if save:
        result.to_csv(save, index=0)

    return result


def read_targets(path, types):
    targets = []
    labels = []
    for p in path:
        labels.append(p.split("/")[-1].split("_")[-1])
        file_name = "_".join(["voc", types, "0.5", "0.75"]) + ".pkl"
        file_path = os.path.join(p, file_name)
        if not os.path.exists(file_path):
            file_name_0 = "_".join(["voc", types, "0.5", "0.75", "0.9"]) + ".pkl"
            file_path = os.path.join(p, file_name_0)
        with open(file_path, "rb") as f:
            target = pickle.load(f)
            targets.append(target)

    assert len(targets) == len(labels)
    return targets, labels


def parse_targets(targets, cla, curve):
    x, y, z = [], [], []
    for target in targets:
        if curve == "roc":
            fpr = target["fprs"]
            tpr = target["tprs"]
            aucs = target["aucs"]
            aucs, fpr, tpr = aucs[cla], fpr[cla], tpr[cla]
            x.append(aucs), y.append(fpr), z.append(tpr)
        if curve == "pr":
            precision = target["precisions"]
            recall = target["recalls"]
            maps = target["maps"]
            aps, pre, rec = maps[cla], precision[cla], recall[cla]
            x.append(aps), y.append(pre), z.append(rec)

    return x, y, z


def plot_voc_comparsion(targets, labels, thresh, types, curve,
                        cla="crack", save=False, root="."):
    x, y, z = parse_targets(targets, cla, curve)
    thresh_id = 0 if thresh == "0.5" else 1
    _, ax = plt.subplots(1, figsize=(8, 8))

    if curve == "pr":
        for i, label in enumerate(labels):
            _ = ax.plot(y[i][thresh_id], z[i][thresh_id],
                        ls=linestyles[i], c=colors[i], lw=2.5,
                        label="{}(AP={:.3f})".format(label, x[i][thresh_id]))
        ax.set_title("Precision-Recall Curve", font)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 1.05)
        plt.xlabel("Recall", font)
        plt.ylabel("Precision", font)
        plt.legend(loc="lower left", prop=font_legend,
                   edgecolor='None', frameon=False,
                   labelspacing=0.4)
    if curve == "roc":
        for i, label in enumerate(labels):
            _ = ax.plot(y[i][thresh_id], z[i][thresh_id],
                        ls=linestyles[i], c=colors[i], lw=2.5,
                        label="{}(AUC={:.3f})".format(label, x[i][thresh_id]))
        # Standard line
        ax.plot([0, 1], [0, 1], ls='--', c='#778899', lw=2.5, label="Standard line")
        # Set plot title
        ax.set_title("ROC Curve", font)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, 1.0)
        plt.xlabel("False Positive Rate (FPR)", font)
        plt.ylabel("True Positive Rate (TPR)", font)
        plt.legend(loc="lower right", prop=font_legend,
                   edgecolor='None', frameon=False,
                   labelspacing=0.4)

    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(20) for label in labels]
    ax.set_xticklabels(('0', '0.2', '0.4', '0.6', '0.8', '1.0'))
    ax.set_yticklabels(('', '0.2', '0.4', '0.6', '0.8', '1.0'))
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'

    if save:
        filename = "_".join([types, curve, str(thresh)]) + ".png"
        fpath = os.path.join(root, filename)
        plt.savefig(fpath, dpi=300, bbox_inches='tight')
        print("Save done![{}]".format(fpath))

    plt.show()