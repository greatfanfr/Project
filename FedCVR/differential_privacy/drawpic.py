import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np

def plot_results(exp_name: str, train_loss: [], test_accuracy: [],title:str):
    matplotlib.rcdefaults()
    markers = ['o', '+', '*', '>', 's', 'd', 'v', '^']
    colors = ['r', 'b', 'y', 'g', 'o', 'p', 'k']
    marker_style = dict(markersize=25, markerfacecoloralt='tab:red', markeredgewidth=2,
                        markevery=max(1, int(len(train_loss[0]) / 5)), fillstyle='none')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15.5, 8))
    ax[0].plot(train_loss[0], lw=1, marker=markers[0], color=colors[0], label='FedAdam', **marker_style)
    ax[0].plot(train_loss[1], lw=1, marker=markers[1], color=colors[1], label='FedAvg', **marker_style)
    ax[0].plot(train_loss[2], lw=1, marker=markers[2], color=colors[2], label='SCAFFOLD', **marker_style)
    ax[0].plot(train_loss[3], lw=1, marker=markers[3], color=colors[3], label='FedProx', **marker_style)
    ax[0].set_ylabel('Train Loss', fontsize=25, color='k')
    ax[0].set_xlabel('Round $r$', fontsize=25)
    ax[0].set_xlim([0, 300])
    ax[0].grid(linestyle=':', alpha=0.2, lw=2)
    ax[0].tick_params(axis='both', labelsize=15)
    ax[0].legend(fontsize="25", edgecolor='k')

    ax[1].plot(test_accuracy[0], lw=1, marker=markers[0], color=colors[0], label='FedAdam',**marker_style)
    ax[1].plot(test_accuracy[1], lw=1, marker=markers[1], color=colors[1], label='FedAvg', **marker_style)
    ax[1].plot(test_accuracy[2], lw=1, marker=markers[2], color=colors[2], label='SCAFFOLD',**marker_style)
    ax[1].plot(test_accuracy[3], lw=1, marker=markers[3], color=colors[3], label='FedProx', **marker_style)
    ax[1].set_ylabel('Accuracy', fontsize=25, color='k')
    ax[1].set_xlabel('Round $r$', fontsize=25)
    ax[1].set_xlim([0, 300])
    ax[1].set_ylim([0, 1])
    ax[1].grid(linestyle=':', alpha=0.2, lw=2)
    ax[1].tick_params(axis='both', labelsize=15)
    ax[1].legend(fontsize="25", edgecolor='k')
    fig.suptitle(title, fontsize=30, ha='center')
    fig.subplots_adjust(
        top=0.9,
        bottom=0.080,
        left=0.06,
        right=0.9,
        hspace=0.1,
        wspace=0.15
    )

    plt.savefig(Path("D://MLandDL//Project//FedCVR//results//", f"{exp_name}_results.pdf"), dpi=200)


#### read the data
trainLoss=[]
testacc=[]

root_path="D://MLandDL//Project//FedCVR//results//mnist-nodp-later//"

testacc.append(np.loadtxt(Path(root_path+"FedAdam_testacc.csv"),delimiter=','))
trainLoss.append(np.loadtxt(Path(root_path+"FedAdam_trainloss.csv"),delimiter=','))
testacc.append(np.loadtxt(Path(root_path+"FedAvg_testacc.csv"),delimiter=','))
trainLoss.append(np.loadtxt(Path(root_path+"FedAvg_trainloss.csv"),delimiter=','))
testacc.append(np.loadtxt(Path(root_path+"scaffold_testacc.csv"),delimiter=','))
trainLoss.append(np.loadtxt(Path(root_path+"scaffold_trainloss.csv"),delimiter=','))
testacc.append(np.loadtxt(Path(root_path+"FedProx_testacc.csv"),delimiter=','))
trainLoss.append(np.loadtxt(Path(root_path+"FedProx_trainloss.csv"),delimiter=','))
plot_results("mnist-nodp-later",trainLoss,testacc,"mnist-no-dp")
