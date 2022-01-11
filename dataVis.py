import matplotlib.pyplot as plt
import numpy as np


def processAllLines():
    lines = []
    with open("model_trainTime_minLoss.txt") as f:
        lines = f.readlines()
    retLines = []
    for line in lines:
        lineArr = lineProcessor(line)
        if lineArr == []: continue
        retLines.append(lineArr)
    return np.array(retLines)


def lineProcessor(line):
    if "Train Time" in line: return []
    model,time,loss = line.split(" ")
    time = float(time[:-1])
    loss = float(loss)
    hFactor = int(model.split("hFactor-")[1][:-1])
    L = int(model.split("_L-")[1].split("_hFactor")[0])
    return [L, hFactor, loss, time]

def plot_L_H_time(data):
    fig = plt.figure()
    plt.scatter(data[:,0], data[:,1], data[:,3])
    plt.xlabel("Layers")
    plt.ylabel("h-Factor")
    plt.title("Total Time for Layer and h-Factor MLMC")
    # plt.show()
    fig.savefig("./graphs/layers_hFactor_time.png")

def plot_L_H_minLoss(data):
    fig = plt.figure()
    plt.scatter(data[:,0], data[:,1], data[:,2])
    plt.xlabel("Layers")
    plt.ylabel("h-Factor")
    plt.title("Minimum Loss for Layer and h-Factor MLMC")
    # plt.show()
    fig.savefig("./graphs/layers_hFactor_loss.png")


def plot_L_H_minLoss_time(data):
    fig = plt.figure()
    cm = plt.cm.get_cmap('RdYlBu')

    plt.scatter(data[:,0], data[:,1], s=data[:,3], c=data[:,2], cmap=cm)
    for a,b,c in zip(data[:,0], data[:,1], data[:,2]): 
        plt.text(a, b, str(round(c,2)), fontsize="x-small")
    plt.xlabel("Layers")
    plt.ylabel("h-Factor")
    plt.title("Minimum Loss for Layer and h-Factor MLMC")
    # plt.show()
    plt.colorbar()
    fig.savefig("./graphs/layers_hFactor_loss_time.png")


if __name__ == "__main__":
    lines = processAllLines()
    plot_L_H_time(lines)
    plot_L_H_minLoss(lines)
    plot_L_H_minLoss_time(lines)

