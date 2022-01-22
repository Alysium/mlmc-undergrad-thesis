import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


one_layer_h_factor_experiment_include = [2,3,4,5,6,7]

def processAllLines(D):
    lines = []
    with open("./D{}/model_trainTime_minLoss.txt".format(D)) as f:
        lines = f.readlines()
    retLines = []
    for line in lines:
        lineArr = lineProcessor(line)
        if lineArr == []: continue
        retLines.append(lineArr)
    retLines = getTotalNumberTimesteps(np.array(retLines))
    return separateOneLayerControlPoints(retLines)

def separateOneLayerControlPoints(data):
    experimentalData = []
    controlData = []
    for dataRow in data:
        if int(dataRow[0]) == 1:
            controlData.append(dataRow.tolist())
            if int(dataRow[1]) in one_layer_h_factor_experiment_include:
                experimentalData.append(dataRow.tolist())
        else:
            experimentalData.append(dataRow.tolist())

    return np.array(experimentalData), np.array(controlData)


def lineProcessor(line):
    if "Train Time" in line: return []
    model,time,loss = line.split(" ")
    time = float(time[:-1])
    loss = float(loss)
    hFactor = int(model.split("hFactor-")[1][:-1])
    L = int(model.split("_L-")[1].split("_hFactor")[0])
    return [L, hFactor, loss, time]

def organizeToHeatMapMatrix(data, ind):
    #ind = 3 for time, 2 for minLoss
    maxLayer = int(max(data[:,0]))
    maxHFactor = int(max(data[:,1]))
    heatMapMatrix = np.zeros((maxHFactor-1,maxLayer))
    for dataRow in data: heatMapMatrix[int(dataRow[1])-2, int(dataRow[0])-1] = dataRow[ind]
    heatMapMatrix = heatMapMatrix[::-1]
    mask = np.ma.masked_where(heatMapMatrix==0,heatMapMatrix)
    x_axis_labels = [i for i in range(1,maxLayer+1)]
    y_axis_labels = [i for i in range(2,maxHFactor+1)][::-1]
    return heatMapMatrix, mask, x_axis_labels, y_axis_labels

def plot_L_H_time_heatmap(data,D):
    heatMatrix, mask, x_axis_labels, y_axis_labels = organizeToHeatMapMatrix(data, 3)
    norm = heatMatrix/np.linalg.norm(heatMatrix)
    f,ax = plt.subplots(figsize = (11,5))
    ax = sns.heatmap(heatMatrix, mask=mask.mask, \
        xticklabels=x_axis_labels, yticklabels=y_axis_labels, \
        linewidths=0.1, annot=True)
    plt.title("Total Time in Seconds for Layer and h-Factor MLMC")
    plt.xlabel("Layers")
    plt.ylabel("h-Factor")
    f.savefig("./D{}/graphs/layers_hFactor_time_heatmap.png".format(D))

def plot_L_H_time(data, D):
    fig = plt.figure()
    plt.scatter(data[:,0], data[:,1], data[:,3])
    for a,b,c in zip(data[:,0], data[:,1], data[:,3]): 
        plt.text(a, b, str(round(c,2)), fontsize="x-small")
    plt.title("Total Time in Seconds for Layer and h-Factor MLMC")
    plt.xlabel("Layers")
    plt.ylabel("h-Factor")
    fig.savefig("./D{}/graphs/layers_hFactor_time.png".format(D))

def plot_L_H_minLoss_heatmap(data,D):
    heatMatrix, mask, x_axis_labels, y_axis_labels = organizeToHeatMapMatrix(data, 2)
    norm = heatMatrix/np.linalg.norm(heatMatrix)
    f,ax = plt.subplots(figsize = (11,5))
    ax = sns.heatmap(heatMatrix, mask=mask.mask, \
        xticklabels=x_axis_labels, yticklabels=y_axis_labels, \
        linewidths=0.1, annot=True)
    plt.title("Minimum Loss for Layer and h-Factor MLMC")
    plt.xlabel("Layers")
    plt.ylabel("h-Factor")
    f.savefig("./D{}/graphs/layers_hFactor_loss_heatmap.png".format(D))


def plot_L_H_minLoss(data, D):
    fig = plt.figure()
    plt.scatter(data[:,0], data[:,1])
    for a,b,c in zip(data[:,0], data[:,1], data[:,2]): 
        plt.text(a, b, str(round(c,2)), fontsize="x-small")
    plt.xlabel("Layers")
    plt.ylabel("h-Factor")
    plt.title("Minimum Loss for Layer and h-Factor MLMC")
    # plt.show()
    fig.savefig("./D{}/graphs/layers_hFactor_loss.png".format(D))


def plot_L_H_minLoss_time(data, D):
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
    fig.savefig("./D{}/graphs/layers_hFactor_loss_time.png".format(D))


#calculate total number of timesteps across all epochs
#assumed for now that total epochs is set as constant and that epochs is evenly split between levels
def getTotalNumberTimesteps(data):
    const_epochs = 2000
    layerTimesteps_data = []
    for i in range(len(data)):
        dataRow = data[i]
        layer,h_factor = int(dataRow[0]),int(dataRow[1])
        if layer == 1:
            layerTimesteps_data.append(const_epochs * h_factor)
            continue

        epochs_per_layer = const_epochs/layer
        layerTimesteps = [h_factor]
        for _ in range(layer-1): layerTimesteps.append(layerTimesteps[-1]*h_factor) 
        layerTimesteps = [epochs_per_layer*timesteps for timesteps in layerTimesteps]
        layerTimesteps_data.append(sum(layerTimesteps))
    layerTimesteps_data = np.array(layerTimesteps_data).reshape(-1,1)
    return np.append(data, layerTimesteps_data, axis=1)

def plot_totalTimeSteps_loss(mlmc, mc, D):
    fig = plt.figure()
    plt.scatter(mlmc[:,-1], mlmc[:,2])
    plt.scatter(mc[:,-1], mc[:,2], c='r')
    plt.xlabel("Total Number of Timesteps")
    plt.ylabel("Minimum Loss")
    plt.title("Total Number of Timesteps during training and Resulting Minimum Loss")
    # plt.show()
    fig.savefig("./D{}/graphs/totalTimeSteps_loss.png".format(D))    

def plot_totalTimeSteps_time(mlmc, mc, D):
    fig = plt.figure()
    plt.scatter(mlmc[:,-1], mlmc[:,3])
    plt.scatter(mc[:,-1], mc[:,3], c='r')
    plt.xlabel("Total Number of Timesteps")
    plt.ylabel("Total Training Time (s)")
    plt.title("Total Number of Timesteps and Resulting Total Training Time")
    # plt.show()
    fig.savefig("./D{}/graphs/totalTimeSteps_time.png".format(D))    

if __name__ == "__main__":
    D = 10
    mlmc, mc_control = processAllLines(D)
    #lines = array of tuples [layers, h factor, train time, minloss]
    # plot_totalTimeSteps_loss(mlmc, mc_control, D)
    # plot_totalTimeSteps_time(mlmc, mc_control, D)
    #plot_L_H_time(mlmc, D)
    plot_L_H_time_heatmap(mlmc,D)
    plot_L_H_minLoss_heatmap(mlmc,D)
    #plot_L_H_minLoss(mlmc, D)
    # plot_L_H_minLoss_time(lines, D)

