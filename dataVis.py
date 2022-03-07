import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

import utils

one_layer_h_factor_experiment_include = [2,3,4,5,6,7]

def getRelativeError(titles,D):
    relativeErrorAverages = []
    for title in titles:
        try:
            avgRelativeErrorPath = utils.get_D_data_model(D,title)+"/"+utils.AVERAGED_SEED_RELATIVE_ERROR_FILE_NAME
            loadedMeanArr = None
            with np.load(avgRelativeErrorPath) as data:
                loadedMeanArr = data['meanRelativeError']
            relativeErrorAverages.append(np.mean(loadedMeanArr))
        except Exception as _:
            print("Cannot load Average Relative Error for "+title)

    return np.array(relativeErrorAverages)[..., None]

def processAllLines(D):
    lines = []
    with open("./{}/D{}/model_trainTime_minLoss.txt".format(utils.DATA_FOLDER,D)) as f:
        lines = f.readlines()
    retLines = []
    titles = []
    for line in lines:
        title, lineArr = lineProcessor(line)
        if lineArr == []: continue
        retLines.append(lineArr)
        titles.append(title)
    retLines = getTotalNumberTimesteps(np.array(retLines))
    relativeErrorAverages = getRelativeError(titles, D)
    retLines = np.append(retLines, relativeErrorAverages, axis=1)

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
    if "Train Time" in line: return None,[]
    if len(line.split(" "))==5:
        model,time,loss,totalTimeSteps,lossThreshold = line.split(" ")
        totalTimeSteps = float(totalTimeSteps[:-1])
        lossThreshold = float(lossThreshold)
    else:
        model,time,loss = line.split(" ")
        totalTimeSteps = None
        lossThreshold = None
    time = float(time[:-1])
    loss = float(loss[:-1])

    hFactor = int(model.split("hFactor-")[1][:-1])
    L = int(model.split("_L-")[1].split("_hFactor")[0])
    title = line.split(": ")[0]
    return title,[L, hFactor, loss, time, totalTimeSteps, lossThreshold]

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
    plt.title("Total Time in Seconds for L and M MLMC")
    plt.xlabel("L")
    plt.ylabel("M")
    f.savefig("./{}/D{}/graphs/layers_hFactor_time_heatmap.png".format(utils.DATA_FOLDER,D))

def plot_L_H_time(data, D):
    fig = plt.figure()
    plt.scatter(data[:,0], data[:,1], data[:,3])
    for a,b,c in zip(data[:,0], data[:,1], data[:,3]): 
        plt.text(a, b, str(round(c,2)), fontsize="x-small")
    plt.title("Total Time in Seconds for L and M MLMC")
    plt.xlabel("L")
    plt.ylabel("M")
    fig.savefig("./{}/D{}/graphs/layers_hFactor_time.png".format(utils.DATA_FOLDER,D))

def plot_L_H_relativeError(data, D):
    fig = plt.figure()
    plt.scatter(data[:,0], data[:,1])
    for a,b,c in zip(data[:,0], data[:,1], data[:,5]): 
        plt.text(a, b, str(round(c,4)), fontsize="x-small")
    plt.xlabel("L")
    plt.ylabel("M")
    plt.title("Average Relative Error for Layer and M MLMC")
    fig.savefig("./{}/D{}/graphs/layers_hFactor_relative_error.png".format(utils.DATA_FOLDER,D))

def plot_L_H_relativeError_heatmap(data,D):
    heatMatrix, mask, x_axis_labels, y_axis_labels = organizeToHeatMapMatrix(data, 5)
    norm = heatMatrix/np.linalg.norm(heatMatrix)
    f,ax = plt.subplots(figsize = (11,5))
    ax = sns.heatmap(heatMatrix, mask=mask.mask, \
        xticklabels=x_axis_labels, yticklabels=y_axis_labels, \
        linewidths=0.1, annot=True)
    plt.title("Average Relative Error for L and M MLMC")
    plt.xlabel("L")
    plt.ylabel("M")
    f.savefig("./{}/D{}/graphs/layers_hFactor_relative_error_heatmap.png".format(utils.DATA_FOLDER,D))

def plot_L_H_minLoss_heatmap(data,D):
    heatMatrix, mask, x_axis_labels, y_axis_labels = organizeToHeatMapMatrix(data, 2)
    norm = heatMatrix/np.linalg.norm(heatMatrix)
    f,ax = plt.subplots(figsize = (11,5))
    ax = sns.heatmap(heatMatrix, mask=mask.mask, \
        xticklabels=x_axis_labels, yticklabels=y_axis_labels, \
        linewidths=0.1, annot=True)
    plt.title("Minimum Loss for L and M MLMC")
    plt.xlabel("L")
    plt.ylabel("M")
    f.savefig("./{}/D{}/graphs/layers_hFactor_loss_heatmap.png".format(utils.DATA_FOLDER,D))

def plot_L_H_minLoss(data, D):
    fig = plt.figure()
    plt.scatter(data[:,0], data[:,1])
    for a,b,c in zip(data[:,0], data[:,1], data[:,2]): 
        plt.text(a, b, str(round(c,2)), fontsize="x-small")
    plt.xlabel("L")
    plt.ylabel("M")
    plt.title("Minimum Loss for Layer and M MLMC")
    fig.savefig("./{}/D{}/graphs/layers_hFactor_loss.png".format(utils.DATA_FOLDER,D))


def plot_L_H_minLoss_time_legend(data, D):
    fig, ax = plt.subplots()
    cm = plt.cm.get_cmap('RdYlBu')

    scatter = ax.scatter(data[:,0], data[:,1], s=data[:,3], c=data[:,2], cmap=cm)
    for a,b,c in zip(data[:,0], data[:,1], data[:,2]): 
        plt.text(a, b, str(round(c,2)), fontsize="x-small")
    plt.xlabel("Layers")
    plt.ylabel("M")
    plt.title("Minimum Loss for Layer and M MLMC")
    plt.legend(loc=(1.04,0))
    legend1 = ax.legend(*scatter.legend_elements(num=5), bbox_to_anchor = (1.04,1), loc="upper left", title="Loss")
    ax.add_artist(legend1)
    kw = dict(prop="sizes", color="red")

    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=5)
    legend2 = ax.legend(handles, labels, bbox_to_anchor=(1.04, 0), loc="lower left", title="Training Time (s)")

    # legend2 = ax.legend(*scatter.legend_elements(**kw), title="Training Time (s)", \
    #      bbox_to_anchor=(1.04, 0), loc="lower left")
    #plt.show()

    fig.savefig("./{}/D{}/graphs/layers_hFactor_loss_time_legend.png".format(utils.DATA_FOLDER,D), bbox_inches="tight")

def plot_L_H_minLoss_time(data, D):
    fig = plt.figure()
    cm = plt.cm.get_cmap('RdYlBu')

    plt.scatter(data[:,0], data[:,1], s=data[:,3], c=data[:,2], cmap=cm)
    for a,b,c in zip(data[:,0], data[:,1], data[:,2]): 
        plt.text(a, b, str(round(c,2)), fontsize="x-small")
    plt.xlabel("Layers")
    plt.ylabel("M")
    plt.title("Minimum Loss for Layer and M MLMC")
    # plt.show()
    plt.colorbar()
    fig.savefig("./{}/D{}/graphs/layers_hFactor_loss_time.png".format(utils.DATA_FOLDER,D))

def plot_L_H_relativeError_time(data, D):
    fig = plt.figure()
    cm = plt.cm.get_cmap('RdYlBu')
    normRelErrorCol = data[:,5]/data[:,5].max(axis=0)
    plt.scatter(data[:,0], data[:,1],s=data[:,3], c=data[:,5], cmap=cm)
    for a,b,c in zip(data[:,0], data[:,1], data[:,5]): 
        plt.text(a, b, str(round(c,4)), fontsize="x-small")
    plt.xlabel("Layers")
    plt.ylabel("M")
    plt.title("Average Relative Error for Layer and M MLMC")
    # plt.show()
    plt.colorbar()
    fig.savefig("./{}/D{}/graphs/layers_hFactor_realtiveError_time.png".format(utils.DATA_FOLDER,D))


#calculate total number of timesteps across all epochs
#assumed for now that total epochs is set as constant and that epochs is evenly split between levels
def getTotalNumberTimesteps(data):
    if data.all()!=None:
        return data
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
    plt.scatter(mlmc[:,4], mlmc[:,2])
    plt.scatter(mc[:,4], mc[:,2], c='r')
    plt.xlabel("Total Number of Timesteps")
    plt.ylabel("Minimum Loss")
    plt.title("Total Number of Timesteps and Resulting Minimum Loss")
    plt.legend(["MLMC", "MC"])
    fig.savefig("./{}/D{}/graphs/totalTimeSteps_loss.png".format(utils.DATA_FOLDER,D))    

def plot_totalTimeSteps_relativeError(mlmc, mc, D):
    fig = plt.figure()
    plt.scatter(mlmc[:,4], mlmc[:,5])
    plt.scatter(mc[:,4], mc[:,5], c='r')
    plt.xlabel("Total Number of Timesteps")
    plt.ylabel("Average Relative Error")
    plt.title("Total Number of Timesteps and Resulting Average Relative Error")
    plt.legend(["MLMC", "MC"])
    # plt.show()
    fig.savefig("./{}/D{}/graphs/totalTimeSteps_relativeError.png".format(utils.DATA_FOLDER,D))   

def plot_totalTimeSteps_time(mlmc, mc, D):
    fig = plt.figure()
    plt.scatter(mlmc[:,4], mlmc[:,3])
    plt.scatter(mc[:,4], mc[:,3], c='r')
    plt.xlabel("Total Number of Timesteps")
    plt.ylabel("Total Training Time (s)")
    plt.title("Total Number of Timesteps and Resulting Total Training Time")
    plt.legend(["MLMC", "MC"])
    fig.savefig("./{}/D{}/graphs/totalTimeSteps_time.png".format(utils.DATA_FOLDER,D))    

def plot_relativeError_totalTimeSteps(mlmc,mc,D):
    '''
    mlmc- mlmc data
    mc-mc data
    D- number of dimensions
    '''
    fig, ax = plt.subplots()
    print(mlmc[0])
    plt.scatter(mlmc[:,5],mlmc[:,4])
    plt.scatter(mc[:,5],mc[:,4], c='r')
    #change this decimals rounding later to be more relevant
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.xlabel("Validation Relative Error Threashold")
    plt.ylabel("Total Number of Timestemps")
    plt.title("Total Number of Timesteps to Reach Validation Relative Errors")
    plt.legend(["MLMC", "MC"])
    fig.savefig("./{}/D{}/graphs/relativeError_totalTimeSteps.png".format(utils.DATA_FOLDER,D))    


if __name__ == "__main__":
    D = 10
    '''
    mlmc and mc = array of tuples 
    [layers, 
     h factor, 
     train time, 
     minloss, 
     total time steps, 
     relative error threshold, 
     min relative error
    ]
    '''

    mlmc, mc = processAllLines(D)
    # print("mlmc",mlmc)
    # print("mc",mc)
    #plot_L_H_minLoss_time(mlmc,D)
    # plot_L_H_relativeError_time(mlmc,D)
    plot_relativeError_totalTimeSteps(mlmc,mc,D)