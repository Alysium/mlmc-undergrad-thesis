import matplotlib.pyplot as plt
import numpy as np


def processAllLines(D):
    lines = []
    with open("./D{}/model_trainTime_minLoss.txt".format(D)) as f:
        lines = f.readlines()
    retLines = []
    for line in lines:
        lineArr = lineProcessor(line)
        if lineArr == []: continue
        retLines.append(lineArr)
    return removeFirstControlPoint(np.array(retLines))

def removeFirstControlPoint(data):
    control = data[0]
    return control, data[1:]

def lineProcessor(line):
    if "Train Time" in line: return []
    model,time,loss = line.split(" ")
    time = float(time[:-1])
    loss = float(loss)
    hFactor = int(model.split("hFactor-")[1][:-1])
    L = int(model.split("_L-")[1].split("_hFactor")[0])
    return [L, hFactor, loss, time]

def plot_L_H_time(data, D):
    fig = plt.figure()
    plt.scatter(data[:,0], data[:,1], data[:,3])
    plt.xlabel("Layers")
    plt.ylabel("h-Factor")
    plt.title("Total Time for Layer and h-Factor MLMC")
    # plt.show()
    fig.savefig("./D{}/graphs/layers_hFactor_time.png".format(D))

def plot_L_H_minLoss(data, D):
    fig = plt.figure()
    plt.scatter(data[:,0], data[:,1], data[:,2])
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

        epochs_per_layer = const_epochs/layer
        layerTimesteps = [h_factor]
        for _ in range(layer-1): layerTimesteps.append(layerTimesteps[-1]*h_factor) 
        layerTimesteps = [epochs_per_layer*timesteps for timesteps in layerTimesteps]
        layerTimesteps_data.append(sum(layerTimesteps))
    layerTimesteps_data = np.array(layerTimesteps_data).reshape(-1,1)
    return np.append(data, layerTimesteps_data, axis=1)

def plot_totalTimeSteps_loss(data, control, D):
    data = getTotalNumberTimesteps(data)
    fig = plt.figure()
    plt.scatter(data[:,-1], data[:,2])
    plt.scatter(2000*50, control[2], c='r')
    plt.xlabel("Total Number of Timesteps")
    plt.ylabel("Minimum Loss")
    plt.title("Total Number of Timesteps during training and Resulting Minimum Loss")
    # plt.show()
    fig.savefig("./D{}/graphs/totalTimeSteps_loss.png".format(D))    

def plot_totalTimeSteps_time(data, control, D):
    data = getTotalNumberTimesteps(data)
    fig = plt.figure()
    plt.scatter(data[:,-1], data[:,3])
    plt.scatter(2000*50, control[3], c='r')
    plt.xlabel("Total Number of Timesteps")
    plt.ylabel("Total Training Time (s)")
    plt.title("Total Number of Timesteps and Resulting Total Training Time")
    # plt.show()
    fig.savefig("./D{}/graphs/totalTimeSteps_time.png".format(D))    

if __name__ == "__main__":
    D = 10
    control,lines = processAllLines(D)
    #lines = array of tuples [layers, h factor, train time, minloss]
    # plot_totalTimeSteps_loss(lines)
    plot_totalTimeSteps_time(lines, control, D)
    plot_totalTimeSteps_loss(lines, control, D)
    plot_L_H_time(lines, D)
    plot_L_H_minLoss(lines, D)
    plot_L_H_minLoss_time(lines, D)

