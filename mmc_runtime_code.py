import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from mmc_raissi_fbsnn import FBSNN
import os
import utils
#from FBSNNs import FBSNN

"""
t1 = t[:, n + 1, :]
W1 = W[:, n + 1, :]
#calculating each time step in equation (7)
X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.squeeze(
    torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1)), dim=-1)
Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(
    Z0 * torch.squeeze(torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1))), dim=1,
    keepdim=True)
Y1, Z1 = self.net_u(t1, X1)

"""

class BlackScholesBarenblatt(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation, seed):
        super().__init__(Xi, T, M, N, D, layers, mode, activation, seed)

    #used to calculate Y_{n+1}
    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        interestRate = 0.05
        return interestRate * (Y - torch.sum(X * Z, dim=1, keepdim=True))  # M x 1

    #used to calculate the terminal value
    def g_tf(self, X):  # M x D
        return torch.sum(X ** 2, 1, keepdim=True)  # M x 1 
        # g(s) = |x|^2

    #for BSB eqn, mu_tf = 0
    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z)  # M x D

    #used to calculate X_{n+1} and Y_{n+1}
    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        sigma = 0.4
        return sigma * torch.diag_embed(X)  # M x D x D

    ###########################################################################


# #equation (16), used to calculate loss
# # only used for to generate test graph (plotting 100-dimensional B-S-B eq) where comparing learned with exact
# def u_exact(t, X):  # (N+1) x 1, (N+1) x D
#     r = 0.05
#     sigma_max = 0.4
#     return np.exp((r + sigma_max ** 2) * (T - t)) * np.sum(X ** 2, 1, keepdims=True)  # (N+1) x 1


def run_model(model, N_Iter, learning_rate, L, h_Factor, rel_error_threshold, rel_error_fixedIter, Nl,\
                fixed=0, seed=42, modelTitle=""):
    tot = time.time()
    samples = 5 #number of lines to plot
    # print(model.device)

    graph, totalTimeSteps = model.train(N_Iter, learning_rate, L, h_Factor, Xi, rel_error_threshold, 
            rel_error_fixedIter, Nl, fixed=fixed,modelTitle=modelTitle)
    
    if type(graph).__module__ != np.__name__:
        print("Error Occured: Iterations-Loss graph returned as None")
        return {}
    
    print(modelTitle + " total time:", time.time() - tot, "s")

    np.random.seed(seed)
    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)

    if type(t_test).__module__ != 'numpy':
        t_test = t_test.cpu().numpy()
    if type(X_pred).__module__ != 'numpy':
        X_pred = X_pred.cpu().detach().numpy()
    if type(Y_pred).__module__ != 'numpy':
        Y_pred = Y_pred.cpu().detach().numpy()

    # Y_test = np.reshape(model.u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, D])),
                        # [M, -1, 1])
    Y_test = model.exact_y(t_test, X_pred)
    #Y_test and Y_pred shape: [10,51,1] -> [M, N+1, 1] -> [batch size, number of timesnaps, 1]
    test_relative_errors = model.relative_error(Y_test,Y_pred)
    mean_errors = np.mean(test_relative_errors, 0) #average across all the batches for each timestamp (number of timesteps constant since validation timesteps)
    std_errors = np.std(test_relative_errors, 0)

    plt.figure()
    plt.plot(graph[0], graph[1])
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.yscale("log")
    plt.title('Evolution of the training loss')
    # plt.title('Evolution of the training loss')
    plt.savefig(utils.get_D_data_model(D,modelTitle,seed)+"/"+str(D) + '-dimensional Black-Scholes-Barenblatt loss, ' + modelTitle, bbox_inches='tight')
    minLoss = min(graph[1])

    #plots one path in the batch
    plt.figure()
    plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, 'r--', label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], 'ko', label='$Y_T = u(T,X_T)$')

    plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, 'b')
    plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, 'r--')
    plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], 'ko')

    plt.plot([0], Y_test[0, 0, 0], 'ks', label='$Y_0 = u(0,X_0)$')

    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title(str(D) + '-dimensional Black-Scholes-Barenblatt paths, ' + modelTitle)
    plt.legend(bbox_to_anchor=(1.02, 1.02))
    plt.savefig(utils.get_D_data_model(D,modelTitle,seed)+"/"+str(D) + '-dimensional Black-Scholes-Barenblatt paths, ' + modelTitle, bbox_inches='tight')

    plt.figure()
    plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + modelTitle)
    # plt.legend()
    # plt.savefig(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + model.mode + "-" + model.activation)
    plt.legend(bbox_to_anchor=(1.02, 1.02))
    plt.savefig(utils.get_D_data_model(D,modelTitle,seed)+"/"+str(D) + '-dimensional Black-Scholes-Barenblatt, ' + modelTitle, bbox_inches='tight')
    
    #save stats into a text file in each seed run
    np.savez(os.path.join(utils.get_D_data_model(D,modelTitle,seed), utils.SEED_RELATIVE_ERROR_FILE_NAME),\
         meanRelativeError=mean_errors,stdRelativeError=std_errors)

    # mean_relative_errors, batch_averaged_mean_relative_error = model.validation_relative_error()
    return {"runtime":time.time() - tot,"minLoss":minLoss, "meanRelativeError": mean_errors, "stdRelativeError": std_errors, "totalTimeSteps":totalTimeSteps}


if __name__ == "__main__":


    M = 10 # number of trajectories (batch size)
      #in the paper, M = 100
    N = 50  # number of time snapshots
    D = 10 #100  # number of dimensions

    layers = [D + 1] + 4 * [256] + [1] #represents the number of neurons for each layer
    #there are 4 hidden layers of 256 neurons
    #layers = [D + 1] + [256] +[1]

    Xi = np.array([1.0, 0.5] * int(D / 2))[None, :] #initaliztion of X0, sahpe (1,10)
    T = 1.0 #total time

    mode = "FC"
    activation = "ReLU"  # Sine and ReLU are available
    learning_rate = 1e-3
    iterations = 2*10**3
    # model = BlackScholesBarenblatt(Xi, T,
    #                                 M, N, D,
    #                                 layers, mode, activation)

    def runMLMC(L_arr,h_Factor_arr,rel_error_thresholds=[float('inf')], rel_error_fixedIter=1000, Nl=0, seeds=[41,42,43], fixedArr=[0]):
        for L in L_arr:
            for i,h_Factor in enumerate(h_Factor_arr):
                fixed = fixedArr[i]
                for rel_error_threshold in rel_error_thresholds:

                    print("rel_error_threshold", rel_error_threshold)

                    runTimes = []
                    minLosses = []
                    totalTimeStepsArr=[]
                    meanRelativeErrors = None
                    stdRelativeErrors = None
                    modelTitle = mode + "_" + activation + "_iters-"+str(iterations)+"_L-"+str(L)+"_hFactor-"+str(h_Factor)

                    try:
                        os.mkdir(utils.get_D_data_model(D, modelTitle))
                    except FileExistsError:
                        print("model aleady exists")
                        continue

                    for seed in seeds:
                        try:
                            os.mkdir(utils.get_D_data_model(D, modelTitle,seed=seed))
                        except FileExistsError:
                            print("seed already exists")
                            continue
                        model = BlackScholesBarenblatt(Xi, T,
                            M, N, D,
                            layers, mode, activation, seed)

                        modelDict = run_model(model, iterations, learning_rate, L, h_Factor, rel_error_threshold, rel_error_fixedIter, Nl,\
                            seed=seed, modelTitle=modelTitle, fixed=fixed)
                        if modelDict == None: continue
                        runtime,minLoss,totalTimeSteps = modelDict['runtime'],modelDict['minLoss'],modelDict['totalTimeSteps']
                        meanRelativeError,stdRelativeError = modelDict['meanRelativeError'], modelDict['stdRelativeError']
                        runTimes.append(runtime)
                        minLosses.append(minLoss)
                        totalTimeStepsArr.append(totalTimeSteps)
                        
                        if meanRelativeErrors is None: 
                            meanRelativeErrors = np.copy(meanRelativeError)
                            stdRelativeErrors = np.copy(stdRelativeError)
                        else:
                            meanRelativeErrors = np.append(meanRelativeErrors, meanRelativeError, axis=1)
                            stdRelativeErrors = np.append(stdRelativeErrors, stdRelativeError, axis=1)

                    runTimeAvg = float(sum(runTimes))/len(runTimes)
                    minLossesAvg = float(sum(minLosses))/len(minLosses)

                    totalTimeStepsArrAvg = float(sum(totalTimeStepsArr))/len(totalTimeStepsArr)
                    meanRelativeErrors = np.average(meanRelativeErrors, axis=1)
                    stdRelativeErrors = np.average(stdRelativeErrors, axis=1)

                    # raise Exception
                    with open('./{}/D{}/model_trainTime_minLoss.txt'.format(utils.DATA_FOLDER,D), 'a') as f:
                        f.writelines(modelTitle + ": " + str(runTimeAvg)+", "+\
                            str(minLossesAvg)+", "+str(totalTimeStepsArrAvg)+", "+ str(rel_error_threshold)+"\n")
                    
                    np.savez(os.path.join(utils.get_D_data_model(D,modelTitle), utils.AVERAGED_SEED_RELATIVE_ERROR_FILE_NAME),\
                        meanRelativeError=meanRelativeErrors,stdRelativeError=stdRelativeErrors, \
                        rel_error_threshold=np.array([rel_error_threshold]), totalTimeStepsArrAvg=np.array([totalTimeStepsArrAvg]))


    # L_arr = [2,3]
    # h_Factor_arr = [2,3,4,5,6,7]
    # seeds = [41,42,43]

    # rel_error_thresholds = [0.02]
    # rel_error_fixedIter = 0
    # Nl = 200 #set to 0 if training for fixed number of iterations
    # #l , h_factor
    # runMLMC(L_arr, h_Factor_arr, rel_error_thresholds = rel_error_thresholds, 
    #     rel_error_fixedIter= rel_error_fixedIter, Nl = Nl, seeds = seeds)
    # print("-----")

    #if L_arr = [1] => MC ; L_arr != [1] => MLMC
    L_arr = [2]
    #in the code: L does not actually do anything; only thing that matters is h
    #             and number of iterations per layer before moving on
    h_Factor_arr = [10]#[100,250,500]
    seeds = [41,42,43]#43,44]

    rel_error_thresholds = [0.02]
    rel_error_fixedIter = 0
    Nl = 200 #set to 0 if training for fixed number of iterations
    
    #if fixed=0 -> mlmc, varies by h_factor
    #if fixed>0 -> mc fixed at same number of timesteps defined by fixed
    fixedArr = h_Factor_arr.copy() if L_arr==[1] else [0]
    #l , h_factor
    runMLMC(L_arr, h_Factor_arr, rel_error_thresholds = rel_error_thresholds, 
        rel_error_fixedIter= rel_error_fixedIter, Nl = Nl, seeds = seeds, fixedArr=fixedArr)

