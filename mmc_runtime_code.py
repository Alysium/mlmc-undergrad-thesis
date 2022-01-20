import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from mmc_raissi_fbsnn import FBSNN
import os

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
    def __init__(self, Xi, T, M, N, D, layers, mode, activation):

        super().__init__(Xi, T, M, N, D, layers, mode, activation)

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


#equation (16), used to calculate loss
# only used for to generate test graph (plotting 100-dimensional B-S-B eq) where comparing learned with exact
def u_exact(t, X):  # (N+1) x 1, (N+1) x D
    r = 0.05
    sigma_max = 0.4
    return np.exp((r + sigma_max ** 2) * (T - t)) * np.sum(X ** 2, 1, keepdims=True)  # (N+1) x 1


def run_model(model, N_Iter, learning_rate, L, h_Factor, fixed=0, seed=42, modelTitle="", onlyRelativeError=False):
    if not onlyRelativeError:
        os.mkdir("./D"+str(D)+"/data/"+modelTitle+"/seed"+str(seed))

    tot = time.time()
    samples = 5
    print(model.device)
    graph = model.train(N_Iter, learning_rate, L, h_Factor, fixed=fixed, seed=seed,modelTitle=modelTitle)
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

    Y_test = np.reshape(u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, D])),
                        [M, -1, 1])


    plt.figure()
    plt.plot(graph[0], graph[1])
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.yscale("log")
    plt.title('Evolution of the training loss')
    # plt.title('Evolution of the training loss')
    plt.savefig("./D"+str(D)+"/data/"+modelTitle+"/seed"+str(seed)+"/"+str(D) + '-dimensional Black-Scholes-Barenblatt loss, ' + modelTitle, bbox_inches='tight')
    minLoss = min(graph[1])


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
    # plt.legend()
    plt.legend(bbox_to_anchor=(1.02, 1.02))
    plt.savefig("./D"+str(D)+"/data/"+modelTitle+"/seed"+str(seed)+"/"+str(D) + '-dimensional Black-Scholes-Barenblatt paths, ' + modelTitle, bbox_inches='tight')

    errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
    mean_errors = np.mean(errors, 0) #average across all the batches for each timestamp (number of timesteps constant since validation timesteps)
    std_errors = np.std(errors, 0)
    plt.figure()
    plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + modelTitle)
    # plt.legend()
    # plt.savefig(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + model.mode + "-" + model.activation)
    plt.legend(bbox_to_anchor=(1.02, 1.02))
    plt.savefig("./D"+str(D)+"/data/"+modelTitle+"/seed"+str(seed)+"/"+str(D) + '-dimensional Black-Scholes-Barenblatt, ' + modelTitle, bbox_inches='tight')
    
    #save stats into a text file in each seed run
    np.savez(os.path.join(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__),"D{}/data".format(D)),modelTitle),"seed"+str(seed)), "reltiveErrors.npz"), meanRelativeError=mean_errors,stdRelativeError=std_errors)

    return {"runtime":time.time() - tot,"minLoss":minLoss, "meanRelativeError": mean_errors, "stdRelativeError": std_errors}


if __name__ == "__main__":
    onlyRelativeError = True


    M = 10  # number of trajectories (batch size)
      #in the paper, M = 100
    N = 50  # number of time snapshots
    D = 10  # number of dimensions

    layers = [D + 1] + 4 * [256] + [1] #represents the number of neurons for each layer
    #there are 4 hidden layers of 256 neurons
    #layers = [D + 1] + [256] +[1]

    Xi = np.array([1.0, 0.5] * int(D / 2))[None, :] #initaliztion of X0, sahpe (1,10)
    T = 1.0 #total time
    "Available architectures"

    mode = "FC"
    activation = "ReLU"  # Sine and ReLU are available
    learning_rate = 1e-3
    iterations = 2*10**3
    model = BlackScholesBarenblatt(Xi, T,
                                    M, N, D,
                                    layers, mode, activation)

    def runMLMC(L_arr,h_Factor_arr):
        seeds = [41,42,43]
        for L in L_arr:
            for h_Factor in h_Factor_arr:
                runTimes = []
                minLosses = []
                meanRelativeErrors = None
                stdRelativeErrors = None
                modelTitle = model.mode + "_" + model.activation + "_iters-"+str(iterations)+"_L-"+str(L)+"_hFactor-"+str(h_Factor)
                if not onlyRelativeError:
                    try:
                        os.mkdir(os.path.join(os.path.join(os.path.dirname(__file__),"D{}/data".format(D)),modelTitle))
                    except FileExistsError:
                        print("model aleady exists")
                        continue

                for seed in seeds:
                    modelDict = run_model(model, iterations, learning_rate, L, h_Factor, seed=seed, modelTitle=modelTitle, onlyRelativeError=onlyRelativeError)
                    if modelDict == None: continue
                    runtime,minLoss = modelDict['runtime'],modelDict['minLoss']
                    meanRelativeError,stdRelativeError = modelDict['meanRelativeError'], modelDict['stdRelativeError']
                    runTimes.append(runtime)
                    minLosses.append(minLoss)
                    
                    if meanRelativeErrors is None: 
                        meanRelativeErrors = np.copy(meanRelativeError)
                        stdRelativeErrors = np.copy(stdRelativeError)
                    else:
                        meanRelativeErrors = np.append(meanRelativeErrors, meanRelativeError, axis=1)
                        stdRelativeErrors = np.append(stdRelativeErrors, stdRelativeError, axis=1)

                runTimeAvg = float(sum(runTimes))/len(runTimes)
                minLossesAvg = float(sum(minLosses))/len(minLosses)
                meanRelativeErrors = np.average(meanRelativeErrors, axis=1)
                stdRelativeErrors = np.average(stdRelativeErrors, axis=1)

                if not onlyRelativeError:
                    with open('./D{}/model_trainTime_minLoss.txt'.format(D), 'a') as f:
                        f.writelines(modelTitle + ": " + str(runTimeAvg)+", "+str(str(minLossesAvg))+"\n")

                def writeRelErr(txtFileName, relErrors):
                    f = open(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__),"D{}/data".format(D)),modelTitle),txtFileName), "w+")
                    f.write(str(relErrors))
                    f.close()
                writeRelErr("meanRelativeErrors.txt", meanRelativeErrors)
                writeRelErr("stdRelativeErrors.txt", stdRelativeErrors)


    L_arr = [1]
    h_Factor_arr = [2]
    #l , h_factor
    runMLMC(L_arr, h_Factor_arr)
    #runMLMC([6], [2,3])

    
#the number of iterations and learning rate are the same as the Raissi Paper for the first go around
    #the number of iterations and learning rate are the same as the Raissi Paper for the first go around


    # #fixed ------
    # mode = "FC"
    # activation = "ReLU"  # Sine and ReLU are available
    # learning_rate = 1e-3
    # iterations = 2*10**3
    # model = BlackScholesBarenblatt(Xi, T,
    #                                 M, N, D,
    #                                 layers, mode, activation)
    # L = 1 #are not used since fixed is set
    # h_factor=1 #are not used since fixed is set
    # run_model(model, iterations, learning_rate, L,h_factor,fixed=32)
    # #the number of iterations and learning rate are the same as the Raissi Paper for the first go around