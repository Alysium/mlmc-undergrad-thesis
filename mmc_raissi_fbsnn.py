import numpy as np
from abc import ABC, abstractmethod

from sqlalchemy import null
from mmc_raissi_nn_model import Sine, Resnet, VerletNet, SDEnet, LSTM
import time
import utils

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
#from Models import Resnet, Sine


class FBSNN(ABC):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation, seed):
        device_idx = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device("cpu")

        #  We set a random seed to ensure that your results are reproducible
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.Xi = torch.from_numpy(Xi).float().to(self.device)  # initial point
        self.Xi.requires_grad = True
        self.Xi_numpy = np.copy(Xi)

        self.T = T  # terminal time
        self.M = M  # number of trajectories
        self.N = N  # number of time snapshots
        self.D = D  # number of dimensions
        self.mode = mode  # architecture: FC, Resnet and NAIS-Net are available
        self.activation = activation
        if activation == "Sine":
            self.activation_function = Sine()
        elif activation == "ReLU":
            self.activation_function = nn.ReLU()

        # initialize NN
        if self.mode == "FC":
            self.layers = []
            for i in range(len(layers) - 2):
                self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
                self.layers.append(self.activation_function)
            self.layers.append(nn.Linear(in_features=layers[-2], out_features=layers[-1]))

            self.model = nn.Sequential(*self.layers).to(self.device)

        elif self.mode == "NAIS-Net":
            self.model = Resnet(layers, stable=True, activation=self.activation_function).to(self.device)
        elif self.mode == "Resnet":
            self.model = Resnet(layers, stable=False, activation=self.activation_function).to(self.device)
        elif self.mode == "Verlet":
            self.model = VerletNet(layers, activation=self.activation_function).to(self.device)
        elif self.mode == "SDEnet":
            self.model = SDEnet(layers, activation=self.activation_function).to(self.device)
        elif self.mode == "LSTM":
            self.model = LSTM(layers, stable=True, activation=self.activation_function).to(self.device)

        self.model.apply(self.weights_init)

        # Record the loss

        self.training_loss = []
        self.iteration = []

    def weights_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)

    #calculates next Yt and Zt value
    # also calculates initail Y0 and Z0 calue
    def net_u(self, t, X):  # M x 1, M x D

        input = torch.cat((t, X), 1)
        u = self.model(input)  # M x 1
        Du = torch.autograd.grad(outputs=[u], inputs=[X], grad_outputs=torch.ones_like(u), allow_unused=True,
                                 retain_graph=True, create_graph=True)[0]
        return u, Du

    def Dg_tf(self, X):  # M x D

        g = self.g_tf(X)
        Dg = torch.autograd.grad(outputs=[g], inputs=[X], grad_outputs=torch.ones_like(g), allow_unused=True,
                                 retain_graph=True, create_graph=True)[0]  # M x D
        return Dg

    #performs one iteration to predict option price Y from underlying asset X
    def loss_function(self, t, W, Xi):
        loss = 0
        X_list = []
        Y_list = []

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]

        X0 = Xi.repeat(self.M, 1).view(self.M, self.D)  # M x D
        #there are M separate samples of initial inputs
        Y0, Z0 = self.net_u(t0, X0)  # M x 1, M x D
        #M samples of outputs and change in outputs (Z)

        X_list.append(X0)
        Y_list.append(Y0)

        size = t.shape[1] #size = number of timestamps N

        for n in range(0, size-1): #iterates through the timestamps
            #specifies the current timestamp
            t1 = t[:, n + 1, :] 
            W1 = W[:, n + 1, :]
            #calculating each time step in equation (7)
            X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.squeeze(
                torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1)), dim=-1)
            Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(
                Z0 * torch.squeeze(torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1))), dim=1,
                keepdim=True)
            #this gets the next values of Y1 based on the NN model
            Y1, Z1 = self.net_u(t1, X1) 

            loss += torch.sum(torch.pow(Y1 - Y1_tilde, 2))
            #Y1_tilde = Y^n_m + phi*delta_t + (Z^n_m)'*Sigma^n_m*deltaW  

            t0 = t1
            W0 = W1
            X0 = X1 # M x D
            Y0 = Y1 # M x 1
            Z0 = Z1

            X_list.append(X0)
            Y_list.append(Y0)

        #Y1 is the "payoff" for final time
        #self.g_tf
        loss += torch.sum(torch.pow(Y1 - self.g_tf(X1), 2))
        loss += torch.sum(torch.pow(Z1 - self.Dg_tf(X1), 2)) # can remove this and perform training too
        #now we have iterated over all the timestamps and have calcualted the losses and accumulated the X's and Y's used to caluclate the losses

        #length of X_list and Y_list is N + 1 (number of timestamps + 1)
        # + 1 is present due to beginning and ending times [0,T]
        # ie: for N = 2 -> time steps are [0, T/2, T]
        X = torch.stack(X_list, dim=1) #shape: M x N+1 x D
        Y = torch.stack(Y_list, dim=1) #shape: M x N+1 x 1

        #returns loss (scalar), X (vector of all the X's generated), Y (vector of all resulting Y's), Y0
        return loss, X, Y, Y[0, 0, 0]

    def fetch_minibatch(self, size=0):  # Generate time + a Brownian motion
        T = self.T

        M = self.M #number of samples that we are going to generate
        N = size if size!=0 else self.N #number of timesteps that will be uised
        D = self.D #number of dimensions

        Dt = np.zeros((M, N + 1, 1))  # M x (N+1) x 1
        DW = np.zeros((M, N + 1, D))  # M x (N+1) x D

        dt = T / N

        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))

        #cumsum = cumulative sum
        t = np.cumsum(Dt, axis=1)  # M x (N+1) x 1
        W = np.cumsum(DW, axis=1)  # M x (N+1) x D
        t = torch.from_numpy(t).float().to(self.device)
        W = torch.from_numpy(W).float().to(self.device)

        return t, W

    def train(self, N_Iter, learning_rate, L, h_Factor, Xi, rel_error_threshold, rel_error_fixedIter, Nl, \
            fixed=0, modelTitle=""):
        '''
          L = number of layers that will be performed
          h_Factor = factor in which each number of timesteps will be divided against
        '''

        loss_temp = np.array([])

        previous_it = 0
        # if self.iteration != []:
        #     previous_it = self.iteration[-1]
          
        #want to divide evently across the layers
        layerIters = N_Iter // L if Nl==0 else Nl

        # Optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        time_per_epoch = []
        
        numLayers = []
        graph = None
        start_time = time.time()
        totalTimeSteps = 0

        def forward_backward(it):
            nonlocal start_time, loss_temp, totalTimeSteps, Nl, numLayers, fixed
            '''
            The vectors are as:
            [M, N+1, 1] or [M, N+1, D]
            -> index 0: number of samples that will be run
            -> index 1: number of timesteps run for each sample
            -> index 2: number of dimensions
              -> time only has 1 dimensions
              -> Brownian motion has 5 dimensions for 5 dimensional input

            '''
            if fixed==0:
              l = it//layerIters+1 #layers 1 to L
              size = h_Factor**l       
              if numLayers == [] or l != numLayers[-1][0]:
                numLayers.append([l,size])   
            else:
              size = fixed
              numLayers.append([size])
            totalTimeSteps+=size
            self.optimizer.zero_grad()
            #this step fetches the samples of the number of timestamps that will be used for training
            t_batch, W_batch = self.fetch_minibatch(size=size)  # M x (N+1) x 1, M x (N+1) x D

            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_temp = np.append(loss_temp, loss.cpu().detach().numpy())
            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                time_per_epoch.append(elapsed)
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' %
                      (it, loss, Y0_pred, elapsed, learning_rate))
                start_time = time.time()

            # Loss
            if it % 100 == 0:
                self.training_loss.append(loss_temp.mean())
                loss_temp = np.array([])

                self.iteration.append(it)

            graph = np.stack((self.iteration, self.training_loss))
            return graph
        
        it=0
        if rel_error_threshold == float('inf'):
            it = previous_it
            while it < previous_it + N_Iter:
                #size = number of time steps between [0,T]
                #this is the point where multilayer monte carlo is used------
                graph = forward_backward(it)
                it+=1
        else:
            it = 0
            rel_error = float('inf')
            while rel_error > rel_error_threshold:
                graph = forward_backward(it)
                if it > rel_error_fixedIter:
                    mean_relative_errors,std_relative_errors,rel_error = self.validation_relative_error(Xi)
                    # print("Mean relative erros:", mean_relative_errors)
                    if (it % 200==0): print("rel_error: ", rel_error)

                it+=1
            if type(numLayers)==int: print(it + " iterations of size " + numLayers)
            else: print("numLayers: " , numLayers)
        # time per epoch graph-----------------------------------------------------
        # plt.plot(list(np.linspace(0,N_Iter-100,int(N_Iter/100))), time_per_epoch)
        # plt.xlabel("epoch")
        # plt.ylabel("time(seconds)")
        # plt.title("Time Per Epoch for "+modelTitle+"_seed-"+str(seed))
        # plt.savefig(utils.get_D_data_model(self.D,modelTitle, seed=seed)+"/TimePerEpoch_{0}_seed-{1}".format(modelTitle,seed))
        # -------------------------------------------------------------------------
        return graph, totalTimeSteps

    def validation_relative_error(self,Xi):
        t_val, W_val = self.fetch_minibatch()
        X_pred, Y_pred = self.predict(Xi, t_val, W_val)

        if type(t_val).__module__ != 'numpy':
            t_val = t_val.cpu().numpy()
        if type(X_pred).__module__ != 'numpy':
            X_pred = X_pred.cpu().detach().numpy()
        if type(Y_pred).__module__ != 'numpy':
            Y_pred = Y_pred.cpu().detach().numpy()
        Y_val = self.exact_y(t_val, X_pred)
        # Y_val = np.reshape(self.u_exact(np.reshape(t_val[0:self.M, :, :], [-1, 1]), np.reshape(X_pred[0:self.M, :, :], [-1, self.D])),
        #             [self.M, -1, 1])

        #Y_val and Y_pred shape: [M, N+1, 1] -> [batch size, number of timesnaps, 1]
        relative_errors = self.relative_error(Y_val,Y_pred)
        #average across all the batches for each timestamp (number of timesteps constant since validation timesteps)
        mean_relative_errors = np.mean(relative_errors, 0)
        std_relative_errors = np.std(relative_errors,0)
        #shape is now [N+1,1]
        batch_averaged_mean_relative_error = np.mean(mean_relative_errors,0)[0] 
        #batch_averaged_mean_relative_error is a scalar float
        #returns relative error averaged across batches and a single relative error value
        return mean_relative_errors, std_relative_errors, batch_averaged_mean_relative_error

    def predict(self, Xi_star, t_star, W_star):
        Xi_star = torch.from_numpy(Xi_star).float().to(self.device)
        Xi_star.requires_grad = True
        loss, X_star, Y_star, Y0_pred = self.loss_function(t_star, W_star, Xi_star)
        return X_star, Y_star

    def relative_error(self, Y_actual, Y_pred):
        return np.sqrt((Y_actual - Y_pred) ** 2 / Y_actual ** 2)

    def u_exact(self, t, X):  # (N+1) x 1, (N+1) x D
        r = 0.05
        sigma_max = 0.4
        return np.exp((r + sigma_max ** 2) * (self.T - t)) * np.sum(X ** 2, 1, keepdims=True)  # (N+1) x 1

    def exact_y(self, t, X_pred):
        return np.reshape(
            self.u_exact(
                np.reshape(t[0:self.M, :, :], [-1, 1]), 
                np.reshape(X_pred[0:self.M, :, :], [-1, self.D])),
            [self.M, -1, 1]
        )

    ###########################################################################
    ############################# Change Here! ################################
    ###########################################################################
    @abstractmethod
    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        pass  # M x1

    @abstractmethod
    def g_tf(self, X):  # M x D
        pass  # M x 1

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        M = self.M
        D = self.D
        return torch.zeros([M, D]).to(self.device)  # M x D

    @abstractmethod
    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        M = self.M
        D = self.D
        return torch.diag_embed(torch.ones([M, D])).to(self.device)  # M x D x D
    ###########################################################################

