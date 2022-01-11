import numpy as np
from abc import ABC, abstractmethod
from mmc_raissi_nn_model import Sine, Resnet, VerletNet, SDEnet, LSTM
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
#from Models import Resnet, Sine


class FBSNN(ABC):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation):
        device_idx = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device("cpu")

        #  We set a random seed to ensure that your results are reproducible
        # torch.manual_seed(0)

        self.Xi = torch.from_numpy(Xi).float().to(self.device)  # initial point
        self.Xi.requires_grad = True

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
        #print(self.model)

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
            # print(n)
            # print(size)
            # print(t)
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

    def fetch_minibatch_train(self, size):  # Generate time + a Brownian motion
        T = self.T

        M = self.M #number of samples that we are going to generate
        N = size #number of timesteps that will be uised
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

    def fetch_minibatch(self):  # Generate time + a Brownian motion
        T = self.T

        M = self.M
        N = self.N
        D = self.D

        Dt = np.zeros((M, N + 1, 1))  # M x (N+1) x 1
        DW = np.zeros((M, N + 1, D))  # M x (N+1) x D

        dt = T / N

        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))

        t = np.cumsum(Dt, axis=1)  # M x (N+1) x 1
        W = np.cumsum(DW, axis=1)  # M x (N+1) x D
        t = torch.from_numpy(t).float().to(self.device)
        W = torch.from_numpy(W).float().to(self.device)

        return t, W

    def train(self, N_Iter, learning_rate, L, h_Factor, fixed=0, modelTitle="",seed=42):
        '''
          L = number of layers that will be performed
          h_Factor = factor in which each number of timesteps will be divided against
        '''
        loss_temp = np.array([])

        previous_it = 0
        # if self.iteration != []:
        #     previous_it = self.iteration[-1]
          
        #want to divide evently across the layers
        layerIters = N_Iter // L

        # Optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        start_time = time.time()
        time_per_epoch = []
        
        numLayers = []
        for it in range(previous_it, previous_it + N_Iter):
            #size = number of time steps between [0,T]
            #this is the point where multilayer monte carlo is used------

            if fixed==0:
              l = it//layerIters+1 #layers 1 to L
              size = h_Factor**l       
              if numLayers == [] or l != numLayers[-1][0]:
                numLayers.append([l,size])   
            else:
              size = fixed



            # if it < N_Iter/5:
            #   size = 2
            # elif N_Iter/5 <= it < (2*N_Iter)/5:
            #   size = 4
            # elif (2*N_Iter)/5 <= it < (3*N_Iter)/5:
            #   size = 8
            # elif (3*N_Iter)/5 <= it < (4*N_Iter)/5:
            #   size = 16
            # else:
            #   size = 32
            #------------------------------------------------------------

            self.optimizer.zero_grad()

            #this step fetches the samples of the number of timestamps that will be used
            t_batch, W_batch = self.fetch_minibatch_train(size)  # M x (N+1) x 1, M x (N+1) x D
            #---------------------------------------------------------------------------
            '''
            The vectors are as:
            [M, N+1, 1] or [M, N+1, D]
            -> index 0: number of samples that will be run
            -> index 1: number of timesteps run for each sample
            -> index 2: number of dimensions
              -> time only has 1 dimensions
              -> Brownian motion has 5 dimensions for 5 dimensional input

            '''

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
        plt.plot(list(np.linspace(0,N_Iter-100,int(N_Iter/100))), time_per_epoch)
        plt.xlabel("epoch")
        plt.ylabel("time(seconds)")
        plt.title("Time Per Epoch for "+modelTitle+"_seed-"+str(seed))
        plt.savefig("./data/"+modelTitle+"/seed"+str(seed)+"/TimePerEpoch_"+modelTitle+"_seed-"+str(seed))
        print("Number of layers arr", numLayers)
        return graph

    def predict(self, Xi_star, t_star, W_star):
        Xi_star = torch.from_numpy(Xi_star).float().to(self.device)
        Xi_star.requires_grad = True
        loss, X_star, Y_star, Y0_pred = self.loss_function(t_star, W_star, Xi_star)

        return X_star, Y_star

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

