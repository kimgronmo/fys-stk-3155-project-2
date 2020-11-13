

# import functions
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import to check code
import sklearn.linear_model as skl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# imports linear regression from scikitlearn to 
# check my code against
from sklearn.metrics import mean_squared_error

# imports Ridge regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# imports functions for real world data
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image

import functions
from sklearn.metrics import accuracy_score
# Where figures and data files are saved..
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)
if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)
if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)
def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)
def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)
def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

class Printerfunctions():
    
    def __init__(self):
        print("\n##### Printing data to files: #####")
        print("Starting printer to generate data:\n")
    
    def partA(self,X_train,y_train,X_test,y_test,epochs,mini_batch_size,eta,beta):
        print("Printing and calculating results for a:\n")

        #data1 = functions.SGD(X_train,y_train,X_test,y_test,epochs,mini_batch_size,eta,beta,"OLS",True)
        #data2 = functions.SGD(X_train,y_train,X_test,y_test,epochs,mini_batch_size,eta,beta,"Ridge",True)        
        #print(data1)
        
        numEpochs = np.array(range(1,101))
        R2scoreOLS = []
        for i in numEpochs:
            data1 = functions.SGD(X_train,y_train,X_test,y_test,i,mini_batch_size,eta,beta,"OLS",True)
            R2scoreOLS.append(data1)
            #print(data1)
            if (i == numEpochs[99]):
                print("R2 score for printer: ",data1)
            
        plt.figure(1)
        plt.title("Part a) R2 score as a function of number of epochs OLS", fontsize = 10)    
        plt.xlabel(r"Number of epochs: 100", fontsize=10)
        plt.ylabel(r"R2 score", fontsize=10)
        plt.plot(numEpochs, R2scoreOLS, label = "R2 score")
        plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', 'parta plot R2 vs epochs.png') \
                    , transparent=True, bbox_inches='tight')

    def partD(self,X_train, Y_train_onehot,eta,lmbd,epochs,batch_size,hidden_neurons,n_categories,X_test,Y_test,Y_train):
        
        # code from lecturenotes in FYS-STK4155 to print to heatmap
        eta_vals = np.logspace(-5, 1, 7)
        lmbd_vals = np.logspace(-5, 1, 7)
        # store the models for later use
        DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
        import NewNeural
        
        # grid search
        for i, eta in enumerate(eta_vals):
            for j, lmbd in enumerate(lmbd_vals):
                dnn = NewNeural.NewNeural(X_train, Y_train_onehot, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    hidden_neurons=hidden_neurons, n_categories=n_categories)
                dnn.train()
        
                DNN_numpy[i][j] = dnn
        
                test_predict = dnn.predict(X_test)
        
                #print("Learning rate  = ", eta)
                #print("Lambda = ", lmbd)
                #print("Accuracy score on test set: ", accuracy_score(Y_test, test_predict))
                #print()
                
                
        
        # visual representation of grid search
        # uses seaborn heatmap
        import seaborn as sns

        sns.set()

        train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
        test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

        for i in range(len(eta_vals)):
            for j in range(len(lmbd_vals)):
                dnn = DNN_numpy[i][j]
        
                train_pred = dnn.predict(X_train) 
                test_pred = dnn.predict(X_test)

                train_accuracy[i][j] = accuracy_score(Y_train, train_pred)
                test_accuracy[i][j] = accuracy_score(Y_test, test_pred)

        
        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Training Accuracy")
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', 'training accuracy part d.png') \
                    , transparent=True, bbox_inches='tight')
        #plt.show()

        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Test Accuracy")
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', 'test accuracy part d.png') \
                    , transparent=True, bbox_inches='tight')
        #plt.show()
        