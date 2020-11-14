

import numpy as np
import sklearn.linear_model as skl

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# import helper functions
# from project 1
import functions
import Printerfunctions

# imports neural network
import NewNeural # for image classification
import NewNeuralNet # for regression analysis

# Using a seed to ensure that the random numbers are the same everytime we run
# the code. Useful to debug and check our code.
np.random.seed(3155)

# Basic information for the FrankeFunction
# The degree of the polynomial (number of features) is given by
n = 5
# the number of datapoints
N = 1000
# the highest number of polynomials
maxPolyDegree = 10
# number of bootstraps
n_bootstraps = 100

# lambda values
nlambdas = 10
lambdas = np.logspace(-6,5, nlambdas)



x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)

# Remember to add noise to function 
z = functions.FrankeFunction(x, y) + 0.01*np.random.rand(N)

X = functions.create_X(x, y, n=n)

# split in training and test data
# assumes 75% of data is training and 25% is test data
X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)

# scaling the data
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

error = np.zeros(maxPolyDegree)
bias = np.zeros(maxPolyDegree)
variance = np.zeros(maxPolyDegree)
polynomial = np.zeros(maxPolyDegree)
    
    
if __name__ == '__main__':
    print("---------------------")
    print("Running main function")
    print("---------------------")
    print("\n""\n")

    print("Starting Project 2: part a")
    print("\n")
    # data has already been scaled..
    betaValues = functions.OLS(X_train,y_train)
    ytilde = X_train @ betaValues
    ypredict = X_test @ betaValues
    
    print("Checking if my code works correctly:")
    print("My MSE is: ",functions.MSE(y_test,ypredict))
    print("My R2 score is: ",functions.R2(y_test,ypredict))
    
    print("\nTesting against sklearn OLS: ")
    clf = skl.LinearRegression(fit_intercept=False).fit(X_train, y_train)
    print("MSE after scaling: {:.4f}".format(mean_squared_error(clf.predict(X_test), y_test)))
    print("R2 score after scaling {:.4f}".format(clf.score(X_test,y_test)))
    
    print("\n""\n")

    beta = np.random.randn(X_train[0].size)
    # eta above 1.0 seems to give overflow errors in beta
    eta = 0.05 #1.0 #0.01 #1.0 #0.01
    Niterations = 10000

    for iter in range(Niterations):
        gradient = (2.0/N)*X_train.T @ ((X_train @ beta)-y_train)
        beta -= eta*gradient

    ypredict = X_test @ beta
    
    print("Checking to see if my GradientDescent code works correctly:")
    print("My MSE is: ",functions.MSE(y_test,ypredict))
    print("My R2 score is: ",functions.R2(y_test,ypredict))    
 

    epochs = 10000
    mini_batch_size = 12 #12
    eta = 0.05 #1.0 #0.01
    
    beta = np.random.randn(X_train[0].size)

    print("\nTrying the same with Stochastic Gradient Descent:")

    # comment away to speed up
    #
    # I seem to be getting a strange result here and cannot see why
    # when I call SGD from here my R2 score for test data is very low (0.89)
    # when I call the same function with same data (random beta) from
    # printerfunctions I get a much betts R2 score (0.93) both using OLS
    # tried with and without np.seed here and its still the same.
    # tried same number of epochs here (100) as in printerfunction.
    # and different batch sizes

    functions.SGD(X_train,y_train,X_test,y_test,epochs,mini_batch_size,eta,beta,"OLS",False)
    
    beta = np.random.randn(X_train[0].size)
    functions.SGD(X_train,y_train,X_test,y_test,epochs,mini_batch_size,eta,beta,"Ridge",False)

    beta = np.random.randn(X_train[0].size)
    printer = Printerfunctions.Printerfunctions()
    printer.partA(X_train,y_train,X_test,y_test,epochs,mini_batch_size,eta,beta)

    
    print("\n""\n")

    print("Starting Project 2: part b and c")
    print("\n")

    epochs = 10000
    batch_size = 100
    eta = 0.0001
    lmbd = 0.01

    # 30 10 10 ok
    # 30 20 error
    # 30 10 and 30 28 ok overflow error
    # Depending on the number of hidden layers and neurons
    # RELU and LeakyRELU seem to be susceptible to overflow errors when 
    # calculating the gradient. Varies with #epochs and eta
    # Not sure if it is something wrong with my code that causes it or
    # if they just need the right parameters to function correctly
    # Left hidden neurons as two layers below to show that it works
    # for multiple layers.

    
    hidden_neurons = [30,28]
    hidden_neurons = [30]
    n_categories = 1

    ######## Sigmoid
    dnnRegression = NewNeuralNet.NewNeuralNet(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    hidden_neurons=hidden_neurons, n_categories=n_categories,activation_f="sigmoid")
    dnnRegression.train()
    dnnRegression.predict(X_test,y_test,X_train,y_train)

    ######## RELU
    dnnRegressionRELU = NewNeuralNet.NewNeuralNet(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    hidden_neurons=hidden_neurons, n_categories=n_categories,activation_f="RELU")
    dnnRegressionRELU.train()
    dnnRegressionRELU.predict(X_test,y_test,X_train,y_train)

    ######## LeakyRELU
    dnnRegressionLeakyRELU = NewNeuralNet.NewNeuralNet(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    hidden_neurons=hidden_neurons, n_categories=n_categories,activation_f="LeakyRELU")
    dnnRegressionLeakyRELU.train()
    dnnRegressionLeakyRELU.predict(X_test,y_test,X_train,y_train)
    

    print("\n""\n")

    print("Starting Project 2: part d")
    print("\n")
    
    import matplotlib.pyplot as plt
    from sklearn import datasets


    # ensure the same random numbers appear every time
    np.random.seed(0)


    # download MNIST dataset
    digits = datasets.load_digits()

    # define inputs and labels
    inputs = digits.images
    labels = digits.target

    print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
    print("labels = (n_inputs) = " + str(labels.shape))


    # flatten the image
    # the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
    n_inputs = len(inputs)
    inputs = inputs.reshape(n_inputs, -1)


    # one-liner from scikit-learn library
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size,
                                                    test_size=test_size)

    # to categorical turns our integer vector into a onehot representation
    from sklearn.metrics import accuracy_score

    #    one-hot in numpy
    def to_categorical_numpy(integer_vector):
        n_inputs = len(integer_vector)
        n_categories = np.max(integer_vector) + 1
        onehot_vector = np.zeros((n_inputs, n_categories))
        onehot_vector[range(n_inputs), integer_vector] = 1
        return onehot_vector

    Y_train_onehot, Y_test_onehot = to_categorical_numpy(Y_train), to_categorical_numpy(Y_test)    
    
    epochs = 100
    batch_size = 100
    eta = 0.01
    lmbd = 0.01
    n_hidden_neurons = 30
    n_categories = 10
    hidden_neurons = [30,20,10]

    dnn2 = NewNeural.NewNeural(X_train, Y_train_onehot, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    hidden_neurons=hidden_neurons, n_categories=n_categories)
    dnn2.train()
    test_predict = dnn2.predict(X_test)

    # accuracy score from scikit library
    # uncertain whether I needed to enable this function manually...
    # accuracy on training and test data
    training_predict = dnn2.predict(X_train)
    print("Accuracy score on training set: ", accuracy_score(Y_train, training_predict))    
    print("Accuracy score on test set: ", accuracy_score(Y_test, test_predict))

    hidden_neurons = [30]
    eta = 0.01
    printer.partD(X_train,Y_train_onehot,eta=eta,lmbd=lmbd,epochs=epochs,batch_size=batch_size,
                hidden_neurons=hidden_neurons,n_categories=n_categories,X_test=X_test,Y_test=Y_test,Y_train=Y_train)

    print("\n""\n")

    print("Starting Project 2: part e")
    print("\n")    
    
    print("Logistic regression using image data")

    print("\nUsing Scikit-Learn:")
    logreg = skl.LogisticRegression(random_state=1,verbose=0,max_iter=1E4,tol=1E-8)
    logreg.fit(X_train,Y_train)
    train_accuracy    = logreg.score(X_train,Y_train)
    test_accuracy     = logreg.score(X_test,Y_test)

    print("Accuracy of training data: ",train_accuracy)
    print("Accuracy of test data: ",test_accuracy)


    epochs = 1000
    beta = np.random.randn(X_train[0].size,10)
    functions.LogRegression(X_train,Y_train_onehot,Y_train,X_test,Y_test,epochs,mini_batch_size,eta,beta)
    
    
    print("\n\n")
    print("#####################")
    print("Setting up directories for analysis: DONE")
    #import printerfunctions
    #printerfunctions.partA()
    #"""