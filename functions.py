import numpy as np
from sklearn.metrics import accuracy_score

# defines some statistical functions

def R2(y_data, y_model):
    return 1 - ( (np.sum((y_data - y_model) ** 2)) / (np.sum((y_data - np.mean(y_data)) ** 2)) )
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# own code for svd algorithm pensum side 25
def SVDinv(A):
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = np.linalg.svd(A)
    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
        
    # inv or pinv here????
    # what is the difference?
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.pinv(D)
    return np.matmul(V,np.matmul(invD,UT))

# defines some basic functions
# given by the projects assignment
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# creates the design matrix, from lecture materials p20
def create_X(x, y, n ):
    if len(x.shape) > 1:
        x = np.ravel(x) 
        y = np.ravel(y)
    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in beta
    X = np.ones((N,l))
    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)
    return X
    
# Start by defining the different reggression functions:
# OLS, Ridge, Lasso

# Ordinary Linear Regression
# returns beta values and checks against sklearn for errors
def OLS(xtrain,ytrain):
    # Testing my regression versus sklearn version
    # svd inversion ols regression
    OLSbeta_svd = SVDinv(xtrain.T @ xtrain) @ xtrain.T @ ytrain
    return OLSbeta_svd

# Ridge Regression
# returns beta values  
def RidgeManual(xtrain,lmb,identity,ytrain):
    Ridgebeta = SVDinv((xtrain.T @ xtrain) + lmb*identity) @ xtrain.T @ ytrain
    #print("Ridgebeta in function size is: ",Ridgebeta.size)
    return Ridgebeta
    
    # for scaling the learning rate
def learning_schedule(t):
    t0, t1 = 5, 50
    return t0/(t+t1)

# stochastic gradient descent
def SGD(training_data,y_train,X_test,y_test,epochs,mini_batch_size,eta,beta,method,printing):
    n = len(training_data)
    for j in range(epochs):
        #print("Starting epoch: ",j)
        # Should the batches or training data be shuffled??
        mini_batches = [training_data[k:k+mini_batch_size]
            for k in range(0,n,mini_batch_size)]

        mini_batches_y = [y_train[k:k+mini_batch_size]
            for k in range(0,n,mini_batch_size)]    
            
        # go through all batches, pick a mini batch and update beta
        m = len(mini_batches)
        counter = 0
        for mini_batch in mini_batches:
            # pick a random mini batch
            random_int = np.random.randint(0,len(mini_batches)-1)
            if (method == "OLS"):
                #number = mini_batches[random_int].shape
                #print("number is : ",number)
                gradient = (2.0)*mini_batches[random_int].T @ \
                    ((mini_batches[random_int] @ beta)-mini_batches_y[random_int])
            if (method == "Ridge"):
                lmb = 0.0001 #0.0001
                gradient = (2.0)*(mini_batches[random_int].T @ \
                    ((mini_batches[random_int] @ beta)-mini_batches_y[random_int]) \
                        + lmb*beta)
                    
            # scaling the learning rate:
            eta = learning_schedule(j*m+counter)
            counter += 1
            beta -= eta*gradient
            
        # prints how well we are doing
        ypredict = X_test @ beta
        if (j==(epochs-1)):
            if (printing == False):
                print("\nMethod completed is: ",method," regression")
                print("Number of epochs completed: ", j+1)
                print("My R2 score is: ",R2(y_test,ypredict))


    if (printing == True):
        return R2(y_test,ypredict)
    # to categorical turns our integer vector into a onehot representation
    #    one-hot in numpy
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    return onehot_vector

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def predict(X):
    #probabilities = self.feed_forward_out(X)
    return np.argmax(X, axis=1)


def LogRegression(training_data,y_train,Y_train,X_test,y_test,epochs,mini_batch_size,eta,beta):     
    n = len(training_data)
    for j in range(epochs):
        #print("Starting epoch: ",j)
        # Should the batches or training data be shuffled??
        
        
        mini_batches = [training_data[k:k+mini_batch_size]
            for k in range(0,n,mini_batch_size)]

        mini_batches_y = [y_train[k:k+mini_batch_size]
            for k in range(0,n,mini_batch_size)]    

        #print("y_train size is", y_train.shape)
        #print("mini batch size : ",mini_batch_size)
        #print("length of mini batches y: ",len(mini_batches_y))
        # go through all batches, pick a mini batch and update beta
        m = len(mini_batches)
        counter = 0
        #print("m is: ",m," n is: ",n)        
        for mini_batch in mini_batches:
            # pick a random mini batch
            random_int = np.random.randint(0,len(mini_batches)-1)

            #z = np.dot(X, beta)
            #p = sigmoid(z)
            lmb = 0.001
            #print(mini_batches_y[random_int])
            gradient = (2.0)*(mini_batches[random_int].T @ \
                    (sigmoid(mini_batches[random_int] @ beta) - mini_batches_y[random_int]) \
                        + lmb*beta)
                   
            # scaling the learning rate:
            eta = learning_schedule(j*m+counter)
            counter += 1
            beta -= eta*gradient
            
        # prints how well we are doing
        ypredict = predict(X_test @ beta)
        ypredict_training = predict(training_data @ beta)
        if (j==(epochs-1)):
            print("\nUsing my own code: ")
            print("Method completed is: Logistic regression")
            print("Number of epochs completed: ", j+1)
            print("Accuracy score on training set: ", accuracy_score(Y_train, ypredict_training)) 
            #print("y_test is ",y_test.shape," y predict is: ",ypredict.shape)
            print("Accuracy score on test set: ", accuracy_score(y_test, ypredict))