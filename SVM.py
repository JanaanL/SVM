#HW4 CS_5350

import numpy as np

def load_data(path, add_bias=True):
    """
    Loads and processes the bank note data set
    
    Inputs:
    -path:  string representing the path of the file
    -add_bias:  boolean representing whether an extra column of ones is added to the data, representing
                a bias value
    
    Returns:
    -X:  a numpy array of shape [no_samples, no_attributes (+1 if add_bias is True)]
    -y:  a numpy array of shape [no_samples] that represents the labels {-1, 1} for the dataset X
    """
    
    import numpy as np
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            example = line.strip().split(',')
            if len(example) > 0:
                example = [float(i) for i in example]
                data.append(example)
    X = np.array(data, dtype=np.float64)
    y = X[:,-1]
    y = y.astype(int)
    y[y == 0] = -1
    X = X[:,:-1]
    
    if add_bias == True:
        bias = np.ones((X.shape[0],1),dtype=np.float64) 
        X = np.hstack((X,bias))

    return X, y


def SVM_primal(X, y, epochs, C=0.01, gamma=1, d=1, lr=1):
    """
    Uses the SVM algorithm to predict a linear classifier.
    
    Input:
    -X:  a numpy array of shape [no_samples, no_attributes] representing the dataset
    -y:  a numpy array of shape [no_samples] that has the labels {-1, 1} for the dataset
    -epochs:  an int that represents the number of epochs to perform.
    -C: a float representing the hyperparameter that weights the tradeoff between the regularization term
        and the loss term in the optimization
    -gamma:  A float representing the learning rate to be used in calculating the gradient.
    -d:  a hyperparamter used in the first learning rate schedule
    -lr:  an int (1,2) representing the learning rate schedule
    
    Returns:
    -w:  a numpy array of shape [no_attributes + bias] representing the set of weights learned in the algorithm.
    """

    import numpy as np
    

    N = X.shape[0]
    w = np.zeros(X.shape[1]) # initialize the weights as zero
    gamma_0 = gamma
    subgradient = []
    t = 0
    for epoch in range(epochs):
        
        #Randomly shuffle data
        s = np.arange(N)
        np.random.shuffle(s)
        X = X[s]
        y = y[s]
        
        #Iterate through randomly shuffled training examples
        for i in range(N):
            t += 1
            
            y_pred = y[i] * np.dot(w, X[i])
            if y_pred <= 1:
                w = (1 - gamma) * w + gamma * C * N * y[i] * X[i]
            else:
                w = (1 - gamma) * w
            if lr == 1:
                gamma = gamma_0 / (1 + gamma_0 / d * t)
            else:
                gamma = gamma_0 / (1 + t)
                
            c_val = 0.5 * np.dot(w, w) + c * max(0, 1 - y[i]*np.dot(w, X[i]))
            subgradient.append(c_val)
     
    #Plot the convergence data
    from matplotlib import pyplot as plt
    plt.title('Convergence of Primal SVM')
    plt.xlabel('Iteration')
    plt.ylabel('Update')
    plt.plot(subgradient)
    plt.show()
    
    return w
    

def predict(X, y, w):
    """
    Computes the average prediction error of a dataset X and given weight vector w.
    
    Inputs:
    -X:  a numpy array of shape [no_samples, no_attributes + 1] representing the dataset
    -y:  a numpy array of shape [no_samples] that has the labels for the dataset
    -w:  a numpy array of shape [no_attributes + bias] representing the set of weights to predict the labels.
    
    Returns:
    -error:  A float representing the average prediction error of for the dataset X.
    """
    
    incorrect = 0
    for i in range(X.shape[0]):
        if np.sign(w.dot(X[i])) != y[i]:
            incorrect += 1
           
    print("The total incorrect is " + str(incorrect) + " out of " + str(X.shape[0]))
    return float(incorrect) / X.shape[0]



#Test the Primal Implementations
X, y = load_data('train.csv')
X_test, y_test = load_data("test.csv")

#Using the C parameters specified in HW4
C = [float(1)/873, float(10)/873, float(50)/873, float(100)/873, float(300)/873, float(500)/873, float(700)/873]
print("Results for first learning rate schedule:")
print("\n")
for c in C:
    print("Results for the Primal SVM algorithm using hyperparameters C = " + "{0:.5f}".format(c) + " :")
    w = SVM_primal(X, y, 100, c, gamma=.5, d=1, lr=1)
    print("The weights are: ")
    print(w)
    print("The average training error for SVM_primal is " + "{0:.3f}".format(predict(X, y, w)))
    print("The average test error for SVM_primal is " + "{0:.3f}".format(predict(X_test, y_test, w)))
    print('\n')

print("Results for second learning rate schedule:")
print("\n")    
for c in C:
    print("Results for the Primal SVM algorithm using hyperparameters C = " + "{0:.5f}".format(c) + " :")
    w = SVM_primal(X, y, 100, c, gamma=.5, lr=2)
    print("The weights are: ")
    print(w)
    print("The average training error for SVM_primal is " + "{0:.3f}".format(predict(X, y, w)))
    print("The average test error for SVM_primal is " + "{0:.3f}".format(predict(X_test, y_test, w)))
    print('\n')
    


from scipy.optimize import minimize
import numpy as np
from numpy import linalg as LA
from math import exp

def gaussian_kernel(x1, x2, gamma):
    return exp(-1 * LA.norm(x1 - x2, 2) / gamma)

def K_matrix(x1, x2, gamma):
    """
    Computes the guassian equation matrix for 2 2D numpy arrays
    
    Inputs:
    -x1:  a 2d numpy array of shape [n,n] 
    -x2:  a 2d numpy array of shape [n, n]
    -gamma:  a hyperparameter for the guassian equation
    
    Returns:
    -K:  An matrix of shape [n, n]
    """
    
    K = np.zeros((x1.shape[0], x2.shape[0]))
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            K[i,j] = gaussian_kernel(x1[i], x2[j], gamma)
    return K

def H_matrix(X, y, kernel=False, gamma = 0.01):
    H = np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if kernel:
                H[i,j] = gaussian_kernel(X[i], X[j], gamma) * y[i] * y[j]
            else:
                H[i,j] = np.dot(X[i],X[j])*y[i]*y[j]
    return H

def loss(alphas, H):
    return 0.5 * np.dot(alphas, np.dot(H, alphas)) - np.sum(alphas)

def constraint1(alphas, y):
    return np.dot(alphas,y)

def jac(alphas, H):
    return np.dot(alphas.T,H)-np.ones(alphas.shape[0])

def SVM_dual(X, y, epochs, C=0.1):
    """
    Uses the SVM dual algorithm to predict a linear classifier.  Uses the scipy.optimize method to
    determine the constrained optimization.
    
    Input:
    -X:  a numpy array of shape [no_samples, no_attributes] representing the dataset
    -y:  a numpy array of shape [no_samples] that has the labels {-1, 1} for the dataset
    -epochs:  an int that represents the number of epochs to perform.
    -C: a float representing the hyperparameter that weights the tradeoff between the regularization term
        and the loss term in the optimization
    
    Returns:
    -w:  a numpy array of shape [no_attributes] representing the set of weights learned in the algorithm.
    -b:  a float representing the bias learned in the algorithm
    """
    
    #Define constraints and run scipy.optimize on loss function
    H = H_matrix(X,y)
    args = (H,)
    cons = {'type':'eq', 
            'fun':constraint1,
            'args':(y,)}
    bounds = [(0,C)]*X.shape[0] 
    x0 = np.random.rand(X.shape[0])
    sol = minimize(loss, x0, args=args, jac=jac, constraints=cons, method='L-BFGS-B', bounds = bounds)
    print(sol.message)
    
    #Calculate support vectors and recover w and b
    print ("The number of support vectors is " + str(np.sum(sol.x!=0.0)))
    w = np.sum([sol.x[i] * y[i] * X[i,:] for i in range(X.shape[0])], axis=0)
    b = np.mean(y-np.dot(X,w))

    return w, b
               
def SVM_kernel(X, y, epochs, C = 0.01, gamma=2):
    
    """
    Uses the SVM dual algorithm to predict a linear classifier using a Guassian kernel function.  
    Uses the scipy.optimize method to determine the constrained optimization.
    
    Input:
    -X:  a numpy array of shape [no_samples, no_attributes] representing the dataset
    -y:  a numpy array of shape [no_samples] that has the labels {-1, 1} for the dataset
    -epochs:  an int that represents the number of epochs to perform.
    -C: a float representing the hyperparameter that weights the tradeoff between the regularization term
        and the loss term in the optimization
    -gamma:  a hyperparameter for the guassian equation

    Returns:
    -w:  a numpy array of shape [no_attributes] representing the set of weights learned in the algorithm.
    -b:  a float representing the bias learned in the algorithm
    -alphas:  a numpy array of shape[no_samples] representing the alphas learned in the optimization and 
        used for predictions
    """

    #Define constraints and run scipy.optimize on loss function
    H = H_matrix(X, y, kernel=True, gamma=gamma)
    args = (H,)
    cons = {'type':'eq', 
            'fun':constraint1,
            'args': (y,)}
    bounds = [(0,C)]*X.shape[0] #alpha>=0
    x0 = np.random.rand(X.shape[0])
    sol = minimize(loss, x0, args=args, constraints=cons, method='L-BFGS-B', bounds = bounds)
    print(sol.message)
    
    #Calculate support vectors and recover w and b
    print ("The number of support vectors is " + str(np.sum(sol.x!=0.0)))
    w = np.zeros(X.shape[1])
    w = np.sum([sol.x[i] * y[i] * X[i,:] for i in range(X.shape[0])], axis=0)
    K = K_matrix(X, X, gamma)
    K = K * sol.x * y
    b = np.mean(y - np.sum(K, axis=0))
    
    return w, b, sol.x


def predict_dual(X, y, w, bias):
    """
    Computes the average prediction error of a dataset X and given weight vector w for the dual SVM algorithm.
    
    Inputs:
    -X:  a numpy array of shape [no_samples, no_attributes] representing the dataset
    -y:  a numpy array of shape [no_samples] that has the labels for the dataset
    -w:  a numpy array of shape [no_attributes] representing the set of weights to predict the labels.
    -bias:  a float representing the bias
    
    Returns:
    -error:  A float representing the average prediction error of for the dataset X.
    """
    
    incorrect = 0
    for i in range(X.shape[0]):
        if np.sign(w.dot(X[i]) + bias) != y[i]:
            incorrect += 1
           
    print("The total incorrect is " + str(incorrect) + " out of " + str(X.shape[0]))
    return float(incorrect) / X.shape[0]


# In[775]:


def predict_kernel(X, y, x_i, y_i, alphas, bias=0, gamma=.5):
    """
    Computes the average prediction error of a dataset X and given weight vector w.
    
    Inputs:
    -X:  a numpy array of shape [no_samples, no_attributes] representing the test dataset
    -y:  a numpy array of shape [no_samples] that has the labels for the test dataset
    -x_i: a numpy array of shape [no_training_samples, no_attributes] representing the training dataset
    -y_i: a numpy array of shape [no_training_samples] representing the labels for the training dataset
    -alphas:  a numpy array of shape [no_training_samples] representing the alpha values learned in the dual
         algorithm
    -bias:  a float representing the bias
    -gamma:  a float representing a hyperparameter for the guassian equation
         
    Returns:
    -error:  A float representing the average prediction error of for the dataset X.
    """
    
    incorrect = 0
    K = K_matrix(x_i, X, gamma=gamma)
    alphas = np.reshape(alphas, (alphas.shape[0],1))
    y_i = np.reshape(y_i, (y_i.shape[0],1))
    
    K = alphas * y_i * K
    s = np.sum(K, axis=0)+bias
    for i in range(X.shape[0]):
        if np.sign(s[i]) != y[i]:
            incorrect += 1
           
    print("The total incorrect is " + str(incorrect) + " out of " + str(X.shape[0]))
    return float(incorrect) / X.shape[0]


# In[ ]:


#Test the dual SVM implementation
X, y = load_data('train.csv', add_bias=False)
X_test, y_test = load_data("test.csv", add_bias=False)

C = [float(100)/873, float(500)/873, float(700)/873]


for c in C:
    print("Results for the SVM dual algorithm are: ")
    w, bias = SVM_dual(X, y, 100, c)
    print("The weights for the dual SVM implementation with a parameter of C as " + "{0:.5f}".format(c) + " are: ")
    print(w)
    print(bias)
    print("The average training error for SVM_dual is " + "{0:.3f}".format(predict_dual(X, y, w, bias)))
    print("The average test error for SVM_dual is " + "{0:.3f}".format(predict_dual(X_test, y_test, w, bias)))
    print('\n')
    
#Test the dual SVM implementation with the Guassian kernel
gammas = [0.01, 0.1, 0.5, 1, 2, 5, 10, 100]
for c in C:

    if c == float(500)/873:
        zeroes = np.zeros(X.shape[0])
        a = np.arange(X.shape[0])
    for gamma in gammas:
        print("Results for the SVM kernel algorithm are: ")
        w, bias, alphas = SVM_kernel(X, y, 100, c, gamma=gamma)
        if c == float(500)/873:
            b = np.argwhere(alphas>0)
            print("The number of similar support vectors from the last iteration are: ")
            print((np.intersect1d(a,b)).size)
            a = b
        print("The weights for the kernel implementation with a paramter of C as " + "{0:.5f}".format(c) + 
              " and gamma is " + str(gamma) + " are: ")
        print(w)
        print(bias)
        print("The average training error for SVM_kernel is " + "{0:.3f}".format(predict_kernel(X, y, X, y, alphas, bias, gamma)))
        print("The average test error for SVM_kernel is " + "{0:.3f}".format(predict_kernel(X_test, y_test, X, y, alphas, bias, gamma)))
        print('\n')
    

def perceptron_kernel(X, y, epochs, gamma=1):
    """
    Uses the perceptron algorithm with a Gaussian kernel to predict a linear classifier.
    
    Input:
    -X:  a numpy array of shape [no_samples, no_attributes + 1] representing the dataset
    -y:  a numpy array of shape [no_samples] that has the labels {-1, 1} for the dataset
    -epochs:  an int that represents the number of epochs to perform.
    -gamma:  A float representing a hyperparameter for the guassian equation.
    
    Returns:
    -alphas:  a numpy array of shape [no_samples] representing the "counts" for the algorithm

    """

    import numpy as np
    w = np.zeros(X.shape[1]) # initialize the weights as zero
    alphas = np.ones(X.shape[0]) # initalize counts (alphas) as zero
    K = K_matrix(X, X, gamma)
    yK = np.dot(K,y)
    update = 0
    correct = 0
    
    for t in range(epochs):
        for j in range(X.shape[0]):
            s = np.dot(alphas*y,K)
            y_pred = np.sign(s[j])
            if y_pred != y[j]:
                alphas[j] = alphas[j] + 1
                update += 1
            else:
                correct += 1
    
    print("The total number of updates to alpha is: " + str(update) + " and the number of correct iterations is "
         + str(correct))
    #w = np.dot((alphas * y), X)
    
    return alphas
    


#Test the kernal Perceptron implementation
X, y = load_data('train.csv')
X_test, y_test = load_data("test.csv")

gammas = [0.01, 0.1, 0.5, 1, 2, 5, 10, 100]
for gamma in gammas:
    print("Results for the Perceptron Kernel algorithm are: ")
    alpahs = perceptron_kernel(X, y, 10, gamma=gamma)
    print("The results for gamma = " + str(gamma) + " are: ")
    print("The average training error for perceptron_kernel is " + "{0:.3f}".format(predict_kernel(X, y, X, y, alphas, 0, gamma)))
    print("The average test error for perceptron_dual is " + "{0:.3f}".format(predict_kernel(X_test, y_test, X, y, alphas, 0, gamma)))
    print('\n')


