import numpy as np
from helper import *


#Homework 1: Logistic Regression

def sigmoid(z):
        '''
        Calculating the sigmoid
        Theta(z)= 1/(1+exp(-z))
        '''
        return 1 / (1 + np.exp(-z))



def logistic_regression(data, label, max_iter, learning_rate):
    
        weight = np.zeros(data.shape[1])
        N=len(data)
        '''
        Calculation of the logistic regression formula in this method
        '''
        for i in range(0, max_iter):
            for features in range(0, len(weight)):
                sig = 0
                for row_element in range(0, N):
                    upper_term = label[row_element] * data[row_element, features]
                    sig += sigmoid(-weight[features] * data[row_element, features] * label[row_element]) * upper_term
                if(sig>=0.5):
                    gradient = -sig/N  #Calculating gradient
                    weight[features] = weight[features] - (learning_rate * gradient)  # update weights using gradient and learning rate.
        return weight

def thirdorder(data):
        '''
        This function is used for a 3rd order polynomial transform of the data.
        Args:
        data: input data with shape (:, 3) the first dimension represents 
                  total samples (training: 1561; testing: 424) and the 
                  second dimesion represents total features.

        Return:
                result: A numpy array format new data with shape (:,10), which using 
                a 3rd order polynomial transformation to extend the feature numbers 
                from 3 to 10. 
                The first dimension represents total samples (training: 1561; testing: 424) 
                and the second dimesion represents total features.
        '''
        '''
        Calculating third order regression for the logistic function
        in this method
        '''
        new_data = data
        new_features = []
        x3 = data[:,1]*data[:,2]
        new_features.append(x3)
        x4 = np.square(data[:,1])
        new_features.append(x4)
        x5 = np.square(data[:,2])
        new_features.append(x5)
        x6 = np.square(data[:,1]) * data[:,2]
        new_features.append(x6)
        x7 = np.square(data[:,2]) * data[:,1]
        new_features.append(x7)
        x8 = np.power(data[:,1],3)
        new_features.append(x8)
        x9 = np.power(data[:,2],3)
        new_features.append(x9)

        for i in new_features:
                i.shape = (len(data[:,1]),1)
                new_data = np.column_stack((new_data,i))
        return new_data

def accuracy(x, y, w):
        n, _ = x.shape
        m = 0
        '''
        Calculating Accuracy Percentage in this method via the formula
        '''
        for i in range(n):
            sig = sigmoid(np.dot(x[i,:],np.transpose(w)))
            if((sig >= 0.5 and y[i] == -1) or (sig < 0.5 and y[i] == 1)):
                m += 1
        return (n-m)/n
