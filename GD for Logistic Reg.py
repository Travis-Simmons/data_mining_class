# This program applies gradient descent algorithm to find the optimal theta values for logistic regression.
# The data set uses HW 4 Problem 2.

import numpy as np

X = np.array([[1,9,5,4,3],[1,4,7,9,2],[1,7,5,8,6],[1,6,4,5,3],[1,8,6,11,7],[1,5,9,6,8],[1,7,4,10,6],[1,6,5,3,10]])
Y = np.array([1,1,0,0,1,0,0,1])


# Define the sigmoid function.

def sig(x):   # the sigmoid function will be applied on array, thus we need to use np.exp().
    return 1/(1+np.exp(-x))


theta0 = 0
theta1 = 0
theta2 = 0
theta3 = 0
theta4 = 0

alpha = 0.05       # 0.05 will converge here.


for i in range(50000):
    
    theta = np.array([theta0,theta1,theta2,theta3,theta4])
    
    cost = - 1/len(X)*(Y.dot(np.log(sig(X.dot(theta)))) + (1-Y).dot(np.log(1-sig(X.dot(theta)))))
    # The 'cost' here is not necessary but it helps to check whether cost function decreases after each iteration.
    # For a good alpha, the cost function should always decrease after each iteration.
  
    theta0 -= alpha/len(X)*((sig(X.dot(theta))-Y).dot(X[:,0]))
    theta1 -= alpha/len(X)*((sig(X.dot(theta))-Y).dot(X[:,1]))
    theta2 -= alpha/len(X)*((sig(X.dot(theta))-Y).dot(X[:,2]))
    theta3 -= alpha/len(X)*((sig(X.dot(theta))-Y).dot(X[:,3]))
    theta4 -= alpha/len(X)*((sig(X.dot(theta))-Y).dot(X[:,4]))

print(theta)

 
 









    
   

