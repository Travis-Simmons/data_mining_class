# This program applies gradient descent algorithm and normal equation for linear regression that uses regularization.
# The data set uses the example we did in class.

import numpy as np

X = np.array([[1,5,7,12],[1,4,3,2],[1,11,14,6],[1,8,7,1],[1,9,6,4],[1,3,5,9]])
Y = np.array([11,24,17,16,9,10])



# Below uses Gradient Descent Algorithm

theta0 = 0
theta1 = 0
theta2 = 0
theta3 = 0

alpha = 0.001        
lamb = 100

for i in range(50000):
    
    theta = np.array([theta0,theta1,theta2,theta3])
    
    theta0 -= alpha/len(X)*((X.dot(theta)-Y).dot(X[:,0]))
    theta1 = theta1*(1-lamb*alpha/len(X)) - alpha/len(X)*(X.dot(theta)-Y).dot(X[:,1])
    theta2 = theta2*(1-lamb*alpha/len(X)) - alpha/len(X)*(X.dot(theta)-Y).dot(X[:,2])
    theta3 = theta3*(1-lamb*alpha/len(X)) - alpha/len(X)*(X.dot(theta)-Y).dot(X[:,3])
  

print("GD gives", theta)



# Below uses Normal Equation
    
reg = lamb*np.identity(4)
reg[0][0]=0

theta_NE = np.linalg.inv(X.T.dot(X) + reg).dot(X.T).dot(Y)

print("NE gives", theta_NE)
