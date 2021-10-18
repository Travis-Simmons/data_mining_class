import numpy as np
import math
class data_test:

    # this one does predictions 
    def make_predictions(X, thetas_list, type = 'linear', desicion_rule = 0.5):

        if type == 'linear':
            print('test')

            prediction_list = []

            for x in X:
                prediction = x.dot(thetas_list)
                prediction_list.append(prediction)

            print('Predictions', prediction_list)
            return prediction_list


        if type == 'log':
            # Prediction storage list
            prediction_list = []

            # Itterate through the data by observation
            for x in X:

                # Prediction is just dot product of Vector x and list of thetas
                prediction = 1/(1+(math.e)**-(x.dot(thetas_list)))

                # Store predictions
                prediction_list.append(prediction)

            rounded_list = []
            for i in prediction_list:
                if i > desicion_rule:
                    rounded_list.append(1)
                else:
                    rounded_list.append(0)


            print('Predictions', rounded_list)
            return rounded_list

    # this one does cost fuctions

    def get_cost(thetas_list, inputs, outputs, type = 'log'):

        if type == 'log':
            print(f'Determining cost for thetas {thetas_list}...\n')

            # We will use this list to store all our errors for the summation
            errors_list = []

            cnt = 1
            # Itterate through the data by observation
            for x, y in zip(inputs,outputs):
                
                # Prediction is just dot product of Vector x and list of thetas
                prediction = 1/(1+(math.e)**-(x.dot(thetas_list)))
        

                # Log error function
                obs_error = -(((y*math.log(prediction, 10)) + ((1-y)*math.log(1-prediction, 10))))


                errors_list.append(obs_error)

                print(f'Prediction for observation {cnt}: {prediction}')
            
                print(f'Error for prediction: {errors_list[cnt-1]}\n')
                cnt += 1 

            total_cost = ((1/len(outputs)) * sum(errors_list))

            print(f'Total cost: {total_cost}.\n')

            return total_cost


        if type == 'linear':
            print(f'Determining cost for thetas {thetas_list}...\n')

            # We will use this list to store all our errors for the summation
            errors_list = []

            cnt = 1
            # Itterate through the data by observation
            for x, y in zip(inputs,outputs):
                
                # Prediction is just dot product of Vector x and list of thetas
                prediction = x.dot(thetas_list)
        

                # Log error function
                obs_error = (prediction-y)**2


                errors_list.append(obs_error)

                print(f'Prediction for observation {cnt}: {prediction}')
            
                print(f'Error for prediction: {errors_list[cnt-1]}\n')
                cnt += 1 

            total_cost = ((1/(2*len(outputs))) * sum(errors_list))

            print(f'Total cost: {total_cost}.\n')

            return total_cost

    # This one does ne for thesa values
    def get_ne(X, Y, lamb = 100, regularization = False):
        if regularization == False:
            theta_NE = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

            print("Non-regularized NE gives", theta_NE)

        if regularization == True:
            reg = lamb*np.identity(4)
            reg[0][0]=0

            theta_NE = np.linalg.inv(X.T.dot(X) + reg).dot(X.T).dot(Y)

            print("Regularized NE gives", theta_NE)


    # this one does gd for theta values

    def sig(x):
        return 1/(1+np.exp(-x))

    def gradient_descent( X, Y, learning_rate, itterations, theta_initials = 0, type = 'linear', lamb = 100, regularization = False ):
        
        # Create initial thetas dict
        theta = [theta_initials for i in X[0]]
        theta_dict = {}
                
        cnt = 0
        for i in theta:
            theta_dict[f'theta{cnt}'] = 0
            cnt += 1

        # Do linear gradient descent
        if type == 'linear':
            
            # intermediate theta dict
            update_dict = {}
            cnt = 0
            for i in theta:
                update_dict[f'theta{cnt}'] = 0
                cnt += 1

            if regularization == False:
                for i in range(itterations):

                    cnt = 0
                    for i in theta_dict:
                        update_dict[f'theta{cnt}'] = theta_dict[f'theta{cnt}'] - learning_rate/len(X)*((X.dot(list(theta_dict.values()))-Y).dot(X[:,cnt]))
                        cnt += 1

                    
                    theta_dict = update_dict.copy()
                        

                    

                print("Non regularized linear GD gives", theta_dict)

            if regularization == True:
                for i in range(itterations):

                    cnt = 0

                    for i in theta_dict:

                        if cnt == 0:
                            update_dict[f'theta{cnt}'] = theta_dict[f'theta{cnt}'] - learning_rate/len(X)*((X.dot(list(theta_dict.values()))-Y).dot(X[:,cnt]))
                            cnt += 1
                        
                        else:
                            update_dict[f'theta{cnt}'] = theta_dict[f'theta{cnt}']*(1-lamb*learning_rate/len(X)) - learning_rate/len(X)*((X.dot(list(theta_dict.values()))-Y).dot(X[:,cnt]))
                            cnt += 1

                    
                    theta_dict = update_dict.copy()
                        

                    

                print("Regularized linear GD gives", theta_dict)

        
        if type == 'log':
            
          # intermediate theta dict
            update_dict = {}
            cnt = 0
            for i in theta:
                update_dict[f'theta{cnt}'] = 0
                cnt += 1

            if regularization == False:

                for i in range(itterations):
                
                    cnt = 0
                    for i in theta_dict:
                        theta_dict[f'theta{cnt}'] = update_dict[f'theta{cnt}'] - learning_rate/len(X)*((data_test.sig(X.dot(list(theta_dict.values())))-Y).dot(X[:,cnt]))

                        cnt += 1

                    theta_dict = update_dict.copy()

            if regularization == True:


                for i in range(itterations):
                
                    cnt = 0
                    for i in theta_dict:
                        if cnt == 0:    

                            theta_dict[f'theta{cnt}'] = update_dict[f'theta{cnt}'] - learning_rate/len(X)*((data_test.sig(X.dot(list(theta_dict.values())))-Y).dot(X[:,cnt]))

                            cnt += 1

                        # add in regularized log gd
                        else:
                            print('do regularized log gd')


                    theta_dict = update_dict.copy()



            print("Logistic GD gives", theta_dict)

        return list(theta_dict.values())