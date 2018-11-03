from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
import pandas as pd


def performance_metrics(y_true,y_predicted,y_current,model_name,equity):
    '''Return Scores    
    '''    
    compare=pd.DataFrame(y_true,columns=['True'])
    compare['Prediction']=y_predicted
    compare['Current']=y_current
    
    scores=dict()
    
    #Calculating the R^2 Score of our predictions
    scores['r2_score']=r2_score(y_true,y_predicted)
    
    #Calculating the MSE Score of our predictions 
    scores['mean_squared_error']=mean_squared_error(y_true,y_predicted)
    
    ##Calculating the MAE Score of our predictions
    scores['mean_absolute_error']=mean_absolute_error(y_true,y_predicted)

    ##Calculating the MAE Score of our predictions
    scores['relative_root_mean_squared_error']=sqrt(scores['mean_absolute_error']*100)
       
    print('\nThe Performance Results for the {} for {} is :'.format(model_name,equity))
    print('The R^2 score is {}'.format(scores['r2_score'])) 
    print('The Mean Absolute Percent Error (MAPE) is {} %'.format(scores['mean_absolute_error']*100))
    print('The Mean Absolute Error (MAE) is {}'.format(scores['mean_absolute_error']))
    print('The Mean Squared Error (MSE) is {}'.format(scores['mean_squared_error']))
    print('The Relative Root Mean Squared Error (rRMSE) is {} %'.format(scores['relative_root_mean_squared_error']))    
    print('') 
    return scores
