import pandas as pd

def n_day_prediction(data_set,n_days,prediction_feature):
    '''Creating the target attribute, by shifting the specified feature backwards
    
    
    Keyword arguments:
    data_set -- Scaled Dataset
    n_days --  No. of days ahead
    prediction_feature -- feature that is to be predicted
    '''
    data_set['Target'] = data_set[prediction_feature].shift(-1*n_days)
    data_set.dropna(inplace=True)
    return data_set

# Splitting the dataset, into 2 parts
def create_dataset(data_set,start_date,end_date,prediction_feature,current):
    dates = pd.date_range(start_date,end_date)
    new_set = pd.DataFrame(index=dates)
    new_set = new_set.join(data_set)
    new_set.dropna(inplace=True)
    new_result = pd.DataFrame(new_set[prediction_feature],index=new_set.index)
    new_set.drop(prediction_feature,axis=1,inplace=True)
    current_value = new_set[current]
    return new_set,new_result,current_value
