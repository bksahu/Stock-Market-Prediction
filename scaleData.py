from sklearn.preprocessing import MinMaxScaler
import pandas as pd    

def scale_data(transformed_data):

    '''Return Scaled data
    '''
    # Min Max Normalisation
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(transformed_data)
    
    # Drop -infs
    pd.options.mode.use_inf_as_na = True    
    scaled_data.dropna(inplace=True)
    
    # Standardising data
    scaled_data = (scaled_data-scaled_data.mean())/((1.0*scaled_data.std())/(scaled_data.count()**0.5))


    numericals=scaled_data.columns
    # Scaling standardised attributes
    scaled_data[numericals] = scaler.fit_transform(scaled_data[numericals])
    
    return scaled_data
