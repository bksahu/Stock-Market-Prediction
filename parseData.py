import pandas as pd


def raw_dataset(dataset,start_date,end_date):
    '''Return the dataset as pandas dataframe  
    
    Keyword arguments:
    dataset -- Name of dataset (It must be in the same folder)
    start_date -- Starting date
    end_date -- Ending date    
    '''
    dates = pd.date_range(start_date,end_date)
    raw_data = pd.DataFrame(index = dates)

    #Import the data
    raw_data = raw_data.join(pd.read_csv('{}.csv'.format(dataset),index_col='Date',parse_dates=True))

    #Drop the null values
    raw_data.dropna(inplace=True)

    return raw_data
