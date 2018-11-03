import warnings; warnings.simplefilter('ignore')
from sklearn.neural_network import MLPRegressor as mlp
from parseData import raw_dataset
from techVars import transform
from scaleData import scale_data
from createTrainTest import n_day_prediction, create_dataset
from scores import performance_metrics
from plot import make_graph
from dimReduce import dimReduce
import matplotlib.pyplot as plt


'''List of Dataset & their Alias
-----------------------------------
Shanghai Stock Exchange Data --> SE
S&P 500 --> ^GSPC
Dow Jones Industrial Average --> DJI
State Bank of India --> SBIN.NS
'''
dataset = 'SE'
# Start and end date for parsing 
# Date in order of 'YYYY-MM-DD'
start_date = '2003-01-04'
end_date = '2005-12-31'

# Start and End date for training and testing
start_train = '2003-01-04'
end_train = '2004-12-31'
start_test = '2005-01-01'
end_test = '2005-12-31'

# Predict for next n days
n_days = 1

# Total number components to keep
n_components_pca=30
n_components_ica=11


# Import/Parse data
raw_data=raw_dataset(dataset,start_date,end_date)

# Define Technical Variables
transformed_data = transform(raw_data,30)

# Scale Data
scaled_data = scale_data(transformed_data)

# Predict for next n days
scaled_data=n_day_prediction(scaled_data,n_days,'Adj Close')

# Split into Training and Testing Set
training_set,training_result,training_current=create_dataset(scaled_data,start_train,end_train,'Target','Adj Close')
testing_set,testing_result,testing_current=create_dataset(scaled_data,start_test,end_test,'Target','Adj Close')

# PCA and ICA dimention Reduction
training_set, testing_set = dimReduce(training_set, testing_set, n_components_pca, n_components_ica)

# Model Definition 
clf= mlp(solver='lbfgs',hidden_layer_sizes=(4,))
clf.fit(training_set,training_result)
result_ann=clf.predict(testing_set)
actual_ann=testing_result.values
current_ann=testing_current.values
performance_metrics(actual_ann,result_ann,current_ann,'ANN',dataset)

plt.figure(figsize=(10,2))
make_graph(actual_ann,result_ann,'ANN',str(n_days),plt)
plt.show()