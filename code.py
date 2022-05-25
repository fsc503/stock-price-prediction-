
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split

from keras.layers import LSTM, Input, Dense
from keras.models import Model

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray

import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(1120)


from datetime import date
from nsepy import get_history
stocks_data = get_history(symbol='IBM',
            start=date(2000,4,29),
            end=date(2022,4,29))				   
stocks_data.to_csv('stocks_data.csv')
stocks_data.reset_index(level=0, inplace=True)

stocks_data['Date'] = pd.to_datetime(stocks_data['Date'], format='%Y-%m-%d')

stocks_data.head()

stocks_data_duplicate= stocks_data

stocks_data_high = stocks_data['High']
stocks_data_date = stocks_data['Date']

plt.plot(stocks_data_date,stocks_data_high, linewidth=1)
plt.xlabel("Date")
plt.ylabel('Price')
plt.title("Stock Price")

train_size = int(stocks_data.shape[0]*0.80)

stocks_data = np.asarray(stocks_data_high, dtype=np.float64).reshape(-1, 1)

train_data = stocks_data[:train_size]
test_data = stocks_data[train_size:]

def prepare_dataset(data, window_size):
    X, Y = np.empty((0,window_size)), np.empty((0))
    for i in range(len(data)-window_size-1):
        X = np.vstack([X,data[i:(i + window_size),0]])
        Y = np.append(Y,data[i + window_size,0])   
    X = np.reshape(X,(len(X),window_size,1))
    Y = np.reshape(Y,(len(Y),1))
    return X, Y  
    
def train_evaluate(ga_individual_solution):   
    # Decode GA solution to integer for window_size and num_units
    window_size_bits = BitArray(ga_individual_solution[0:6])
    num_units_bits = BitArray(ga_individual_solution[6:]) 
    window_size = window_size_bits.uint
    num_units = num_units_bits.uint
    print('\nWindow Size: ', window_size, ', Num of Units: ', num_units)
    # Return fitness score of 100 if window_size or num_unit is zero
    if window_size == 0 or num_units == 0:
        return 100, 
        
        
        
        
    
    X,Y = prepare_dataset(train_data,window_size)
    X_train, X_val, y_train, y_val = split(X, Y, test_size = 0.20, random_state = 1120)
    # Train LSTM model and predict on validation set
    inputs = Input(shape=(window_size,1))
    x = LSTM(num_units, input_shape=(window_size,1))(inputs)
    predictions = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32,shuffle=True)
    y_pred = model.predict(X_val)
    # Calculate the RMSE score as fitness score for GA
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print('Validation RMSE: ', rmse,'\n')
    return rmse,



population_size = 5
num_generations = 5
gene_length = 15

# As we are trying to minimize the RMSE score, that's why using -1.0. 
# In case, when you want to maximize accuracy for instance, use 1.0
creator.create('FitnessMax', base.Fitness, weights = (-1.0,))
creator.create('Individual', list , fitness = creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, 
n = gene_length)
toolbox.register('population', tools.initRepeat, list , toolbox.individual)

toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
toolbox.register('select', tools.selRoulette)
toolbox.register('evaluate', train_evaluate)

population = toolbox.population(n = population_size)
r = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.1, 
ngen = num_generations, verbose = False)

best_individuals = tools.selBest(population,k = 1)
best_window_size = None
best_num_units = None

for bi in best_individuals:
    window_size_bits = BitArray(bi[0:6])
    num_units_bits = BitArray(bi[6:]) 
    best_window_size = window_size_bits.uint
    best_num_units = num_units_bits.uint
    print('\nWindow Size: ', best_window_size, ', Num of Units: ', best_num_units)
    
    
    
# Train the model using best configuration on complete training set 
#and make predictions on the test set
X_train,y_train = prepare_dataset(train_data,best_window_size)
X_test, y_test = prepare_dataset(test_data,best_window_size)

inputs = Input(shape=(best_window_size,1))
x = LSTM(best_num_units, input_shape=(best_window_size,1))(inputs)
predictions = Dense(1, activation='linear')(x)
model = Model(inputs = inputs, outputs = predictions)
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32,shuffle=True)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Test RMSE: ', rmse)

from sklearn.preprocessing import MinMaxScaler
test_data = stocks_data_duplicate.iloc[-100:, :]
real_stock_price = test_data.iloc[:,5:6]
real_stock_price.head(10)

mm = MinMaxScaler(feature_range = (0,44))
stock_data_df = mm.fit_transform(real_stock_price)

inputs = real_stock_price
inputs = mm.transform(inputs)
predicted_stock_price = model.predict(inputs)
predicted_stock_price = mm.inverse_transform(predicted_stock_price)

for i in range(len(predicted_stock_price)):
    print(f"Real Price:{real_stock_price.values[i]} --> Predicted Price: {predicted_stock_price[i]}")
    
    
    

