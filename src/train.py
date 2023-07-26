#Importing Libraries
import numpy as np
import random
import pandas as pd
from pylab import mpl, plt
mpl.rcParams['font.family'] = 'serif'
#%matplotlib inline

# from pandas import datetime
import math, time, itertools, datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

def stocks_data(dates):
    df = pd.DataFrame(index=dates)
    pwd = os.getcwd()
    dir = f'{pwd if "/" in pwd[-1] else f"{pwd}/"}dataset/'
    dataset_filename = os.listdir(dir)[1]
    df_temp = pd.read_csv(f"{dir}{dataset_filename}", index_col='Date', parse_dates=True, usecols=['Date', 'Close/Last'], na_values=['nan'])
    df_temp = df_temp.rename(columns={'Close': 'aapl'})
    df = df.join(df_temp)
    return df

dates = pd.date_range('06-10-2017', "07-24-2023", freq='B')
df = stocks_data(dates)
df = df.fillna(method='pad')
# df = df.interpolate()

#Normalize data
max = -1
for column in df:
    max = df[column].abs().max()
    df[column] = df[column] / max

def load_data(stock, look_back):
    data_raw = stock.values #Convert to numpy array
    data = []

    for idx in range(len(data_raw) - look_back):
        data.append(data_raw[idx:idx+look_back])
    
    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]

look_back = 60 #sequence length
x_train, y_train, x_test, y_test = load_data(df, look_back)

#Training and test data shapes..
# print("x_train.shape =", x_train.shape)
# print("y_train.shape =", y_train.shape)
# print("x_test.shape =", x_test.shape)
# print("y_test.shape =", y_test.shape)

#Convert training and test sets to torch.Tensor
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

#Building the structure of the model
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        #Hidden dimensions
        self.hidden_dim = hidden_dim

        # Numb of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape;
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state w/ 0s
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backprop through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        #Idx hidden state of last time step
        # out.size() -> 100, 32, 100
        # out[:, -1, :] -> 100, 100 -> just want last time step
        out = self.fc(out[:, -1, :])
        # out.size() -> 100, 10
        return out

#Configure model before training
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss()

optim = torch.optim.Adam(model.parameters(), lr=0.015)

# print(model)
# print(len(list(model.parameters())))
# for i in range(len(list(model.parameters()))):
#     print(list(model.parameters())[i].size())

#Start training
num_epochs = 100
hist = np.zeros(num_epochs)

seq_dim = look_back-1

for t in range(num_epochs):
    #Hidden state should only be available if you don't want your LSTM to be stateful
    #model.hidden = model.init_hidden()

    #Forward pass
    y_train_pred = model(x_train)

    loss = loss_fn(y_train_pred, y_train)
    if t % 10 == 0 and t != 0:
        print(f"Epoch {t} MSE: {loss.item()}")
    hist[t] = loss.item()

    #Zero out gradient (accum between epochs)
    optim.zero_grad()

    #backward pass
    loss.backward()

    #update params
    optim.step()

# plt.plot(hist, label="Training loss")
# plt.legend()
# plt.show()

y_test_pred = model(x_test)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit_transform(df['Close/Last'].values.reshape(-1,1))

y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())

trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print("Train Score: %.2f RMSE" % trainScore)
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print("Test Score: %.2f RMSE" % testScore)

figure, axes = plt.subplots(figsize=(15,6))
axes.xaxis_date()

#Scaling y_test and y_test_pred to dollar amounts again..
for i in range(len(y_test)):
    y_test[i] = y_test[i] * max

for j in range(len(y_test_pred)):
    y_test_pred[j][0] = y_test_pred[j][0] * max

#Plot and show predictions and what the results actually were
axes.plot(df[len(df)-len(y_test):].index, y_test, color='red', label="Real AAPL Stock Price")
axes.plot(df[len(df)-len(y_test):].index, y_test_pred, color='blue', label='Predicted AAPL Stock Price')
plt.title("AAPL Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.savefig('aapl_pred.png')
plt.show()
