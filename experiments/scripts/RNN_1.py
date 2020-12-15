import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd.variable import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import mean_squared_error

device = 'cuda' if torch.cuda.is_available()  else 'cpu'

########################################################################################################################
########################################################################################################################
### Parameters
input_dim = 1
hidden_dim = 128
num_layers = 2
output_dim = 1
num_epochs = 100
lookback = 100 # choose sequence length

########################################################################################################################
########################################################################################################################
### Functions 
########################################################################################################################
########################################################################################################################

def split_data(stock, lookback):
  #  data_raw = stock.to_numpy() # convert to numpy array Do not need this if you run minmaxscaler
    data_raw = stock
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]


class MyDataset(Dataset):
    def __init__(self, data, pred, transform=None):
        self.data = data.float()
        self.pred = pred.float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.pred[index]
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

    def __len__(self):
        return len(self.data)
    

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
    

########################################################################################################################
########################################################################################################################
## Training process
########################################################################################################################
########################################################################################################################

data = pd.read_csv("data.csv")[-100000:]
data.head()
data.max(axis=0)
data.min(axis=0)
#minmax = MinMaxScaler()
minmax = MinMaxScaler(feature_range=(-1,1)) #Range scale to -1 to 1
##############
# Create a column of successive differnces 
data['diff'] = (((data['5'] - data['5'].shift(-1))/100).shift()).fillna(0)


# Picking a day to predict the stock prices for the 10 mins of the next day
#price = data[['5']]

price_diff = data[['diff']]
price_diff = minmax.fit_transform(price_diff)
x_train, y_train, x_test, y_test = split_data(price_diff, lookback)


x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train= torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)


dataset = MyDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size = 32, shuffle = False)
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
hist = np.zeros(num_epochs)
start_time = time.time()
error = []

for t in range(num_epochs):
    #Minibatching for Stochastic Gradient Descent 
    count = 0
    for n_batch, batch in enumerate(loader):
        n_data = Variable(batch[0].to(device), requires_grad=True)
        ground_truth = Variable(batch[1].to(device), requires_grad=True)
        optimizer.zero_grad()
        y_train_pred = model(n_data)
        loss = criterion(y_train_pred, ground_truth)
        loss.backward()
        optimizer.step()
        #print("Epoch ", t, "MSE: ", loss.item())
    error.append(loss)
    hist[t] = loss.item()
    
torch.save(model.state_dict(), 'rnn_1.pkl')
########################################################################################################################
########################################################################################################################
#### compute error and create output file
########################################################################################################################
########################################################################################################################
prediction = model.to('cpu')(x_test)

rmse = criterion(prediction, y_test)

prediction = minmax.inverse_transform(prediction.detach().numpy())
original = minmax.inverse_transform(y_test.detach().numpy())

#prediction = torch.reshape(prediction, [1,len(prediction)]).tolist()[0]
prediction = [elem[0] for elem in prediction.tolist()]
truth = [elem[0] for elem in original.tolist()]

#rmse = mean_squared_error(truth,prediction)

df = pd.DataFrame({'Real Price': truth,
                   'Predicted Price': prediction,
                   'RMS Error': [rmse]+['']*(len(truth)-1)})
df.to_csv('out_1.csv',index=False)