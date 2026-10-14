from torch import nn
class CNNLSTM(nn.Module):
 def __init__(self,n_classes):
  super().__init__()
  self.conv=nn.Sequential(nn.Conv1d(40,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(.25),nn.Conv1d(64,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(.25))
  self.lstm=nn.LSTM(128,96,batch_first=True,bidirectional=True)
  self.head=nn.Sequential(nn.Linear(192,96),nn.ReLU(),nn.Dropout(.35),nn.Linear(96,n_classes))
 def forward(self,x):
  x=self.conv(x.transpose(1,2)).transpose(1,2); x,_=self.lstm(x); return self.head(x[:,-1,:])
