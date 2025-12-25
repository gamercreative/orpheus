import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import json

if torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

#data
with open("dataset/ready.json","r") as f:
    data = json.load(f)
    
sequences = []
for stroke in data:
    seq = list(zip(stroke["dx"],stroke["dy"]))
    sequences.append(seq)
    
tensors = []
for seq in sequences:
    tens = torch.tensor(seq, dtype=torch.float32)
    tensors.append(tens)

motor_seq = pad_sequence(tensors, batch_first=True , padding_value=0.0)

X = motor_seq[:,:-1,:]
Y = motor_seq[:,1:,:]

letter_embed = torch.tensor([1.0,0.0], dtype=torch.float32)
letter_embed = letter_embed.repeat(motor_seq.size(0),1,1)

X = torch.cat([letter_embed,X],dim=1)

print(X.shape)
print(Y.shape)

# model
class DrawerRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_size = 2
        self.hidden_size = 128
        self.num_layers = 3
        self.output_size = 2
        
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size,self.output_size)
        
        
    def forward(self,x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers,batch_size, self.hidden_size, device=x.device)
        
        out,hn = self.rnn(x,h0)
        
        out = self.norm(out)
        
        return self.output_layer(out)
    
model = DrawerRNN()

# train
criterian = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    optimizer.zero_grad()
    out = model(X)
    loss = criterian(out[:,:-1,:],Y)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item()}")

def DrawOut(model):
    out = model(X).detach().cpu()
    motor_seq = out[0]  # take first sequence in batch

    # convert deltas to absolute coordinates
    x = torch.cumsum(motor_seq[:, 0], dim=0)
    y = torch.cumsum(motor_seq[:, 1], dim=0)

    plt.plot(x, y)
    plt.axis('equal')
    plt.show()

DrawOut(model)
DrawOut(model)
DrawOut(model)