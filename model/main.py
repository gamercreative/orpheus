import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import matplotlib.pyplot as plt
import json

# device agonostic code setu
if torch.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
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

base_embed = torch.tensor([1.0, 0.0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
print(base_embed.shape)

base_embed = base_embed.repeat(Y.size(0),1,1)

print(base_embed.shape)
print(X.shape)
print(Y.shape)

X = torch.cat([base_embed,X],dim=1)

# X = letter_embed

# model
class DrawerLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_size = 2
        self.hidden_size = 128
        self.num_layers = 3
        self.output_size = 2
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size,self.output_size)
        
    def forward(self,x):
        batch_size = x.size(0)
        h0 = torch.randn(model.num_layers, X.size(0), model.hidden_size)
        c0 = torch.randn(model.num_layers, X.size(0), model.hidden_size)

        out,(hn,cn) = self.lstm(x,(h0,c0))

        out = self.norm(out)
        
        return self.output_layer(out)

model = DrawerLSTM()

# train
criterian = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    optimizer.zero_grad()
    out = model(X)
    loss = criterian(out[:,:-1,:],Y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item()}")

# def DrawOut(model,inp):
#     model.eval()
#     out = model(inp).detach().cpu()
#     motor_seq = out[0]

#     # convert deltas to absolute coordinates
#     x = torch.cumsum(motor_seq[:, 0], dim=0)
#     y = torch.cumsum(motor_seq[:, 1], dim=0)

#     plt.plot(x, y)
#     plt.axis('equal')
#     plt.show()

def DrawOut(model,lett,steps):
    model.eval()
    h0 = torch.randn(model.num_layers,1,model.hidden_size)
    c0 = torch.randn(model.num_layers,1,model.hidden_size)
    inp = lett

    motor_seq = []
    for _ in range(steps):
        out,(h0,c0) = model.lstm(inp,(h0,c0))
        out = model.norm(out)
        out = model.output_layer(out)
        
        delta = out[:,-1:,:] + torch.rand_like(out[:,-1,:]) * 0.2
        motor_seq.append(delta.squeeze(0))
        inp = delta

    motor_seq = torch.stack(motor_seq)
    print(motor_seq.shape)
    # convert deltas to absolute coordinates
    x = torch.cumsum(motor_seq[:,:, 0], dim=0).detach().cpu()
    y = torch.cumsum(motor_seq[:,:, 1], dim=0).detach().cpu()

    plt.plot(x, y)
    plt.axis('equal')
    plt.show()

letter_embed = torch.tensor([1.0,0.0]).unsqueeze(0).unsqueeze(1)
DrawOut(model,letter_embed,80)
DrawOut(model,letter_embed,80)

# Rand = torch.rand_like(X)
# X = X + Rand
# DrawOut(model,X)

# Rand = torch.rand_like(X)
# X = X + Rand
# DrawOut(model,X)