import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import matplotlib.pyplot as plt
import data, draw_model
import torch, random, numpy as np

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# device agonostic code setu
if torch.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# load data
X1,Y1 = data.LoadData("dataset/c.json")
X2,Y2 = data.LoadData("dataset/l.json")
X3,Y3 = data.LoadData("dataset/i.json")

X1 = data.AssignStartToken(X1)
X2 = data.AssignStartToken(X2)
X3 = data.AssignStartToken(X3)

X1 = data.AssignStrokeToLetter(X1,[1.0, 0.0, 1.0, 0.0]) # C
X2 = data.AssignStrokeToLetter(X2,[1.0, 0.0, 0.0, 0.0]) # L
X3 = data.AssignStrokeToLetter(X3,[1.0, 1.0, 0.0, 0.0]) # i

Y1 = data.AssignEndToken(Y1)
Y2 = data.AssignEndToken(Y2)
Y3 = data.AssignEndToken(Y3)

print(Y1.shape)
print(Y2.shape)
print(Y3.shape)

X1 = X1.to(device)
X2 = X2.to(device)
X3 = X3.to(device)

Y1 = Y1.to(device)
Y2 = Y2.to(device)
Y3 = Y3.to(device)

# init
model = draw_model.DrawerLSTM(device).to(device)

for names in model.lstm._all_weights:
    for name in filter(lambda n: "bias" in n, names):
        bias = getattr(model.lstm, name)
        n = bias.size(0)
        # forget gate is in the 2nd quarter of biases
        bias.data[n//4:n//2].fill_(1.0)

# train
criterian = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

for epoch in range(220):
    for X_batch, Y_batch in [(X1, Y1), (X2, Y2), (X3, Y3)]:
        optimizer.zero_grad()
        out = model(X_batch)
        
        loss_start_token = criterian(out[:,0,:], Y_batch[:,0,:]) * 1.28
        loss_lett = criterian(out[:,1:-1,:], Y_batch[:,1:-1,:])
        loss_end_token = criterian(out[:,-1,:], Y_batch[:,-1,:]) * 1.28
        loss = loss_start_token + loss_lett + loss_end_token
        
        loss.backward()
        optimizer.step()

        if epoch % 19 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item()}")
            
            # for p in model.parameters():
                # if p.grad is not None:
                    # print("grad min/max ",p.grad.min().item(),p.grad.max().item())

# inference

def InferenceRun(letter_embed):
    x_temp,y_temp,p_temp,t_temp = draw_model.DrawOut(model,letter_embed,100)

    print(letter_embed)
    print(x_temp.size(0))
    print(torch.max(p_temp))
    print(torch.min(p_temp))
    
    x = []
    y = []
    p = []
    t = []
    for i in range(0,p_temp.size(dim=0)):
        if p_temp[i] > 0.2:
            x.append(x_temp[i])
            y.append(y_temp[i])
            p.append(p_temp[i])
            t.append(t_temp[i])
            
    time_axis = torch.tensor(range(0,len(p)))
            
    x = torch.tensor(x)
    y = torch.tensor(y)
    p = torch.tensor(p)
    t = torch.tensor(t)

    dx = torch.diff(x)
    dy = torch.diff(y)
    
    dpos = torch.abs(dx) + torch.abs(dy)
    
    # plt.plot(time_axis, p, label="p")
    # plt.plot(time_axis[1:], dpos, label="dx")
    plt.scatter(x,y, label="letter")

while True:
    letter_embed = [1.0, 0.0, 0.0, 0.0]
    InferenceRun(letter_embed)
    plt.title("L")
    plt.legend()
    plt.show()

    letter_embed = [1.0, 0.0, 1.0, 0.0]
    InferenceRun(letter_embed)
    plt.title("C")
    plt.legend()
    plt.show()
    
    letter_embed = [1.0, 1.0, 0.0, 0.0]
    InferenceRun(letter_embed)
    plt.title("i")
    plt.legend()
    plt.show()