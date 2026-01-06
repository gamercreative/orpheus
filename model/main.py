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
dataset = data.Dataset("dataset/data")

# init
model = draw_model.DrawerLSTM(device).to(device)

dataset.PrepareDatasetEmbeddings(model)

for names in model.lstm._all_weights: # increase rest gate retention
    for name in filter(lambda n: "bias" in n, names):
        bias = getattr(model.lstm, name)
        n = bias.size(0)
        # forget gate is in the 2nd quarter of biases
        bias.data[n//4:n//2].fill_(1.0)

# train
criterianMSE = nn.MSELoss()
criterianCE = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

for epoch in range(300):
    for X_batch, Y_batch in dataset.GetXY():
        weights = torch.ones_like(Y_batch)
        weights[:,-1,:] *= 1.4
        weights[:,0,:] *= 1.28

        optimizer.zero_grad()
        out = model(X_batch)
        
        motor_out = out[:,:,[0,1,3]]
        pen_out = out[:,:,2]
        
        motor_true = Y_batch[:,:,[0,1,3]]
        pen_true = Y_batch[:,:,2]
        
        loss_motor = criterianMSE(motor_out,motor_true)
        loss_pen = criterianMSE(pen_out,pen_true)
        loss = loss_motor + loss_pen
        
        # loss = torch.multiply(loss,weights).mean()
        
        loss.backward()
        optimizer.step()

        if epoch % 19 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item()}")

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
    for letter in dataset.letters:
        letter_embed = model.embed(letter.letter_id)
        print(letter.letter_id)

        InferenceRun(letter_embed)
        plt.title(letter.letter_path)
        plt.legend()
        plt.show()