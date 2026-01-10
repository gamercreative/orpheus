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
weight = torch.tensor([2.0,1.0,1.0],device=device)
criterianCE = nn.CrossEntropyLoss(ignore_index=-1,weight=weight)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=200,
#     eta_min=5e-5
# )

for epoch in range(100):
    for X_batch, Y_batch in dataset.MixedXY():

        optimizer.zero_grad()
        out = model(X_batch)
        
        motor_out = out[:,:,[0,1,2]] # dx dy dt
        motor_true = Y_batch[:,:,[0,1,3]]
        
        pen_out = out[:,:,3:6] # pen_states 0 , 1 ,2
        pen_true = Y_batch[:,:,2].long()
        
        loss_motor = criterianMSE(motor_out,motor_true)
        loss_pen = criterianCE(pen_out.reshape(-1,3), pen_true.reshape(-1)) * 3.0
        loss = loss_motor + loss_pen
        
        loss.backward()
        optimizer.step()

        if epoch % 19 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item()}")

# inference

def InferenceRun(letter_embed):
    x_temp,y_temp,p_temp,t_temp = draw_model.DrawOut(model,letter_embed,500)

    print(letter_embed)
    print(x_temp.size(0))
    print(torch.max(p_temp))
    print(torch.min(p_temp))
    
    x = []
    y = []
    p = []
    t = []
    for i in range(0,p_temp.size(dim=0)):
        if p_temp[i] > -0.5: # cehck this for errors in graph
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
    
    plt.plot(time_axis, p, label="p")
    plt.plot(time_axis[1:], dpos, label="dx")
    plt.show()
    plt.scatter(x,y, label="letter")

while True:
    for letter in dataset.letters:
        letter_embed = model.embed(letter.letter_id)
        print(letter.letter_id)

        plt.title(letter.letter_path)
        plt.legend()
        InferenceRun(letter_embed)
        plt.title(letter.letter_path)
        plt.legend()
        plt.show()