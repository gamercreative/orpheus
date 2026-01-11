import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import matplotlib.pyplot as plt
import data, draw_model
import torch, random, numpy as np
import utils

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

epochs = 50
for epoch in range(epochs + 1):
    for X_batch, Y_batch, letter_id in dataset.MixedXY():

        embed = model.embed(letter_id)

        optimizer.zero_grad()
        
        h0 = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_size, device=X_batch.device)
        c0 = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_size, device=X_batch.device)
        h0[0] = model.letter_to_h(embed)
        c0[0] = model.letter_to_c(embed)
        
        X_input_next = X_batch.clone()
        p_teacher = max(0.5, 1.0 - epoch / epochs)
        
        for t in range(1,X_batch.size(1)):
            out,(h0,c0) = model.lstm(X_batch[:,:t,:],(h0,c0))
            out = model.norm(out)
            
            motor_xy = out[0, -1, 0:2]           # dx, dy
            motor_dt = out[0, -1, 2]              # dt
            pen_logits = out[0, -1, 3:6]          # logts
            pen_state = utils.GetPenStateFromOut(pen_logits)  # integer 0,1,2
            
            pred_next = torch.stack([motor_xy[0], motor_xy[1], pen_state.float(), motor_dt]).view(1,1,4)
            
            use_teacher = (torch.rand(X_batch.size(0), device=X_batch.device) < p_teacher).float().unsqueeze(-1)

            # mix ground truth and predicted input
            X_input_next[:, t, :] = use_teacher * X_batch[:, t, :] + (1 - use_teacher) * pred_next.squeeze(1)
        
        out = model(X_input_next,embed)
        #prase the output
        motor_out = out[:,:,[0,1,2]] # dx dy dt
        motor_true = Y_batch[:,:,[0,1,3]]
        
        pen_out = out[:,:,3:6] # pen_states 0 , 1 ,2
        pen_true = Y_batch[:,:,2].long()
        
        loss_motor = criterianMSE(motor_out,motor_true)
        loss_pen = criterianCE(pen_out.reshape(-1,3), pen_true.reshape(-1)) * 3.0
        loss = loss_motor + loss_pen
        
        loss.backward()
        optimizer.step()

        if epoch % epochs//5 == 0:
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
    # for letter in dataset.letters:
    letter = input("enter your letter")
    letter_id = utils.CharToId(letter)
    letter_embed = model.embed(letter_id)
    print(letter_id)

    plt.title(letter)
    plt.legend()
    InferenceRun(letter_embed)
    plt.title(letter)
    plt.legend()
    plt.show()