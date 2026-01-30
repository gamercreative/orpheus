import torch
import torch.nn as nn
# from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import matplotlib # use this for ubuntu wayland sessions
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import data, draw_model
import torch, random, numpy as np
import utils

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# device agonostic code setu
device = utils.GetDevice()
print(f"using {device}")
print("current cuda gpu device:", torch.cuda.current_device())

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

epochs = 80
for epoch in range(epochs + 1):
    for X_batch, Y_batch, letter_id in dataset.MixedXY():
        
        embed = model.embed(letter_id)

        optimizer.zero_grad()
        
        h0 = torch.zeros( X_batch.size(0), model.hidden_size, device=X_batch.device)
        c0 = torch.zeros( X_batch.size(0), model.hidden_size, device=X_batch.device)
        h0 = model.letter_to_h(embed).detach()
        c0 = model.letter_to_c(embed).detach()
        
        X_input_next = X_batch.clone()
        p_teacher = max(0.5, 1.0 - epoch / epochs)
        
        # insert model predictions in with teacher forcing ( shecduled teacher forcing)
        for t in range(1,X_batch.size(1)):
            h0,c0 = model.cell(X_batch[:,t,:].squeeze(1),(h0,c0))
            h0 = model.norm(h0)
            out = model.output_layer(h0)
            
            motor_xy = out[:, 0:2]           # dx, dy
            motor_dt = out[:, 2]              # dt
            pen_logits = out[:, 3:6]          # logts
            pen_state = utils.GetPenStateFromOut(pen_logits)  # integer 0,1,2

            with torch.no_grad():
                pred_next = torch.cat([motor_xy[:,0].unsqueeze(1),motor_xy[:,1].unsqueeze(1),pen_state.unsqueeze(1),motor_dt.unsqueeze(1)],dim=1)
            
            use_teacher = (torch.rand(X_batch.size(0), device=X_batch.device) < p_teacher).unsqueeze(-1)
            X_input_next[:, t, :] = torch.where(use_teacher, X_batch[:, t, :], pred_next)

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

        if epoch % (epochs//5) == 0:
            print(f"Epoch {epoch} - Loss: {loss.item()}")

# inference

def InferenceRun(letter_embed,index = 0):
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
            
    x = torch.tensor(x) + index*200
    y = torch.tensor(y)
    p = torch.tensor(p)
    t = torch.tensor(t)

    dx = torch.diff(x)
    dy = torch.diff(y)
    
    dpos = torch.abs(dx) + torch.abs(dy)
    
    plt.scatter(x,y, label="letter")

while True:
    # for letter in dataset.letters:
    word = input("enter word: ")
    index = 0
    for letter in word:
        letter_id = utils.CharToId(letter)
        letter_embed = model.embed(letter_id)
        weight = torch.rand_like(letter_embed) * 0.05
        letter_embed = letter_embed + weight
        print(letter_id)

        InferenceRun(letter_embed,index)
        index+=1
        
    plt.title(word)
    plt.legend()
    plt.show()