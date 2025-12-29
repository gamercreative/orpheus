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
X1,Y1 = data.LoadData("dataset/s.json")
X2,Y2 = data.LoadData("dataset/b.json")

X1 = data.AssignStartToken(X1)
X2 = data.AssignStartToken(X2)

X1 = data.AssignStrokeToLetter(X1,[1.0, 0.0, 1.0, 0.0]) # S
X2 = data.AssignStrokeToLetter(X2,[1.0, 0.0, 0.0, 0.0]) # C

Y1 = data.AssignEndToken(Y1)
Y2 = data.AssignEndToken(Y2)

print(Y1.shape)
print(Y2.shape)

X1 = X1.to(device)
X2 = X2.to(device)
Y1 = Y1.to(device)
Y2 = Y2.to(device)

# init
model = draw_model.DrawerLSTM(device).to(device)

# train
criterian = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.009)

for epoch in range(200):
    for X_batch, Y_batch in [(X1, Y1), (X2, Y2)]:
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterian(out, Y_batch)
        loss.backward()
        optimizer.step()

        if epoch % 19 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item()}")

# inference
while True:
    letter_embed = [1.0, 0.0, 0.0, 0.0]
    x1,y1,p1,t1 = draw_model.DrawOut(model,letter_embed,100)
    time_axis = torch.tensor(range(0,p1.size(0))).unsqueeze(1)
    
    print("c")
    print(x1.size(0))
    print(torch.max(p1))
    print(torch.min(p1))
    
    plt.plot(x1,y1)
    plt.title("C")
    plt.axis('equal')
    plt.show()
    
    plt.plot(time_axis,p1)
    plt.title("C pen_state")
    plt.axis('equal')
    plt.show()
    
    letter_embed = [1.0, 0.0, 1.0, 0.0]
    x1,y1,p1,t1 = draw_model.DrawOut(model,letter_embed,100)
    time_axis = torch.tensor(range(0,p1.size(0))).unsqueeze(1)
    
    print("s")
    print(x1.size(0))
    print(torch.max(p1))
    print(torch.min(p1))
    
    plt.plot(x1,y1)
    plt.title("S")
    plt.axis('equal')
    plt.show()
    
    plt.plot(time_axis,p1)
    plt.title("S pen_state")
    plt.axis('equal')
    plt.show()