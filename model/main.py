import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import data, draw_model

# device agonostic code setu
if torch.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# load data
X1,Y1 = data.LoadData("dataset/ready.json")
X1 = data.AssignStrokeToLetter(X1,Y1,[1.0, 0.0, 0.0, 0.0])

X2,Y2 = data.LoadData("dataset/ready2.json")
X2 = data.AssignStrokeToLetter(X2,Y2,[1.0, 0.0, 1.0, 0.0])

# init
model = draw_model.DrawerLSTM()

# train
criterian = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

for epoch in range(200):
    for X_batch, Y_batch in [(X1, Y1), (X2, Y2)]:
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterian(out[:,:-1:,:], Y_batch)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item()}")


# inference
letter_embed = torch.tensor([1.0,0.0,0.0,0.0]).unsqueeze(0).unsqueeze(1)
x1,y1 = draw_model.DrawOut(model,letter_embed,30)

letter_embed = torch.tensor([1.0,0.0,0.0,0.0]).unsqueeze(0).unsqueeze(1)
x2,y2 = draw_model.DrawOut(model,letter_embed,30)

x_max = torch.max(x1)

x2 = x1 + x_max + 20

x = torch.cat([x1,x2])
y = torch.cat([y1,y2])

plt.plot(x,y)
plt.axis('equal')
plt.show()