import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
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
X1 = data.AssignStrokeToLetter(X1,Y1,[1.0, 0.0, 1.0, 0.0])

X2,Y2 = data.LoadData("dataset/ready2.json")
X2 = data.AssignStrokeToLetter(X2,Y2,[1.0, 0.0, 0.0, 0.0])

end_token = torch.tensor([0.0,0.0,2.0,1.0]).unsqueeze(0).unsqueeze(0)

print(Y1.shape)
print(end_token.repeat(Y1.size(0),1,1).shape)

let1_end = end_token.repeat(Y1.size(0),1,1)
let2_end = end_token.repeat(Y2.size(0),1,1)

Y1 = torch.cat([Y1,let1_end],dim=1)
Y2 = torch.cat([Y2,let2_end],dim=1)

# init
model = draw_model.DrawerLSTM()

# train
criterian = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

for epoch in range(200):
    for X_batch, Y_batch in [(X1, Y1), (X2, Y2)]:
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterian(out, Y_batch)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item()}")

time_frame = torch.tensor(range(0,100)).unsqueeze(1)
# inference
while True:
    letter_embed = torch.tensor([1.0,0.0,0.0,0.0]).unsqueeze(0).unsqueeze(1)
    x1,y1,p1,t1 = draw_model.DrawOut(model,letter_embed,100)
    
    plt.plot(time_frame,p1)
    plt.plot(time_frame,t1)
    plt.title("C")
    plt.axis('equal')
    plt.show()
    
    letter_embed = torch.tensor([1.0,0.0,1.0,0.0]).unsqueeze(0).unsqueeze(1)
    x2,y2,p2,t2 = draw_model.DrawOut(model,letter_embed,100)
    
    plt.plot(time_frame,p2)
    plt.plot(time_frame,t2)
    plt.title("B")
    plt.axis('equal')
    plt.show()
    
    print(torch.max(p1))
    print(torch.max(p2))