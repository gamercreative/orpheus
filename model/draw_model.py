import torch
import torch.nn as nn
import data

# model
class DrawerLSTM(nn.Module):
    def __init__(self,device) -> None:
        super().__init__()

        self.device = device
    
        self.embedding_size = 4
        self.input_size = 4 + self.embedding_size
        self.hidden_size = 128
        self.num_layers = 3
        self.output_size = 4

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size,self.output_size)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        out,(hn,cn) = self.lstm(x,(h0,c0)) # no need to transfer the contetx

        out = self.norm(out)
        
        return self.output_layer(out)

def DrawOut(model,embed,steps):
    model.eval()
    h0 = torch.zeros(model.num_layers,1,model.hidden_size, device=model.device)
    c0 = torch.zeros(model.num_layers,1,model.hidden_size, device=model.device)
    inp = data.AssignStartToken(None)

    motor_seq = []
    for _ in range(steps):
        inp = data.AssignStrokeToLetter(inp,embed)
        out,(h0,c0) = model.lstm(inp,(h0,c0))
        out = model.norm(out)
        out = model.output_layer(out)
        
        delta = out[:,-1:,:] + torch.rand_like(out[:,-1,:]) * 0.05
        # delta = out[:,-1:,:]
        motor_seq.append(delta.squeeze(0))
        inp = delta
        if delta[0,0,2] > 1.5:
            break

    motor_seq = torch.stack(motor_seq)
    print(motor_seq.shape)
    # convert deltas to absolute coordinates
    x = torch.cumsum(motor_seq[:,:, 0], dim=0).detach().cpu().squeeze(1)
    y = torch.cumsum(motor_seq[:,:, 1], dim=0).detach().cpu().squeeze(1)
    pen = motor_seq[:,:, 2].detach().cpu().squeeze(1)
    t = motor_seq[:,:, 3].detach().cpu().squeeze(1)
    
    return x,y,pen,t