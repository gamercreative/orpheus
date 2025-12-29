import torch
import torch.nn as nn

# model
class DrawerLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_size = 4
        self.hidden_size = 128
        self.num_layers = 3
        self.output_size = 4
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size,self.output_size)
        
    def forward(self,x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out,(hn,cn) = self.lstm(x,(h0,c0))

        out = self.norm(out)
        
        return self.output_layer(out)
    
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
        
        delta = out[:,-1:,:] + torch.rand_like(out[:,-1,:]) * 0.1
        motor_seq.append(delta.squeeze(0))
        inp = delta

    motor_seq = torch.stack(motor_seq)
    print(motor_seq.shape)
    # convert deltas to absolute coordinates
    x = torch.cumsum(motor_seq[:,:, 0], dim=0).detach().cpu()
    y = torch.cumsum(motor_seq[:,:, 1], dim=0).detach().cpu()
    
    return x,y