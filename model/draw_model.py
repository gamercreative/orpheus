import torch
import torch.nn as nn
import data
import utils

# model
class DrawerLSTM(nn.Module):
    def __init__(self,device) -> None:
        super().__init__()

        self.device = device
    
        self.embedding_size = 26 + 2 # 26 letters and 2 tokens
        self.embedding_dim = 64
        self.input_size = 4
        self.hidden_size = 48
        self.num_layers = 3
        self.output_size = 6 # dx , dy , dt , pen_0 , pen_1 , pen_2
        
        self.start_motor = nn.Parameter(torch.randn(4))
        self.end_motor   = nn.Parameter(torch.randn(4))
        
        self.letter_to_h = nn.Linear(self.embedding_dim, self.hidden_size)
        self.letter_to_c = nn.Linear(self.embedding_dim, self.hidden_size)
        
        self.embed = nn.Embedding(self.embedding_size,self.embedding_dim)
        self.cell = nn.LSTMCell(self.input_size,self.hidden_size)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size,self.output_size)

    def forward(self,x,embed):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        h0[0] = self.letter_to_h(embed)
        c0[0] = self.letter_to_c(embed)

        out,(hn,cn) = self.lstm(x,(h0,c0)) # no need to transfer the contetx

        out = self.norm(out)
        
        return self.output_layer(out)

def DrawOut(model,embed,steps): # autoregressive inference
    model.eval()
    h0 = torch.zeros(model.num_layers,1,model.hidden_size, device=model.device)
    c0 = torch.zeros(model.num_layers,1,model.hidden_size, device=model.device)
    
    h0[0] = model.letter_to_h(embed)
    c0[0] = model.letter_to_c(embed)
    
    inp = data.CreateStartToken(model)

    motor_seq = []
    for _ in range(steps):
        out,(h0,c0) = model.lstm(inp,(h0,c0))
        out = model.norm(out)
        out = model.output_layer(out)
        
        # delta = out[:,-1:,:] + torch.rand_like(out[:,-1,:]) * 0.05
        delta = out[:,-1:,:]
        
        motor_xy = out[0, -1, 0:2]           # dx, dy
        motor_dt = out[0, -1, 2]              # dt
        pen_logits = out[0, -1, 3:6]          # logts
        pen_state = utils.GetPenStateFromOut(pen_logits)  # integer 0,1,2
        
        delta = torch.stack([motor_xy[0], motor_xy[1], pen_state.float(), motor_dt]).view(1,1,4)
        
        motor_seq.append(delta.squeeze(0))
        inp = delta
        
        if pen_state == 2:
            break

    motor_seq = torch.stack(motor_seq)
    print(motor_seq.shape)
    # convert deltas to absolute coordinates
    x = torch.cumsum(motor_seq[:,:, 0], dim=0).detach().cpu().squeeze(1)
    y = torch.cumsum(motor_seq[:,:, 1], dim=0).detach().cpu().squeeze(1)
    pen = motor_seq[:,:, 2].detach().cpu().squeeze(1)
    t = motor_seq[:,:, 3].detach().cpu().squeeze(1)
    
    return x,y,pen,t