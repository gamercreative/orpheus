import json
import torch
from torch.nn.utils.rnn import pad_sequence

if torch.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def LoadData(path):
    #data
    with open(path,"r") as f:
        data = json.load(f)

    sequences = []
    for stroke in data:
        seq = list(zip(stroke["dx"],stroke["dy"],stroke["pen_state"],stroke["dt"]))
        sequences.append(seq)

    tensors = []
    for seq in sequences:
        tens = torch.tensor(seq, dtype=torch.float32)
        tensors.append(tens)

    motor_seq = pad_sequence(tensors, batch_first=True , padding_value=0.0)

    X = motor_seq[:,:-1,:]
    Y = motor_seq[:,1:,:]
    
    return X,Y

def AssignStrokeToLetter(X, embed):
    base_embed = torch.tensor(embed, dtype=torch.float32,device=X.device).unsqueeze(0).unsqueeze(0)
    if X.size(0) > base_embed.size(0):
        base_embed = base_embed.repeat(X.size(0),1,1) # batch timestamp features
    
    if X.size(1) > base_embed.size(1):
        base_embed = base_embed.repeat(1,X.size(1),1) # batch timestamp->X features
    
    X = torch.cat([X,base_embed],dim=2)
        
    return X

def AssignEndToken(Y):
    end_tok = [-0.1, -0.1, -1.0, -0.1]
    if Y == None:
        end_token = torch.tensor(end_tok,device=device).unsqueeze(0).unsqueeze(0)
        return end_token
    
    end_token = torch.tensor(end_tok,device=Y.device).unsqueeze(0).unsqueeze(0)
    if Y.size(0) > end_token.size(0):
        end_token = end_token.repeat(Y.size(0),1,1)
        
    Y = torch.cat([Y,end_token],dim=1)
    
    return Y

def AssignStartToken(X):
    start_tok = [0.1, 0.1, 1.0, 0.1]
    if X == None:
        start_token = torch.tensor(start_tok,device=device).unsqueeze(0).unsqueeze(0)
        return start_token
    start_token = torch.tensor(start_tok,device=X.device).unsqueeze(0).unsqueeze(0)
    
    if X.size(0) > start_token.size(0):
        start_token = start_token.repeat(X.size(0),1,1)

    X = torch.cat([start_token,X],dim=1)
    
    return X