import json
import torch
from torch.nn.utils.rnn import pad_sequence

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

def AssignStrokeToLetter(X,Y, embed):
    base_embed = torch.tensor(embed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    base_embed = base_embed.repeat(Y.size(0),1,1)

    X = torch.cat([base_embed,X],dim=1)
    
    return X