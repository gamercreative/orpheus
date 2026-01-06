import json
import torch
from torch.nn.utils.rnn import pad_sequence
import utils

device = utils.GetDevice()
class Dataset:
    def __init__(self,dir):
        self.dir = dir
        self.letters = []
        self.GetStrokeDataset(dir)

    def GetStrokeDataset(self,dir):
        files = utils.GetFiles(dir)
        
        for file in files:
            file_lett = utils.ExtractCharFromFileName(file)
            letter_id = utils.CharToId(file_lett)

            self.letters.append(LetterDataset(file,letter_id))
            
    def PrepareDatasetEmbeddings(self,model):
        for letter in self.letters:
            letter.PrepareStrokeEmbeddings(model)
            
    def GetXY(self):
        xy = []
        for letter in self.letters:
            xy.append((letter.X.detach().to(device), letter.Y.detach().to(device)))

        return xy
    
class LetterDataset:
    def __init__(self,path,letter_id):
        self.letter_id = letter_id
        self.letter_path = path
        self.X = []
        self.Y = []
        self.LoadLetterData()
        
    def LoadLetterData(self):
        x,y = LoadData(self.letter_path)
        self.X = x.to(device)
        self.Y = y.to(device)
        
    def PrepareStrokeEmbeddings(self,model):
        start_token = CreateStartToken(model)
        end_token = CreateEndToken(model)
        
        embed = model.embed(self.letter_id)

        self.X = AssignStrokeToLetter(self.X,embed)
        self.X = AssignStartToken(self.X, start_token)
        self.X = AssignEndTokenX(self.X, end_token)
        self.Y = AssignEndTokenY(self.Y)

def LoadData(path):
    #data
    with open(path,"r") as f:
        data = json.load(f)

    sequences = []
    for stroke in data:
        seq = list(zip(stroke["dx"],stroke["dy"],stroke["pen_state"],stroke["dt"])) # select features to append from json
        sequences.append(seq)

    tensors = []
    for seq in sequences:
        tens = torch.tensor(seq, dtype=torch.float32)
        tensors.append(tens)

    motor_seq = pad_sequence(tensors, batch_first=True , padding_value=0.0)

    X = motor_seq[:,:-1,:]
    Y = motor_seq
    
    return X,Y

def CreateStartToken(model):
    # creates the toekn as 1,1,68 batch,timstamp,feature
    start_token_embed = model.embed(utils.START_TOKEN_ID).unsqueeze(0).unsqueeze(0)
    start_token_motor = model.start_motor.unsqueeze(0).unsqueeze(0)
    
    start_token_embed = start_token_embed.detach()
    start_token_motor = start_token_motor.detach()
    
    return torch.cat([start_token_motor,start_token_embed],dim=2).to(device)

def CreateEndToken(model):
    # creates the toekn as 1,1,68 batch,timstamp,feature
    end_token_embed = model.embed(utils.END_TOKEN_ID).unsqueeze(0).unsqueeze(0)
    end_token_motor = model.end_motor.unsqueeze(0).unsqueeze(0)
    
    end_token_embed = end_token_embed.detach()
    end_token_motor = end_token_motor.detach()
    
    return torch.cat([end_token_motor, end_token_embed],dim=2).to(device)

def AssignStrokeToLetter(X, embed):
    base_embed = embed.unsqueeze(0).unsqueeze(0)

    if X.size(0) > base_embed.size(0):
        base_embed = base_embed.repeat(X.size(0),1,1) # batch timestamp features
    
    if X.size(1) > base_embed.size(1):
        base_embed = base_embed.repeat(1,X.size(1),1) # batch timestamp->X features
    
    X = torch.cat([X,base_embed],dim=2)
        
    return X

def AssignStartToken(X,embed):
    start_token = embed
    if X.size(0) > start_token.size(0):
        start_token = start_token.repeat(X.size(0),1,1)
    
    X = torch.cat([start_token,X],dim=1)
    
    return X

def AssignEndTokenX(X,embed):
    end_token = embed
    if X.size(0) > end_token.size(0):
        end_token = end_token.repeat(X.size(0),1,1)

    X = torch.cat([X,end_token],dim=1)
    
    return X

def AssignEndTokenY(Y):
    end_token_motor = [0,0,2,0]
    end_token = torch.tensor(end_token_motor,device=device)
    
    if Y.size(0) > end_token.size(0):
        end_token = end_token.repeat(Y.size(0),1,1)
        
    Y = torch.cat([Y,end_token], dim=1)
    
    return Y