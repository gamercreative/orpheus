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
        x = []
        y = []
        for letter in self.letters:
            x.append(letter.X)
            y.append(letter.Y)
            
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)
            
        return x,y
            
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
        start_token = model.embed(utils.START_TOKEN_ID)
        end_token = model.embed(utils.END_TOKEN_ID)
        embed = model.embed(self.letter_id)

        self.X = AssignStrokeToLetter(self.X,embed)
        self.X = AssignStartToken(self.X, start_token)
        self.Y = AssignEndToken(self.Y, end_token)

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
    Y = motor_seq[:,1:,:]
    
    return X,Y

def AssignStrokeToLetter(X, embed):
    base_embed = embed.unsqueeze(0).unsqueeze(0)
    print(base_embed.shape)
    print(X.shape)
    if X.size(0) > base_embed.size(0):
        base_embed = base_embed.repeat(X.size(0),1,1) # batch timestamp features
    
    if X.size(1) > base_embed.size(1):
        base_embed = base_embed.repeat(1,X.size(1),1) # batch timestamp->X features
    
    X = torch.cat([X,base_embed],dim=2)
    
    print(X.shape)
        
    return X

def AssignEndToken(Y,embed):
    if Y == None:
        end_token = torch.tensor(embed,device=device).unsqueeze(0).unsqueeze(0)
        return end_token
    
    end_token = torch.tensor(embed,device=Y.device).unsqueeze(0).unsqueeze(0)
    if Y.size(0) > end_token.size(0):
        end_token = end_token.repeat(Y.size(0),1,1)
        
    Y = torch.cat([Y,end_token],dim=1)
    
    return Y

def AssignStartToken(X,embed):
    if X == None:
        start_token = torch.tensor(embed,device=device).unsqueeze(0).unsqueeze(0)
        return start_token
    start_token = embed.unsqueeze(1).unsqueeze(0)
    print(embed.shape)
    
    if X.size(0) > start_token.size(0):
        start_token = start_token.repeat(X.size(0),1,1)

    X = torch.cat([start_token,X],dim=1)
    
    return X