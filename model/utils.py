import os
import torch

def GetFiles(dir):
    dir = dir.strip()
    files = []
    for file in os.listdir(dir):
        file_path = dir + "/" + file 
        files.append(file_path)
        
    return files

def ExtractCharFromFileName(file):
    filename_parts = file.split("/")
    file_char = filename_parts[-1][0]
    
    return file_char

def CharToId(letter):
    # Start_token_ID = 26
    # end_token_ID = 27
    letter_to_id = {chr(i): i for i in range(97,123)}

    return torch.tensor(letter_to_id[letter]).to(device)

def GetDevice():
    if torch.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    return device

device = GetDevice()

START_TOKEN_ID = torch.tensor([26]).squeeze(0).to(device)
END_TOKEN_ID = torch.tensor([27]).squeeze(0).to(device)
