import torch
import torch.nn as nn
import torch

EMBEDDING_SIZE = 512

class latentModel(nn.Module):
    def __init__(self):
        super().__init__()
        #self.GRU = nn.GRU(EMBEDDING_SIZE, EMBEDDING_SIZE, 2) 
        self.fc1 = nn.Linear(EMBEDDING_SIZE * 2, EMBEDDING_SIZE  + EMBEDDING_SIZE // 2)
        self.fc2 = nn.Linear(EMBEDDING_SIZE  + EMBEDDING_SIZE // 2, EMBEDDING_SIZE)

        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2) 

    def forward(self, imageEmbedding, missionEmbedding):
        #imageEmbedding = self.GRU(imageEmbedding)
        combined = torch.cat((imageEmbedding, missionEmbedding), dim=1)
        x = self.fc1(combined)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        #x = self.leaky_relu(x)
        #x = self.dropout(x)

        #x = self.fc3(x)
        return x
