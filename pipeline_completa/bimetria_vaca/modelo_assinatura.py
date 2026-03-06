import torch
import torch.nn as nn
import torch.nn.functional as F

# Classe que gera a "Assinatura Digital" ou "Identidade Matemática" da vaca
class RedeGeradoraDeAssinatura(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super().__init__()

        self.network = nn.Sequential(
             nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            

           
            
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        embedding = self.network(x)

        # Normalização L2 (ESSENCIAL para biometria)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding