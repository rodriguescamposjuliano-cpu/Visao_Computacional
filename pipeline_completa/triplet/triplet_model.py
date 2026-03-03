import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNetwork(nn.Module):

    def __init__(self, input_dim, embedding_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1), # Troque ReLU por LeakyReLU
            nn.Dropout(0.2),

            nn.Linear(512, 512), # Camada extra de processamento
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            
            nn.Linear(512, embedding_dim) # Saída 818
        )

    def forward(self, x):
        embedding = self.network(x)

        # 🔥 Normalização L2 (ESSENCIAL para biometria)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding