import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from triplet.triplet_model import EmbeddingNetwork

class TreinadorTriplet:
    def __init__(self, input_dim, embedding_dim=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = EmbeddingNetwork(input_dim, embedding_dim).to(self.device)
        
        # Margem de 0.5 para biometria é o "padrão ouro" para separação real
        self.loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0002, weight_decay=1e-4)

    def treinar(self, triplets, epochs=500):
        anchors = torch.tensor(np.stack([t[0] for t in triplets]), dtype=torch.float32)
        positives = torch.tensor(np.stack([t[1] for t in triplets]), dtype=torch.float32)
        negatives = torch.tensor(np.stack([t[2] for t in triplets]), dtype=torch.float32)

        loader = DataLoader(TensorDataset(anchors, positives, negatives), batch_size=32, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for a, p, n in loader:
                a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(a), self.model(p), self.model(n))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            #if (epoch + 1) % 50 == 0:
            print(f"Época {epoch+1} | Loss Triplet: {total_loss/len(loader):.4f}")

    def gerar_embedding(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(X_t).cpu().numpy()