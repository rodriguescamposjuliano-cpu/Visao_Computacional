import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from bimetria_vaca.modelo_assinatura import RedeGeradoraDeAssinatura

class TreinadorPorComparacao:
    def __init__(self, dimensao_entrada, dimensao_assinatura=128):
        # Seleciona o hardware disponível (Placa de vídeo ou Processador)
        self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Inicializa a rede geradora de assinaturas
        self.modelo = RedeGeradoraDeAssinatura(dimensao_entrada, dimensao_assinatura).to(self.dispositivo)
        
        # Função de Perda Triplet: O "Juiz" que pune a rede se a vaca errada estiver perto da certa
        # margin=1.5 define a distância mínima de segurança entre vacas diferentes
        self.funcao_perda = nn.TripletMarginLoss(margin=1.0, p=2)
        
        # Otimizador: O algoritmo que ajusta os pesos da rede para diminuir o erro
        self.otimizador = torch.optim.AdamW(self.modelo.parameters(), lr=0.0002, weight_decay=1e-4)

    def executar_treinamento(self, triplas, epocas=500):
        # Converte as listas de triplas em Tensores (formato que a IA entende)
        ancoras = torch.tensor(np.stack([t[0] for t in triplas]), dtype=torch.float32)
        positivos = torch.tensor(np.stack([t[1] for t in triplas]), dtype=torch.float32)
        negativos = torch.tensor(np.stack([t[2] for t in triplas]), dtype=torch.float32)

        # Prepara o carregador de dados em lotes (batches) de 32 para processamento eficiente
        carregador = DataLoader(TensorDataset(ancoras, positivos, negativos), batch_size=32, shuffle=True)

        self.modelo.train() # Coloca o modelo em modo de treinamento
        for epoca in range(epocas):
            perda_total = 0
            for a, p, n in carregador:
                # Move os dados para o hardware escolhido (GPU/CPU)
                a, p, n = a.to(self.dispositivo), p.to(self.dispositivo), n.to(self.dispositivo)
                
                self.otimizador.zero_grad() # Limpa a memória de cálculos anteriores
                
                # A rede gera assinaturas para os três exemplos e calcula o erro
                perda = self.funcao_perda(self.modelo(a), self.modelo(p), self.modelo(n))
                
                perda.backward() # Calcula como ajustar a rede para errar menos
                self.otimizador.step() # Aplica os ajustes nos pesos da rede
                perda_total += perda.item()
            
            print(f"Época {epoca+1} | Erro de Separação: {perda_total/len(carregador):.4f}")

    def extrair_assinatura_final(self, X):
        # Usa a rede treinada para transformar dados brutos em identidades digitais
        self.modelo.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.dispositivo)
            return self.modelo(X_t).cpu().numpy()