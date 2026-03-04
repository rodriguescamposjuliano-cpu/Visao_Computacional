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
        
        # Função de Perda Triplet Margin: Garante que a assinatura da âncora esteja mais próxima da positiva do que da negativa por uma margem mínima
        # margin=1.0 define a distância mínima de segurança entre vacas diferentes
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
            for ancora, positivo, negativo in carregador:
                # Move os dados para o hardware escolhido
                ancora, positivo, negativo = ancora.to(self.dispositivo), positivo.to(self.dispositivo), negativo.to(self.dispositivo)
                
                # Limpa a memória de cálculos anteriores
                self.otimizador.zero_grad()
                
                # A rede gera assinaturas para os três exemplos e calcula o erro
                perda = self.funcao_perda(self.modelo(ancora), self.modelo(positivo), self.modelo(negativo))
                
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