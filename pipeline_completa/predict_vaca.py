import torch
import joblib
import numpy as np
import os
from pipeline_identificacao_vacas import PipelineIdentificacaoVacas
from extrator_visual import ExtratorVisual
from triplet.triplet_model import EmbeddingNetwork

class IdentificadorVacaPreditor:
    def __init__(self, pose_model_path="runs/pose/trabalho_vaca/resultados/weights/best.pt", input_dim=818):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # 1. Carregar os extratores (YOLO + DINOv2)
        self.pipeline = PipelineIdentificacaoVacas(pose_model_path)
        self.extrator_visual = ExtratorVisual()

        # 2. Inicializar a rede de Embedding com a dimensão correta (818)
        self.embed_net = EmbeddingNetwork(input_dim=input_dim, embedding_dim=128)
        
        # Carregar os pesos treinados no Main.py
        pesos_path = "modelo_embedding.pth"
        if os.path.exists(pesos_path):
            self.embed_net.load_state_dict(torch.load(pesos_path, map_location=self.device))
            print(f"✅ Pesos do Embedding carregados com sucesso (Dim: {input_dim})")
        else:
            print("❌ Erro: Arquivo modelo_embedding.pth não encontrado!")
            
        self.embed_net.to(self.device).eval()

        # 3. Carregar o Classificador XGBoost
        self.identificador = joblib.load("identificador_completo.pkl")

    def prever(self, caminho_imagem):
        # Extração Geométrica
        f_geo = self.pipeline.extrair_features_imagem(caminho_imagem)
        # Extração Visual (DINOv2)
        f_vis = self.extrator_visual.extrair(caminho_imagem)
        
        if f_geo is None or f_vis is None:
            return "Falha na detecção"

        # Fusão exata como no treino
        feat = np.concatenate([f_geo, f_vis])
        
        # Converter para tensor
        feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Passar pela rede de embedding
            emb = self.embed_net(feat_tensor).cpu().numpy()

        # Classificar com XGBoost
        return self.identificador.classificar(emb)[0]

    def validar_pasta(self, pasta_val):
        print(f"\n--- Iniciando Validação Cega (Modo 818 Dimensões) ---")
        total, acertos = 0, 0
        
        if not os.path.exists(pasta_val):
            print(f"Pasta {pasta_val} não encontrada.")
            return

        for cow_id in sorted(os.listdir(pasta_val)):
            sub = os.path.join(pasta_val, cow_id)
            if not os.path.isdir(sub): continue
            
            for img in os.listdir(sub):
                if img.lower().endswith((".jpg", ".jpeg")):
                    total += 1
                    res = self.prever(os.path.join(sub, img))
                    status = "✅" if str(res) == str(cow_id) else "❌"
                    if status == "✅": acertos += 1
                    print(f"Pasta Real: {cow_id} | Predito: {res} | Status: {status}")
                    
        if total > 0:
            print(f"\nAcurácia na Validação Cega: {(acertos/total)*100:.2f}%")

if __name__ == "__main__":
    # IMPORTANTE: input_dim deve ser 818 (50 do YOLO + 768 do DINOv2)
    app = IdentificadorVacaPreditor(input_dim=818)
    app.validar_pasta("dataset_validacao")