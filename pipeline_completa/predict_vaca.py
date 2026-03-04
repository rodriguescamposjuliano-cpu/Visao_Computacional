import torch
import joblib
import numpy as np
import os
import cv2
from pipeline_identificacao_vacas import PipelineIdentificacaoVacas
from extrator_visual import ExtratorVisual
from bimetria_vaca.modelo_assinatura import RedeGeradoraDeAssinatura

class IdentificadorVacaPreditor:
    def __init__(self, input_dim=818):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Carregar os extratores (YOLO + DINOv2)
        self.pipeline = PipelineIdentificacaoVacas(os.getenv("CAMINHO_MODELO_POSE"))
        self.extrator_visual = ExtratorVisual()

        # Inicializar a rede de Embedding
        self.embed_net = RedeGeradoraDeAssinatura(input_dim=input_dim, embedding_dim=128)
        pesos_path = os.getenv("MODELO_EMBEDDING")

        if os.path.exists(pesos_path):
            self.embed_net.load_state_dict(torch.load(pesos_path, map_location=self.device))
            print(f"Pesos do Embedding carregados (Dim: {input_dim})")
        
        self.embed_net.to(self.device).eval()

        # Carregar o Classificador XGBoost
        self.identificador = joblib.load(os.getenv("CLASSIFICADOR_XGBOOST"))

    def desenhar_validacao(self, caminho_img, kps, id_real, id_pred, conf_yolo, pasta_out="validacao_manual"):
        """ Gera a prova visual com keypoints e predição """
        img = cv2.imread(caminho_img)
        if img is None: return
        
        os.makedirs(pasta_out, exist_ok=True)
        cor = (0, 255, 0) if str(id_real) == str(id_pred) else (0, 0, 255)
        
        # Desenhar Keypoints (Skeleton)
        for i, kp in enumerate(kps):
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Legenda com Status e Confiança do YOLO
        status = "CORRETO" if str(id_real) == str(id_pred) else "ERRO"
        info = f"Real: {id_real} | Pred: {id_pred} | YOLO Conf: {conf_yolo:.2f}"
        cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

        nome_arq = f"{status}_{id_real}_p{id_pred}_{os.path.basename(caminho_img)}"
        cv2.imwrite(os.path.join(pasta_out, nome_arq), img)

    def prever(self, caminho_imagem, id_real=None):
        # 1. Detecção e Filtro de Confiança (Sujestão 1)
        # Rodamos o YOLO manualmente para extrair confiança dos keypoints
        res_yolo = self.pipeline.modelo_pose(caminho_imagem, verbose=False)[0]
        
        if not res_yolo.keypoints or len(res_yolo.keypoints.data) == 0:
            return "Falha na detecção", 0

        # Média de confiança dos pontos detectados
        conf_media_kps = res_yolo.keypoints.conf[0].mean().item()
        kps_raw = res_yolo.keypoints.data[0].cpu().numpy()

        # FILTRO: Se a detecção for ruim, nem tentamos classificar
        if conf_media_kps < 0.50:
            return "Baixa Confiança YOLO", conf_media_kps

        # 2. Extração de Features (Geométrica + Visual)
        f_geo = self.pipeline.extrair_features_imagem(caminho_imagem)
        f_vis = self.extrator_visual.extrair(caminho_imagem)
        
        if f_geo is None or f_vis is None:
            return "Erro Features", conf_media_kps

        # Fusão e Embedding
        feat = np.concatenate([f_geo, f_vis])
        feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            emb = self.embed_net(feat_tensor).cpu().numpy()

        # 3. Classificação XGBoost
        id_predito = self.identificador.classificar(emb)[0]

        # 4. Geração da Prova Visual (Sugestão 3)
        if id_real:
            self.desenhar_validacao(caminho_imagem, kps_raw, id_real, id_predito, conf_media_kps)

        return id_predito, conf_media_kps

    def validar_pasta(self, pasta_val):
        print(f"\n--- Iniciando Validação com Prova Visual ---")
        total, acertos = 0, 0
        
        for cow_id in sorted(os.listdir(pasta_val)):
            sub = os.path.join(pasta_val, cow_id)
            if not os.path.isdir(sub): continue
            
            for img_name in os.listdir(sub):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    total += 1
                    caminho = os.path.join(sub, img_name)
                    
                    res, conf = self.prever(caminho, id_real=cow_id)
                    
                    status = "✅" if str(res) == str(cow_id) else "❌"
                    if status == "✅": acertos += 1
                    
                    print(f"ID: {cow_id} | Pred: {res} | YOLO: {conf:.2f} | {status}")
                    
        if total > 0:
            print(f"\nAcurácia: {(acertos/total)*100:.2f}% (Total: {total})")
            print(f"Verifique a pasta 'validacao_manual' para analisar os erros.")

if __name__ == "__main__":
    app = IdentificadorVacaPreditor(input_dim=818)
    app.validar_pasta("dataset_validacao")