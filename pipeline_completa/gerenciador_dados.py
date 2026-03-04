import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from extrator_visual import ExtratorVisual

class GerenciadorDados:
    def __init__(self, dataset_folder, pipeline):
        self.dataset_folder = dataset_folder
        self.pipeline = pipeline
        self.label_encoder = LabelEncoder()
        self.extrator_visual = ExtratorVisual()

    def obtenha_informacoes(self):
        matriz_de_assinaturas, cow_ids = [], []
        fotos_descartadas = 0

        pastas = sorted([diretorio for diretorio in os.listdir(self.dataset_folder) 
                        if os.path.isdir(os.path.join(self.dataset_folder, diretorio))])

        for cow_id in pastas:
            pasta_vaca = os.path.join(self.dataset_folder, cow_id)
            imagens = [arquivo for arquivo in os.listdir(pasta_vaca) if arquivo.lower().endswith((".jpg", ".jpeg"))]
            
            print(f"Extraindo Biometria (Geo+Visual) da vaca {cow_id}...")

            for filename in imagens:
                caminhoDaImagem = os.path.join(pasta_vaca, filename)
                features_biometricas = self.pipeline.extrair_features_imagem(caminhoDaImagem)
                features_visuais = self.extrator_visual.extrair(caminhoDaImagem)

                if features_biometricas is None or features_visuais is None:
                    fotos_descartadas += 1
                    continue

                matriz_de_assinaturas.append(np.concatenate([features_biometricas, features_visuais]))
                cow_ids.append(cow_id)

        # Converte para o formato que os algoritmos de IA exigem (Array NumPy)
        X = np.array(matriz_de_assinaturas)

        # Transforma nomes em etiquetas numéricas
        y_encoded = self.label_encoder.fit_transform(cow_ids)

        print(f"\nSucesso: {len(X)} imagens | Falhas: {fotos_descartadas}")

        return X, np.array(cow_ids), y_encoded