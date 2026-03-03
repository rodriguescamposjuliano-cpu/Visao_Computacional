import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from extrator_visual import ExtratorVisual

class GerenciadorDataset:
    def __init__(self, dataset_folder, pipeline):
        self.dataset_folder = dataset_folder
        self.pipeline = pipeline
        self.label_encoder = LabelEncoder()
        self.extrator_visual = ExtratorVisual()

    def load_dataset(self):
        X, cow_ids = [], []
        failed = 0

        pastas = sorted([d for d in os.listdir(self.dataset_folder) 
                        if os.path.isdir(os.path.join(self.dataset_folder, d))])

        for cow_id in pastas:
            pasta_vaca = os.path.join(self.dataset_folder, cow_id)
            imagens = [f for f in os.listdir(pasta_vaca) if f.lower().endswith((".jpg", ".jpeg"))]
            
            print(f"Extraindo Biometria (Geo+Visual) da vaca {cow_id}...")

            for filename in imagens:
                path = os.path.join(pasta_vaca, filename)
                f_geo = self.pipeline.extrair_features_imagem(path)
                f_vis = self.extrator_visual.extrair(path)

                if f_geo is None or f_vis is None:
                    failed += 1
                    continue

                X.append(np.concatenate([f_geo, f_vis]))
                cow_ids.append(cow_id)

        X = np.array(X)
        y_encoded = self.label_encoder.fit_transform(cow_ids)
        print(f"\nSucesso: {len(X)} imagens | Falhas: {failed}")
        return X, np.array(cow_ids), y_encoded