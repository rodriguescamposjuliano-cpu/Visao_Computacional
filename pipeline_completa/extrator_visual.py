import torch
import timm
import numpy as np
import cv2
from PIL import Image

class ExtratorVisual:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # Carrega o modelo DINOv2 pré-treinado pelo Facebook/Google.
        self.modelo = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m", # Oquab, M., et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision".
            pretrained=True, # Usa o conhecimento prévio do modelo
            num_classes=0 # Remove a camada de classificação para pegar apenas as características
        )

        self.modelo.eval().to(self.device)

        # Prepara as transformações necessárias (redimensionar, normalizar cores)
        # para que a foto do seu banco de dados fique no padrão que a IA entende.
        data_config = timm.data.resolve_model_data_config(self.modelo)
        self.transform = timm.data.create_transform(**data_config)

    def extrair(self, image_path):
        """
        Transforma uma foto em um vetor de números que descrevem a aparência.
        """

         # Carrega a imagem do disco usando OpenCV
        imagem = cv2.imread(image_path)

        if imagem is None:
            return None

        # Converte as cores de BGR (padrão OpenCV) para RGB (padrão mundial)
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        # Converte o formato de matriz (OpenCV) para objeto de imagem (PIL)
        # O modelo DINOv2 exige essa conversão para aplicar os filtros.
        imagem = Image.fromarray(imagem)

        # Aplica as transformações (ajusta brilho, contraste e tamanho)
        # O 'unsqueeze(0)' adiciona uma dimensão para simular um lote de imagens.
        img = self.transform(imagem).unsqueeze(0).to(self.device)

        # Passa a imagem pela rede neural (sem calcular gradientes para ser mais rápido)
        with torch.no_grad():
            features = self.modelo(img)

        # Transforma o resultado da IA em uma lista simples de números (NumPy)
        features = features.cpu().numpy().flatten()

        # L2 normalization
        features = features / (np.linalg.norm(features) + 1e-8)

        return features