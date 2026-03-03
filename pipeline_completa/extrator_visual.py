import torch
import timm
import numpy as np
import cv2
from PIL import Image


class ExtratorVisual:
    def __init__(self, device="mps"):
        self.device = device

        self.model = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0
        )

        self.model.eval().to(self.device)

        # 🔥 configuração correta do modelo
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config)

    def extrair(self, image_path):

        imagem = cv2.imread(image_path)

        if imagem is None:
            return None

        # OpenCV → RGB
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        # 🔥 Converter para PIL (obrigatório)
        imagem = Image.fromarray(imagem)

        img = self.transform(imagem).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(img)

        feat = feat.cpu().numpy().flatten()

        # 🔥 L2 normalization
        feat = feat / (np.linalg.norm(feat) + 1e-8)

        return feat