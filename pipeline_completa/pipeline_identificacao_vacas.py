# ============================================================
# PIPELINE DE DETECÇÃO + EXTRAÇÃO DE FEATURES
# ============================================================

from ultralytics import YOLO
from analista_biometrico import AnalistaBiometrico
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

class PipelineIdentificacaoVacas:

    def __init__(self, caminho_modelo_pose):
        """
        Inicializa o pipeline carregando:
        - Modelo YOLO para detecção de pose
        - Extrator responsável por gerar as features das vacas
        """
        self.modelo_pose = YOLO(caminho_modelo_pose)
        self.extrator_features = AnalistaBiometrico()

    # ------------------------------------------------
    # EXTRAÇÃO DE FEATURES A PARTIR DE UMA IMAGEM
    # ------------------------------------------------
    def extrair_features_imagem(self, caminho_imagem):
        """
        Executa:
        1. Detecção da vaca via YOLO
        2. Extração dos keypoints
        3. Conversão dos keypoints em vetor de features
        """

        try:
            resultados = self.modelo_pose(caminho_imagem)

            # Verifica se houve detecção de keypoints
            if resultados[0].keypoints is None:
                logging.warning(f"Nenhuma vaca detectada em {caminho_imagem}")
                return None

            keypoints_todas_vacas = resultados[0].keypoints.xy.cpu().numpy()

            if len(keypoints_todas_vacas) == 0:
                logging.warning(f"Nenhuma vaca válida encontrada em {caminho_imagem}")
                return None

            # Considera apenas a primeira vaca detectada
            keypoints_primeira_vaca = keypoints_todas_vacas[0]

            vetor_features = self.extrator_features.extrair(keypoints_primeira_vaca)

            return vetor_features

        except Exception as erro:
            logging.error(f"Erro ao processar {caminho_imagem}: {str(erro)}")
            return None