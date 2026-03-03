import numpy as np


class ExtratorFeaturesVacas:

    CERNELHA        = 0
    DORSO           = 1
    ANCORA_SUP      = 2
    ANCORA_INF      = 3
    QUADRIL_CENTRAL = 4
    INSERCAO_CAUDA  = 5
    ISQUIO_SUP      = 6
    ISQUIO_INF      = 7

    EPSILON = 1e-6

    # -------------------------------------------------------
    # 1️⃣ NORMALIZAÇÃO DE POSIÇÃO E ESCALA
    # -------------------------------------------------------
    def normalizar(self, pontos):

        # centraliza no quadril
        centro = pontos[self.QUADRIL_CENTRAL]
        pontos = pontos - centro

        # escala pelo comprimento tronco
        tronco = np.linalg.norm(
            pontos[self.CERNELHA] - pontos[self.INSERCAO_CAUDA]
        )

        if tronco < self.EPSILON:
            return None

        pontos = pontos / tronco
        return pontos

    # -------------------------------------------------------
    # 2️⃣ ALINHAMENTO ROTACIONAL (invariância a rotação)
    # -------------------------------------------------------
    def alinhar_rotacao(self, pontos):

        vetor_tronco = pontos[self.INSERCAO_CAUDA] - pontos[self.CERNELHA]

        angulo = np.arctan2(vetor_tronco[1], vetor_tronco[0])

        # matriz de rotação inversa
        cos = np.cos(-angulo)
        sin = np.sin(-angulo)

        R = np.array([
            [cos, -sin],
            [sin,  cos]
        ])

        return pontos @ R.T

    # -------------------------------------------------------
    # 3️⃣ DISTÂNCIA ENTRE DOIS PONTOS
    # -------------------------------------------------------
    def dist(self, a, b):
        return np.linalg.norm(a - b)

    # -------------------------------------------------------
    # 4️⃣ ÂNGULO ENTRE 3 PONTOS
    # -------------------------------------------------------
    def angulo(self, a, v, b):

        va = a - v
        vb = b - v

        denom = (np.linalg.norm(va) * np.linalg.norm(vb))

        if denom < self.EPSILON:
            return 0.0

        cos = np.dot(va, vb) / denom
        cos = np.clip(cos, -1.0, 1.0)

        return np.arccos(cos)

    # -------------------------------------------------------
    # EXTRAÇÃO COMPLETA
    # -------------------------------------------------------
    def extrair(self, keypoints):

        k = self.normalizar(keypoints)
        if k is None:
            return None

        k = self.alinhar_rotacao(k)

        features = []

        # =====================================================
        # 1️⃣ TODAS DISTÂNCIAS ENTRE PARES (28 features)
        # =====================================================
        for i in range(len(k)):
            for j in range(i + 1, len(k)):
                features.append(self.dist(k[i], k[j]))

        # =====================================================
        # 2️⃣ ÂNGULOS IMPORTANTES
        # =====================================================
        combinacoes = [
            (self.CERNELHA, self.DORSO, self.QUADRIL_CENTRAL),
            (self.DORSO, self.QUADRIL_CENTRAL, self.INSERCAO_CAUDA),
            (self.ANCORA_SUP, self.QUADRIL_CENTRAL, self.ANCORA_INF),
            (self.ISQUIO_SUP, self.INSERCAO_CAUDA, self.ISQUIO_INF)
        ]

        for a, v, b in combinacoes:
            features.append(self.angulo(k[a], k[v], k[b]))

        # =====================================================
        # 3️⃣ VETORES DIRECIONAIS (assinatura estrutural)
        # =====================================================
        for i in range(len(k)):
            features.append(k[i][0])
            features.append(k[i][1])

        # =====================================================
        # 4️⃣ RAZÕES BIOMÉTRICAS IMPORTANTES
        # =====================================================
        largura_quadril = self.dist(k[self.ANCORA_SUP], k[self.ANCORA_INF])
        largura_isquio = self.dist(k[self.ISQUIO_SUP], k[self.ISQUIO_INF])
        comprimento_tronco = self.dist(k[self.CERNELHA], k[self.INSERCAO_CAUDA])

        features.append(largura_quadril / (comprimento_tronco + self.EPSILON))
        features.append(largura_isquio / (largura_quadril + self.EPSILON))

        vetor_final = np.array(features, dtype=np.float32)

        if np.isnan(vetor_final).any() or np.isinf(vetor_final).any():
            return None

        return vetor_final