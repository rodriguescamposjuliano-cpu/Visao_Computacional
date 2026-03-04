import numpy as np

class AnalistaBiometrico:
    # Keypoints
    CERNELHA        = 0
    DORSO           = 1
    ANCORA_SUP      = 2
    ANCORA_INF      = 3
    QUADRIL_CENTRAL = 4
    INSERCAO_CAUDA  = 5
    ISQUIO_SUP      = 6
    ISQUIO_INF      = 7

    # Variável de segurança para evitar divisão por zero
    ERRO_MINIMO = 1e-6 

    def normalizar(self, pontos):
        # Subtrai as coordenadas do quadril de todos os outros pontos.
        # Isso faz com que o quadril seja o ponto (0,0), centralizando o animal.
        centro = pontos[self.QUADRIL_CENTRAL]
        pontos = pontos - centro

        # Calcula a distância real entre a cernelha e a cauda (comprimento do tronco)
        tronco = np.linalg.norm(pontos[self.CERNELHA] - pontos[self.INSERCAO_CAUDA])

        # Se o tronco for quase zero (erro na foto), cancela para não dar erro matemático
        if tronco < self.ERRO_MINIMO:
            return None

        # Divide todos os pontos pelo tamanho do tronco. 
        # Assim, a vaca sempre terá "tamanho 1" no sistema, não importa a distância da câmera.
        return pontos / tronco

    def alinhar_rotacao(self, pontos):
        # Calcula a direção do corpo (vetor que vai da cernelha até a cauda)
        vetor_tronco = pontos[self.INSERCAO_CAUDA] - pontos[self.CERNELHA]
        
        # Descobre o ângulo de inclinação do animal na foto
        angulo = np.arctan2(vetor_tronco[1], vetor_tronco[0])

        # Cria uma Matriz de Rotação para girar o animal até que ele fique reto (horizontal)
        cos, sin = np.cos(-angulo), np.sin(-angulo)
        matriz_rotacao = np.array([[cos, -sin], [sin, cos]])

        # Multiplica os pontos pela matriz para aplicar o giro
        return pontos @ matriz_rotacao.T

    def dist(self, ponto_a, ponto_b):
        # Calcula a distância Euclidiana (linha reta) entre dois pontos
        return np.linalg.norm(ponto_a - ponto_b)

    def angulo(self, ponto_a, vertice, ponto_b):
        # Cria dois vetores a partir do ponto central (vértice)
        v_a = ponto_a - vertice
        v_b = ponto_b - vertice

        # Calcula o divisor (multiplicação das normas)
        divisor = (np.linalg.norm(v_a) * np.linalg.norm(v_b))

        if divisor < self.ERRO_MINIMO:
            return 0.0

        # Calcula o cosseno e transforma em graus (radianos)
        cosseno = np.dot(v_a, v_b) / divisor
        cosseno = np.clip(cosseno, -1.0, 1.0)
        return np.arccos(cosseno)

    def extrair(self, keypoints):
        # Centralização e Escala
        k_norm = self.normalizar(keypoints)
        if k_norm is None: return None

        # Alinhamento (Deixar a vaca "reta")
        k_alin = self.alinhar_rotacao(k_norm)

        features = []

        # Distâncias entre todos os pontos (Gera 28 combinações de medidas)
        for i in range(len(k_alin)):
            for j in range(i + 1, len(k_alin)):
                features.append(self.dist(k_alin[i], k_alin[j]))

        # Ângulos estruturais (Curvaturas do corpo)
        combinacoes = [
            (self.CERNELHA, self.DORSO, self.QUADRIL_CENTRAL),
            (self.DORSO, self.QUADRIL_CENTRAL, self.INSERCAO_CAUDA),
            (self.ANCORA_SUP, self.QUADRIL_CENTRAL, self.ANCORA_INF),
            (self.ISQUIO_SUP, self.INSERCAO_CAUDA, self.ISQUIO_INF)
        ]
        for p_a, v, p_b in combinacoes:
            features.append(self.angulo(k_alin[p_a], k_alin[v], k_alin[p_b]))

        # Vetores Direcionais - As coordenadas X e Y
        for i in range(len(k_alin)):
            features.append(k_alin[i][0]) # Coordenada X
            features.append(k_alin[i][1]) # Coordenada Y

        # Razões Biométricas - Proporções de largura vs comprimento
        larg_quadril = self.dist(k_alin[self.ANCORA_SUP], k_alin[self.ANCORA_INF])
        larg_isquio = self.dist(k_alin[self.ISQUIO_SUP], k_alin[self.ISQUIO_INF])
        comp_tronco = self.dist(k_alin[self.CERNELHA], k_alin[self.INSERCAO_CAUDA])

        features.append(larg_quadril / (comp_tronco + self.ERRO_MINIMO))
        features.append(larg_isquio / (larg_quadril + self.ERRO_MINIMO))

        # Transforma a lista em um Array para a Inteligência Artificial
        vetor_final = np.array(features, dtype=np.float32)

        # Verifica se houve erro matemático (NaN ou Infinito)
        if np.isnan(vetor_final).any() or np.isinf(vetor_final).any():
            return None

        return vetor_final