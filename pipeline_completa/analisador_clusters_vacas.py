# ============================================================
# ITEM 4 — ANÁLISE DE FEATURES (CLUSTERIZAÇÃO)
# ============================================================

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class AnalisadorClustersVacas:

    def __init__(self, maximo_clusters=10):
        self.maximo_clusters = maximo_clusters
        self.melhor_k = None
        self.melhor_silhouette = -1
        self.normalizador = StandardScaler()
        self.modelo = None

    # --------------------------------------------------------
    # NORMALIZAÇÃO DAS FEATURES
    # --------------------------------------------------------
    def normalizar_dados(self, matriz_features):
        return self.normalizador.fit_transform(matriz_features)

    # --------------------------------------------------------
    # MÉTODO DO COTOVELO
    # --------------------------------------------------------
    def metodo_cotovelo(self, matriz_features):

        dados_normalizados = self.normalizar_dados(matriz_features)

        lista_inercias = []
        valores_k = range(1, min(self.maximo_clusters, len(dados_normalizados)))

        for k in valores_k:
            modelo_kmeans = KMeans(n_clusters=k, random_state=42)
            modelo_kmeans.fit(dados_normalizados)
            lista_inercias.append(modelo_kmeans.inertia_)

        plt.figure()
        plt.plot(valores_k, lista_inercias, marker='o')
        plt.xlabel("Número de Clusters (k)")
        plt.ylabel("Inércia")
        plt.title("Método do Cotovelo")
        plt.show()

    # --------------------------------------------------------
    # DESCOBERTA AUTOMÁTICA DO NÚMERO DE VACAS
    # --------------------------------------------------------
    def descobrir_numero_vacas(self, matriz_features):

        dados_normalizados = self.normalizar_dados(matriz_features)

        valores_k = range(2, min(self.maximo_clusters, len(dados_normalizados)))
        lista_silhouette = []

        for k in valores_k:
            modelo_kmeans = KMeans(n_clusters=k, random_state=42)
            rotulos = modelo_kmeans.fit_predict(dados_normalizados)

            score = silhouette_score(dados_normalizados, rotulos)
            lista_silhouette.append(score)

            print(f"k={k} | Silhouette={score:.4f}")

            if score > self.melhor_silhouette:
                self.melhor_silhouette = score
                self.melhor_k = k

        print("\nMelhor número de clusters:", self.melhor_k)
        print("Melhor Silhouette Score:", self.melhor_silhouette)

        plt.figure()
        plt.plot(valores_k, lista_silhouette, marker='o')
        plt.xlabel("Número de Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title("Escolha automática do número de clusters")
        plt.show()

        return self.melhor_k

    # --------------------------------------------------------
    # TREINAR MODELO FINAL
    # --------------------------------------------------------
    def treinar_modelo(self, matriz_features):

        if self.melhor_k is None:
            raise ValueError("Execute descobrir_numero_vacas() primeiro.")

        dados_normalizados = self.normalizador.transform(matriz_features)

        self.modelo = KMeans(
            n_clusters=self.melhor_k,
            random_state=42
        )

        rotulos = self.modelo.fit_predict(dados_normalizados)
        return rotulos

    # --------------------------------------------------------
    # CLASSIFICAR NOVOS DADOS
    # --------------------------------------------------------
    def classificar(self, matriz_features):

        if self.modelo is None:
            raise ValueError("Modelo não treinado.")

        dados_normalizados = self.normalizador.transform(matriz_features)
        return self.modelo.predict(dados_normalizados)

    # --------------------------------------------------------
    # PROJEÇÃO DOS DADOS UTILIZANDO PCA (Principal Component Analysis)
    # PCA significa "Análise de Componentes Principais".
    # É uma técnica de redução de dimensionalidade que transforma
    # múltiplas variáveis (features) em um número menor de
    # componentes, preservando a maior variância possível dos dados.
    #
    # Neste caso, os dados são reduzidos para 2 dimensões
    # para permitir a visualização gráfica dos clusters
    # e facilitar a interpretação da separação entre os grupos.
    # --------------------------------------------------------
    def visualizar_pca(self, matriz_features, rotulos):

        dados_normalizados = self.normalizador.transform(matriz_features)

        pca = PCA(n_components=2)
        dados_pca = pca.fit_transform(dados_normalizados)

        plt.figure()
        plt.scatter(dados_pca[:, 0], dados_pca[:, 1], c=rotulos)
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.title("Visualização dos Clusters de Vacas (PCA)")
        plt.show()