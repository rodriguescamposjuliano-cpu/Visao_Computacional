import numpy as np
import random

def gerar_triplets(features, labels, num_triplets=10000):
    triplets = []
    labels_unicos = np.unique(labels)
    num_classes = len(labels_unicos)
    
    # Calcula quantos triplets cada vaca deve ter (ex: 10000 / 30 ≈ 333)
    triplets_por_classe = num_triplets // num_classes

    # Agrupa índices por classe
    indices_por_classe = {
        label: np.where(labels == label)[0]
        for label in labels_unicos
    }

    # Loop pelas classes para garantir o equilíbrio (Âncora fixa por classe)
    for classe_anchor in labels_unicos:
        indices_da_classe = indices_por_classe[classe_anchor]
        
        # Se a vaca tiver poucas fotos, permitimos repetição (replace=True) 
        # para não travar o código, mas o ideal é ter fotos suficientes.
        pode_repetir = len(indices_da_classe) < 2

        for _ in range(triplets_por_classe):
            # 1. Escolhe Âncora e Positivo da mesma vaca
            idx_anchor, idx_positive = np.random.choice(
                indices_da_classe,
                size=2,
                replace=pode_repetir
            )

            # 2. Escolhe uma classe negativa diferente
            classe_neg = random.choice([l for l in labels_unicos if l != classe_anchor])
            idx_negative = random.choice(indices_por_classe[classe_neg])

            triplets.append((
                features[idx_anchor],
                features[idx_positive],
                features[idx_negative]
            ))

    # Opcional: Embaralhar os triplets para o treino não ver uma vaca de cada vez
    random.shuffle(triplets)
    
    print(f"✅ Gerados {len(triplets)} triplets balanceados ({triplets_por_classe} por vaca).")
    return triplets