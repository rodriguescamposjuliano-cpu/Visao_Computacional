import numpy as np
import random

# Função que cria os exemplos de comparação (Âncora, Mesma Vaca, Vaca Diferente)
def criar_triplas_de_comparacao(features, rotulos, total_triplas=10000):
    lista_triplas = []
    # Lista de IDs de todas as 30 vacas
    ids_unicos = np.unique(rotulos) 
    total_classes = len(ids_unicos)
    
    # Garante que cada vaca tenha a mesma importância no treino
    triplas_por_vaca = total_triplas // total_classes

    # Mapeia onde estão as fotos de cada vaca no dataset
    indices_por_vaca = {
        id_vaca: np.where(rotulos == id_vaca)[0]
        for id_vaca in ids_unicos
    }

    # Loop para gerar as triplas de forma equilibrada
    for id_ancora in ids_unicos:
        indices_da_vaca = indices_por_vaca[id_ancora]
        
        # Verifica se a vaca tem fotos suficientes para comparação
        pode_repetir = len(indices_da_vaca) < 2

        for _ in range(triplas_por_vaca):
            # Escolhe a Âncora e um Positivo (duas fotos da mesma vaca)
            idx_ancora, idx_positivo = np.random.choice(
                indices_da_vaca,
                size=2,
                replace=pode_repetir
            )

            # Escolhe um Negativo (uma foto de qualquer outra vaca diferente)
            id_negativo = random.choice([l for l in ids_unicos if l != id_ancora])
            idx_negativo = random.choice(indices_por_vaca[id_negativo])

            # Adiciona o trio na lista: [Foto_A, Outra_Foto_A, Foto_B]
            lista_triplas.append((
                features[idx_ancora],
                features[idx_positivo],
                features[idx_negativo]
            ))

    # Embaralha os trios para que a rede não aprenda uma vaca por vez
    random.shuffle(lista_triplas)
    
    print(f"Sucesso: {len(lista_triplas)} triplas geradas ({triplas_por_vaca} por vaca).")
    return lista_triplas