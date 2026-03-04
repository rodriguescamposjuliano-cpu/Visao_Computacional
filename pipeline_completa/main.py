# ============================================================
# PIPELINE COMPLETO - VERSÃO AJUSTADA E SINCRONIZADA (COM AUGMENTATION)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from pipeline_identificacao_vacas import PipelineIdentificacaoVacas
from gerenciador_dataset import GerenciadorDataset
from identificador_vacas import IdentificadorVacas

from bimetria_vaca.utilitarios_dados import criar_triplas_de_comparacao
from bimetria_vaca.treinador_biometrico import TreinadorPorComparacao 

if __name__ == "__main__":

    # Configurações de Caminho
    POSE_MODEL_PATH = "runs/pose/trabalho_vaca/resultados/weights/best.pt"
    DATASET_FOLDER = "dataset_classificacao/"

    # ============================================================
    # ITEM 3 — FEATURE EXTRACTION
    # ============================================================
    print("\n[ITEM 3] Extração de features...")

    pipeline = PipelineIdentificacaoVacas(POSE_MODEL_PATH)
    dataset = GerenciadorDataset(DATASET_FOLDER, pipeline)

    # X: features (818 dim), cow_ids: nomes das pastas (strings), y_encoded: índices (int)
    X, cow_ids, y_encoded = dataset.load_dataset()

    # --- NOVO: DATA AUGMENTATION PARA GENERALIZAÇÃO ---
    # Isso resolve o problema de 100% no treino vs 60% na validação
    print("\n[AUGMENTATION] Criando variações para robustez na validação cega...")
    
    # Adiciona um ruído gaussiano leve (simula pequenas variações de pose)
    X_ruido_forte = X + np.random.normal(0, 0.04, X.shape) # Mais ruído
    X_escala_menor = X * 0.95                             # Simula vaca mais longe
    X_escala_maior = X * 1.05                             # Simula vaca mais perto

    X_final = np.vstack([X, X_ruido_forte, X_escala_menor, X_escala_maior])
    y_encoded_final = np.concatenate([y_encoded] * 4)
    cow_ids_final = np.concatenate([cow_ids] * 4)

    print(f"Shape após Augmentation: {X_final.shape}")
    print(f"Número total de vacas detectadas: {len(np.unique(cow_ids))}")

    # ============================================================
    # TREINAMENTO TRIPLET (MELHORIA DA REPRESENTAÇÃO)
    # ============================================================
    print("\n[TRIPLET] Gerando triplets para treinamento métrico...")
    # Usamos os dados aumentados para gerar os pares/trios
    triplets = criar_triplas_de_comparacao(X_final, y_encoded_final)
    print(f"Total de triplets gerados: {len(triplets)}")

    treinador_triplet = TreinadorPorComparacao(
        dimensao_entrada=X_final.shape[1], # Será 818
        dimensao_assinatura=128
    )

    # Aumentamos para 400 epochs para lidar com a maior massa de dados
    print("\n[TRIPLET] Treinando rede de embedding (100 epochs)...")
    treinador_triplet.executar_treinamento(triplets, epocas=100)

    print("\n[TRIPLET] Gerando embeddings finais...")
    X_embedding = treinador_triplet.extrair_assinatura_final(X_final)

    # ============================================================
    # ITEM 5 — MODEL TRAINING
    # ============================================================
    print("\n[ITEM 5] Treinando classificador final (XGBoost)...")

    # Instanciamos passando o encoder original do dataset
    identifier = IdentificadorVacas(label_encoder=dataset.label_encoder)

    # Split para validação rápida usando os dados aumentados
    X_train, X_test, y_train, y_test = train_test_split(
        X_embedding, cow_ids_final, test_size=0.15, stratify=cow_ids_final, random_state=42
    )

    identifier.treinar(X_train, y_train)

    # ============================================================
    # ITEM 6 — MODEL EVALUATION
    # ============================================================
    print("\n[ITEM 6] Avaliação do modelo (Hold-out)...")

    y_pred = identifier.classificar(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia no Teste (com Augmentation): {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ============================================================
    # CROSS-VALIDATION (5-FOLD)
    # ============================================================
    print("\n[ITEM 6] Iniciando Cross-validation 5-fold...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X_embedding, cow_ids_final)):
        modelo_cv = IdentificadorVacas(label_encoder=dataset.label_encoder)
        
        X_cv_train, X_cv_test = X_embedding[train_idx], X_embedding[test_idx]
        y_cv_train, y_cv_test = cow_ids_final[train_idx], cow_ids_final[test_idx]

        modelo_cv.treinar(X_cv_train, y_cv_train)
        preds_cv = modelo_cv.classificar(X_cv_test)

        score = accuracy_score(y_cv_test, preds_cv)
        cv_scores.append(score)
        print(f" > Dobra {i+1}: Acurácia = {score:.4f}")

    print(f"\nResultados Finais CV:")
    print(f" - Média: {np.mean(cv_scores):.4f}")
    print(f" - Desvio Padrão: {np.std(cv_scores):.4f}")

    # ============================================================
    # SALVAMENTO FINAL
    # ============================================================
    print("\n[FINAL] Exportando modelos...")

    torch.save(treinador_triplet.modelo.state_dict(), "modelo_embedding.pth")
    joblib.dump(identifier, "identificador_completo.pkl")
    joblib.dump(dataset.label_encoder, "label_encoder.pkl")

    print("Pipeline finalizado. Modelos robustos e prontos para o 'predict_vaca.py'!")