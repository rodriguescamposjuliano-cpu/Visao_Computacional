# =====================================================
# PIPELINE COMPLETO - Identificação Individual de Vacas
# =====================================================
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
from pipeline_identificacao_vacas import PipelineIdentificacaoVacas
from gerenciador_dados import GerenciadorDados
from identificador_vacas import IdentificadorVacas
from bimetria_vaca.utilitarios_dados import criar_triplas_de_comparacao
from bimetria_vaca.treinador_biometrico import TreinadorPorComparacao 

load_dotenv()

if __name__ == "__main__":

    # Configurações de Caminho
    POSE_MODEL_PATH = os.getenv("CAMINHO_MODELO_POSE")
    DATASET_FOLDER = os.getenv("PASTA_DATASET_CLASSIFICACAO")

    # ==================================================================
    # 3 — Generate features that could potentially identify each animal
    # ==================================================================
    
    pipeline = PipelineIdentificacaoVacas(POSE_MODEL_PATH)
    gerenciadorDados = GerenciadorDados(DATASET_FOLDER, pipeline)

    # X: features (818 dim), cow_ids: nomes das pastas (strings), y_encoded: índices (int)
    X, cow_ids, y_encoded = gerenciadorDados.obtenha_informacoes()

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
    
    # Usamos os dados aumentados para gerar os pares/trios
    triplets = criar_triplas_de_comparacao(X_final, y_encoded_final)
    print(f"Total de triplets gerados: {len(triplets)}")

    treinador_triplet = TreinadorPorComparacao(
        dimensao_entrada=X_final.shape[1],
        dimensao_assinatura=128
    )

    treinador_triplet.executar_treinamento(triplets, epocas=100)

    # Obtém os embeddings finais para o classificador
    X_embedding = treinador_triplet.extrair_assinatura_final(X_final)

    # =====================================================================
    # 5 — Design and train a machine learning model to classify the animals
    # =====================================================================

    # Classificador XGBoost
    identificador = IdentificadorVacas(label_encoder=gerenciadorDados.label_encoder)

    # Split para validação rápida usando os dados aumentados
    X_train, X_test, y_train, y_test = train_test_split(
        X_embedding, cow_ids_final, test_size=0.15, stratify=cow_ids_final, random_state=42
    )

    identificador.treinar(X_train, y_train)

    # ====================================================================
    # 6 — Evaluate the model using hold-out and cross-validation techniques
    # ====================================================================
   
    y_pred = identificador.classificar(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ====================================================================
    # GERAÇÃO DA MATRIZ DE CONFUSÃO
    # ====================================================================

    # Criar a matriz numérica
    cm = confusion_matrix(y_test, y_pred)

    # Obter os nomes das vacas para os eixos do gráfico
    nomes_vacas = np.unique(y_test)

    # Configurar o visual do gráfico
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True,      # Escreve os números dentro dos quadrados
        fmt='d',         # Formato de número inteiro
        cmap='Blues',    # Cor azul (claro para poucos erros, escuro para muitos acertos)
        xticklabels=nomes_vacas, 
        yticklabels=nomes_vacas
    )

    plt.title('Matriz de Confusão - Identificação Individual de Vacas')
    plt.ylabel('Identidade Real (Gabarito)')
    plt.xlabel('Identidade Prevista (Sistema)')
    plt.savefig('matriz_confusao_final.png')


    # ============================================================
    # CROSS-VALIDATION (5-FOLD)
    # ============================================================
    print("\n[ITEM 6] Iniciando Cross-validation 5-fold...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X_embedding, cow_ids_final)):
        modelo_cv = IdentificadorVacas(label_encoder=gerenciadorDados.label_encoder)
        
        X_cv_train, X_cv_test = X_embedding[train_idx], X_embedding[test_idx]
        y_cv_train, y_cv_test = cow_ids_final[train_idx], cow_ids_final[test_idx]

        modelo_cv.treinar(X_cv_train, y_cv_train)
        preds_cv = modelo_cv.classificar(X_cv_test)

        score = accuracy_score(y_cv_test, preds_cv)
        cv_scores.append(score)
        print(f" > Dobra {i+1}: Acurácia = {score:.4f}")

    print(f"\nResultados Finais validação cruzada:")
    print(f" - Média: {np.mean(cv_scores):.4f}")
    print(f" - Desvio Padrão: {np.std(cv_scores):.4f}")

    # ============================================================
    # SALVAMENTO FINAL
    # ============================================================
    
    torch.save(treinador_triplet.modelo.state_dict(), os.getenv("MODELO_EMBEDDING"))
    joblib.dump(identificador, os.getenv("CLASSIFICADOR_XGBOOST"))
    joblib.dump(gerenciadorDados.label_encoder, os.getenv("LABEL_ENCODER"))

    print("Pipeline finalizada com sucesso!")