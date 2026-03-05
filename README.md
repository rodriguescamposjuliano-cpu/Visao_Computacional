# 🐄 Biometria Bovina Híbrida: Identificação Individual via Pose e Deep Learning

Este projeto implementa um pipeline de visão computacional de ponta para a identificação individual de gado. Diferente de métodos tradicionais, este sistema utiliza uma abordagem híbrida que combina a estrutura física do animal (biometria geométrica) com padrões de pelagem (biometria visual).

## ✍️ Ciclo de preparação

1. **Annotate Dataset:** Marcação manual dos 8 pontos-chave anatômicos em imagens de vacas (Cernelha, Dorso, Quadril Central, Inserção da Cauda, Âncoras e Isquios).
2. **Treino e avaliação do modelo:** Foi utilizando a versão **YOLO26n-Pose** para o treinamento e validação.
   
## 🚀 Funcionalidades Principais

* **Detecção de Pose (YOLO26n-Pose):** Identificação automática de 8 pontos-chave anatômicos.
* **Extração Visual (DINOv2):** Uso de Vision Transformers (ViT) para capturar texturas e manchas.
* **Metric Learning (Triplet Loss):** Treinamento de uma rede neural para gerar assinaturas digitais de 128 bits com alta separabilidade.
* **Classificação Robusta (XGBoost):** Motor de decisão final treinado com validação cruzada (5-Fold).

---

## 🏗️ Arquitetura do Sistema

O fluxo de processamento é dividido em quatro camadas principais:

1.  **Camada de Percepção:** O `GerenciadorDados` processa o dataset e invoca o **Yolo26n-pose**.
2.  **Camada de Extração:** Fusão das características do `AnalistaBiometrico` (geometria) e `ExtratorVisual` (DINOv2).
3.  **Camada de Embedding:** A `RedeGeradoraDeAssinatura` refina os dados brutos (818D) em uma assinatura compacta (128D).
4.  **Camada de Decisão:** O `IdentificadorVacas` realiza a classificação final.

---

## 🛠️ Instalação e Uso
