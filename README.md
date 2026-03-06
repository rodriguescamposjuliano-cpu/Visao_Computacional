# 🐄 Biometria Bovina Híbrida: Identificação Individual via Pose e Deep Learning:

Este projeto implementa um pipeline de visão computacional de ponta para a identificação individual de gado. Diferente de métodos tradicionais, este sistema utiliza uma abordagem híbrida que combina a estrutura física do animal (biometria geométrica) com padrões de pelagem (biometria visual).

## ✍️ Ciclo de preparação:

1. **Annotate Dataset:** Marcação manual dos 8 pontos-chave anatômicos em imagens de vacas (Cernelha, Dorso, Quadril Central, Inserção da Cauda, Âncoras e Isquios).
2. **Treino e avaliação do modelo:** Foi utilizando a versão **YOLO26n-Pose** para o treinamento e validação.
   
## 🚀 Funcionalidades Principais:

* **Detecção de Pose (YOLO26n-Pose):** Identificação automática de 8 pontos-chave anatômicos.
* **Extração Visual (DINOv2):** Uso de Vision Transformers (ViT) para capturar texturas e manchas.
* **Metric Learning (Triplet Loss):** Treinamento de uma rede neural para gerar assinaturas digitais de 128 bits com alta separabilidade.
* **Classificação Robusta (XGBoost):** Motor de decisão final treinado com validação cruzada (5-Fold).

---

## 🏗️ Arquitetura do Sistema:

O fluxo de processamento é dividido em quatro camadas principais:

1.  **Camada de Percepção:** O `GerenciadorDados` processa o dataset e invoca o **Yolo26n-pose**.
2.  **Camada de Extração:** Fusão das características do `AnalistaBiometrico` (geometria) e `ExtratorVisual` (DINOv2).
3.  **Camada de Embedding:** A `RedeGeradoraDeAssinatura` refina os dados brutos (818D) em uma assinatura compacta (128D).
4.  **Camada de Decisão:** O `IdentificadorVacas` realiza a classificação final.

---

## 🛠️ Instalação e Uso:

#### 1.  **Requisitos do Sistema:**

O projeto foi desenvolvido em Python 3.8+ e utiliza bibliotecas de ponta para processamento de imagem, Deep Learning e Machine Learning. Para instalar todas as dependências necessárias, execute:

```Bash
pip install -r requirements.txt
```

As principais bibliotecas utilizadas são:
* **Visão Computacional:** `ultralytics` (YOLOv26), `opencv-python` e `Pillow`.
* **Deep Learning:** `torch`, `torchvision` e `timm` (para o DINOv2).
* **Machine Learning:** `scikit-learn` e `xgboost`.

#### 2. **Treinamento do Modelo de Pose (YOLO26n-Pose)**
O arquivo `main.py` gerencia o ciclo de treinamento e validação dos 8 pontos-chave anatômicos:

* **Configuração:** O treino utiliza o otimizador `AdamW`, 100 épocas e aumentos de dados (augmentations) específicos como `flipud` e `scale` para garantir robustez.

* **Augmentation:** Foram aplicados ajustes finos de aumento de dados, incluindo `flipud` (1.0), `scale` (0.5) e `degrees` (5.0) para aumentar a robustez do modelo.

* **Execução:**


```Bash
python main.py
```

* **Saída:** Os pesos e métricas de precisão (mAP) são salvos automaticamente no diretório `trabalho_vaca/resultados`.

#### 3. **Execução do Pipeline de Identificação**
Para realizar a identificação individual a partir de uma nova imagem, utilize a classe orquestradora:


```Python
from pipeline_completa.pipeline_identificacao_vacas import PipelineIdentificacaoVacas

# Inicializa o pipeline com o modelo de pose treinado
pipeline = PipelineIdentificacaoVacas(caminho_modelo_pose='yolo26n-pose.pt')

# Extrai o vetor de características (features)
vetor_features = pipeline.extrair_features_imagem('caminho/para/vaca.jpg')
```

## 📂 Estrutura Principal do Repositório
* `main.py`: Script principal para treino e validação do modelo YOLO26.

* `pipeline_completa/`: Contém os módulos de extração biométrica e visual (DINOv2).

* `vaca_data.yaml`: Definição dos caminhos do dataset e classes para o YOLO.

* `requirements.txt`: Listagem de bibliotecas como ultralytics, timm e xgboost.

* `yolo26n-pose.pt`: Pesos pré-treinados do modelo de pose.



## 📝 Notas de Implementação:

* **Hardware:** O script de treino está configurado para utilizar `device='mps'` (Metal Performance Shaders para Mac). Para outros sistemas, altere para `'0'` (GPU NVIDIA) ou `'cpu'`.

* **Logs:** O sistema utiliza a biblioteca `logging` para monitorar falhas na detecção de keypoints durante o processo de extração.