from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class IdentificadorVacas:
    def __init__(self, label_encoder):
        self.label_encoder = label_encoder 
        self.modelo = None
        self.classes_no_treino = None

    def treinar(self, matriz_features, y_labels):
        if isinstance(y_labels[0], (int, np.integer)):
            y_labels = self.label_encoder.inverse_transform(y_labels)
        
        y_labels = np.array(y_labels).astype(str)
        self.classes_no_treino = np.unique(y_labels)
        mapping = {nome: i for i, nome in enumerate(self.classes_no_treino)}
        y_denso = np.array([mapping[n] for n in y_labels])

        # Split interno para o gráfico de Overfitting
        X_train_int, X_val_int, y_train_int, y_val_int = train_test_split(
            matriz_features, y_denso, test_size=0.2, stratify=y_denso, random_state=42
        )

        self.modelo = XGBClassifier(
            n_estimators=1000,
            max_depth=3,            # Reduzido para 3 para ser mais simples
            learning_rate=0.005,    # Aprendizado mais lento e cauteloso
            subsample=0.6,          # Usa apenas 60% dos dados por árvore (força generalização)
            colsample_bytree=0.6,
            reg_alpha=5.0,          # L1 muito mais forte para ignorar ruído
            reg_lambda=10.0,        # L2 muito mais forte
            random_state=42
        )

        # Treina monitorando o erro
        eval_set = [(X_train_int, y_train_int), (X_val_int, y_val_int)]
        self.modelo.fit(X_train_int, y_train_int, eval_set=eval_set, verbose=False)

        self.plot_learning_curve()

    def plot_learning_curve(self):
        results = self.modelo.evals_result()
        plt.figure(figsize=(10, 5))
        plt.plot(results['validation_0']['mlogloss'], label='Treino')
        plt.plot(results['validation_1']['mlogloss'], label='Validação Interna')
        plt.title('Gráfico para Avaliação de Overfitting (Log Loss)')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('curva_aprendizado_xgboost.png')
        print("📊 Gráfico salvo como 'curva_aprendizado_xgboost.png'")
        plt.show()

    def classificar(self, matriz_features):
        if self.modelo is None: return None
        y_pred_idx = self.modelo.predict(matriz_features)
        return np.array([self.classes_no_treino[i] for i in y_pred_idx.astype(int)])