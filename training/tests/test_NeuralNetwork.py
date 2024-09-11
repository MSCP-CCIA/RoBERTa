import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
import numpy as np
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from undersampling import UnderSampling
import warnings
from hydra import main

warnings.filterwarnings("ignore")


@main(config_path="../../config/", config_name="config", version_base=None)
def train_NeuralNetwork(cfg):
    submuestras = UnderSampling(cfg)
    auc_lis = []
    pr_lis = []
    f1_lis = []
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    for sub in tqdm(submuestras):
        model = MLPClassifier()
        model.fit(sub[0], sub[1])

        # Predicción
        y_scores = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Calcular la curva AUC-ROC
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        # Calcular la curva AUC-PR
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recall, precision)

        # Realizar predicciones y calcular F1 Score
        f1 = f1_score(y_test, y_pred)

        auc_lis.append(roc_auc)
        pr_lis.append(pr_auc)
        f1_lis.append(f1)

    roc_auc = round(np.mean(auc_lis) * 100, 1)
    pr_auc = round(np.mean(pr_lis) * 100, 1)
    f1 = round(np.mean(f1_lis) * 100, 1)
    print(f"f1:{f1},pr_auc:{pr_auc},roc_auc:{roc_auc}")


if __name__ == "__main__":
    train_NeuralNetwork()
