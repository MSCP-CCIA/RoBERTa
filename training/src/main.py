import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
from joblib import load
from hydra import main


@main(config_path="../../config/", config_name="config", version_base=None)
def main(cfg):
    # Leer datos desde las rutas configuradas
    X_test = pd.read_csv(cfg.data.x_test_path)
    y_test = pd.read_csv(cfg.data.y_test_path)

    # Cargar el modelo
    model = load(cfg.model.model_path)

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

    print(f"f1:{f1}, pr_auc:{pr_auc}, roc_auc:{roc_auc}")


if __name__ == "__main__":
    main()
