import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve, roc_curve
from tqdm import tqdm
import warnings
from undersampling import UnderSampling
from hydra import main
from joblib import dump
import logging
 
warnings.filterwarnings("ignore")
 
# Configurar logging para guardar las métricas en un archivo log
logging.basicConfig(filename='model_metrics.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', filemode='w')
 
@main(config_path="../../config/", config_name="config", version_base=None)
def CrossValidation(cfg):
    submuestras = UnderSampling(cfg)
    X_test = pd.read_csv(cfg.data.x_test_path)
    y_test = pd.read_csv(cfg.data.y_test_path)
    # Definir los hiperparámetros a ajustar
    param_grid = {
        'n_estimators': cfg.model.n_estimators,           # Número de árboles en el bosque
        'max_depth': cfg.model.max_depth,          # Profundidad máxima de los árboles
        'min_samples_split': cfg.model.min_samples_split,          # Número mínimo de muestras requeridas para dividir un nodo interno
        'min_samples_leaf': cfg.model.min_samples_leaf             # Número mínimo de muestras requeridas para estar en un nodo hoja
    }

    auc_lis = []
    pr_lis = []
    f1_lis = []
    n_estimators_lis = []
    max_depth_lis = []
    min_samples_split_lis = []
    min_samples_leaf_lis = []

    for sub in tqdm(submuestras[:20]):
    # Definir el clasificador RandomForest
        rf_classifier = RandomForestClassifier()
        model = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, scoring='average_precision', cv=2)
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
        n_estimators_lis.append(model.best_params_["n_estimators"])
        max_depth_lis.append(model.best_params_["max_depth"])
        min_samples_split_lis.append(model.best_params_["min_samples_split"])
        min_samples_leaf_lis.append(model.best_params_["min_samples_leaf"])
 

 
if __name__ == "__main__":
        CrossValidation()