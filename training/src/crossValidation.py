import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
from hydra import main
from joblib import dump
import logging

warnings.filterwarnings("ignore")

# Configurar logging para guardar las métricas en un archivo log
logging.basicConfig(filename='model_metrics.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', filemode='w')

@main(config_path="../../config/", config_name="config", version_base=None)
def train_RandomForest(cfg):
    # Leer datos desde las rutas configuradas
    X_train = pd.read_csv(cfg.data.x_train_path)
    y_train = pd.read_csv(cfg.data.y_train_path)

    # Configurar el clasificador RandomForest
    model = RandomForestClassifier(
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        min_samples_split=cfg.model.min_samples_split,
        min_samples_leaf=cfg.model.min_samples_leaf
    )

    # Cross-validation
    skf = StratifiedKFold(n_splits=cfg.model.cv_splits, shuffle=True, random_state=42)
    best_score = 0
    best_model = None

    # Realizar la validación cruzada
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        tqdm(model.fit(X_train_fold, y_train_fold))  # Entrenar el modelo

        # Predicciones en la validación
        y_pred = model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, y_pred)

        # Guardar métricas en el log
        logging.info(f"Fold {fold+1} - Accuracy: {score}")

        # Verificar si el modelo es el mejor hasta ahora
        if score > best_score:
            best_score = score
            best_model = model

    # Guardar el mejor modelo y sus hiperparámetros
    dump(best_model, cfg.model.model_path)

    # Guardar los hiperparámetros del mejor modelo
    with open('best_model_params.log', 'w') as f:
        f.write(f"Best Model Accuracy: {best_score}\n")
        f.write(f"Hiperparameters: {best_model.get_params()}")

if __name__ == "__main__":
    train_RandomForest()
