import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import warnings
from hydra import main
from joblib import dump

warnings.filterwarnings("ignore")


@main(config_path="../../config/", config_name="config", version_base=None)
def train_RandomForest(cfg):
    # Leer datos desde las rutas configuradas
    X_train = pd.read_csv(cfg.data.x_train_path)
    y_train = pd.read_csv(cfg.data.y_train_path)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=30, min_samples_split=2, min_samples_leaf=2
    )
    tqdm(model.fit(X_train, y_train))

    dump(model, cfg.model.model_path)


if __name__ == "__main__":
    train_RandomForest()
