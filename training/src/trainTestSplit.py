import pandas as pd
from sklearn.model_selection import train_test_split
from hydra import main


@main(config_path="../../config/", config_name="config", version_base=None)
def main(cfg):
    # Cargar el archivo CSV desde la ruta configurada
    df = pd.read_csv(cfg.data.raw_data_path)

    # Separar las características (X) y la etiqueta (y)
    X = df.drop(columns=[cfg.split.target_column])  # Asegúrate de reemplazar "Class" con el nombre correcto de la columna
    y = df[cfg.split.target_column]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    stratify = y if cfg.split.stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.split.test_size,
        stratify=stratify,
        random_state=cfg.split.random_state
    )

    # Guardar los conjuntos de entrenamiento y prueba en archivos CSV separados
    X_train.to_csv(cfg.data.x_train_path, index=False)
    X_test.to_csv(cfg.data.x_test_path, index=False)
    y_train.to_csv(cfg.data.y_train_path, index=False)
    y_test.to_csv(cfg.data.y_test_path, index=False)


if __name__ == "__main__":
    main()
