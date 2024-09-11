import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm
from hydra import main


@main(config_path="../../config/", config_name="config", version_base=None)
def UnderSampling(cfg):
    # Leer datos desde las rutas configuradas
    X_train = pd.read_csv(cfg.data.x_train_path)
    y_train = pd.read_csv(cfg.data.y_train_path)

    N_SAMPLES = cfg.sampling.n_samples
    RATIO = cfg.sampling.ratio
    submuestras = []

    for _ in tqdm(range(N_SAMPLES)):
        random_undersampler = RandomUnderSampler(sampling_strategy=RATIO)
        X_resampled, y_resampled = random_undersampler.fit_resample(X_train, y_train)
        submuestras.append((X_resampled, y_resampled))

    return submuestras


if __name__ == "__main__":
    UnderSampling()
