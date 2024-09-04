from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler

N_SAMPLES = 200
RATIO = 0.30
submuestras = []

for i in tqdm(range(N_SAMPLES)):
    random_undersampler = RandomUnderSampler(sampling_strategy=RATIO)
    X_resampled, y_resampled = random_undersampler.fit_resample(X_train, y_train)
    submuestras.append(X_resampled)