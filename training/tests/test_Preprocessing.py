import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm
from scipy.stats import mode
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv(r"C:\Users\Asus\DataspellProjects\creditCardFraud\data\creditcard.csv")
X_train, X_test, y_train, y_test = train_test_split(df, df["Class"], test_size=0.2, stratify=df["Class"], random_state=42)
X_train.groupby("Class")["Time"].count()/X_train.groupby("Class")["Time"].count().sum()
X_test.groupby("Class")["Time"].count()/X_test.groupby("Class")["Time"].count().sum()
N_SAMPLES = 200
RATIO= 0.30
submuestras = []
for i in tqdm(range(N_SAMPLES)):
    random_undersampler = RandomUnderSampler(sampling_strategy=RATIO)
    X_resampled, y_resampled = random_undersampler.fit_resample(X_train, y_train)
    submuestras.append(X_resampled)