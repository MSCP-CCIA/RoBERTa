from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, df["Class"], test_size=0.2, stratify=df["Class"], random_state=42)

N_SAMPLES = 200
RATIO= 0.30
submuestras = []

for i in tqdm(range(N_SAMPLES)):
    random_undersampler = RandomUnderSampler(sampling_strategy=RATIO)
    X_resampled, y_resampled = random_undersampler.fit_resample(X_train, y_train)
    submuestras.append(X_resampled)

resultados = {}
auc_lis = []
pr_lis = []
f1_lis = []

X_train, X_test, y_train, y_test = train_test_split(df, df["Class"], test_size=0.2, stratify=df["Class"], random_state=42)
X_train.groupby("Class")["Time"].count()/X_train.groupby("Class")["Time"].count().sum()

for sub in tqdm(submuestras):
    model = RandomForestClassifier()
    model.fit(sub[x_cols], sub[y_cols])

    # Predicción
    y_scores = model.predict_proba(X_test[x_cols])[:, 1]
    y_pred = model.predict(X_test[x_cols])

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

roc_auc = round(np.mean(auc_lis)*100,1)
pr_auc = round(np.mean(pr_lis)*100,1)
f1 = round(np.mean(f1_lis)*100,1)


resultados["BosqueAleatorio"]=[roc_auc, pr_auc, f1]
#%% md
###**📈Regresión logistica📉**