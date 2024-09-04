from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

auc_lis = []
pr_lis = []
f1_lis = []
resultados = {}

for sub in tqdm(submuestras):
    knn = KNeighborsClassifier()
    knn.fit(sub[x_cols], sub[y_cols])

    # Predicción
    y_scores = knn.predict_proba(X_test[x_cols])[:, 1]
    y_pred = knn.predict(X_test[x_cols])

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


resultados["KNN"]=[roc_auc, pr_auc, f1]