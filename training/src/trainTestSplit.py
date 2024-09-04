from sklearn.model_selection import train_test_split
import pandas as pd


# Cargar el archivo CSV
file_path = 'data/creditcard2023.csv'
df = pd.read_csv(file_path)


# Separar las características (X) y la etiqueta (y)
X = df.drop(columns=["Class"])  # Asegúrate de reemplazar "Class" con el nombre correcto de la columna
y = df["Class"]


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Guardar los conjuntos de entrenamiento y prueba en archivos CSV separados
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)