# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import gzip
import json
import os
import pickle
import zipfile
from glob import glob
from pathlib import Path

import pandas as pd  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore


# ---------------------------------------------------------
# Lectura / Escritura
# ---------------------------------------------------------

def cargar_zip_a_dfs(ruta_dir: str) -> list[pd.DataFrame]:
    """Carga todos los archivos dentro de zips en un directorio y devuelve una lista de DataFrames."""
    dataframes = []

    for ruta_zip in glob(os.path.join(ruta_dir, "*")):
        with zipfile.ZipFile(ruta_zip, "r") as zf:
            for nombre in zf.namelist():
                with zf.open(nombre) as archivo:
                    dataframes.append(pd.read_csv(archivo, sep=",", index_col=0))

    return dataframes


def limpiar_directorio(ruta: str) -> None:
    """Elimina todo dentro de un directorio y lo vuelve a crear."""
    if os.path.exists(ruta):
        for archivo in glob(os.path.join(ruta, "*")):
            try:
                os.remove(archivo)
            except IsADirectoryError:
                continue
        try:
            os.rmdir(ruta)
        except OSError:
            pass

    os.makedirs(ruta, exist_ok=True)


def guardar_objeto_gzip(ruta_salida: str, objeto) -> None:
    """Guarda cualquier objeto serializable en un archivo .gz."""
    limpiar_directorio(os.path.dirname(ruta_salida))
    with gzip.open(ruta_salida, "wb") as fh:
        pickle.dump(objeto, fh)


# ---------------------------------------------------------
# Procesamiento de datos
# ---------------------------------------------------------

def depurar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza y normalización del dataset."""
    df = df.copy()
    df = df.rename(columns={"default payment next month": "default"})

    df = df[(df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)]

    # Normalización de EDUCATION
    df["EDUCATION"] = df["EDUCATION"].clip(upper=4)

    return df.dropna()


def dividir_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["default"])
    y = df["default"]
    return X, y


# ---------------------------------------------------------
# Modelo y Evaluación
# ---------------------------------------------------------

def construir_gridsearch() -> GridSearchCV:
    """Construye un pipeline + GridSearchCV."""
    columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]

    preprocesador = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas)],
        remainder="passthrough",
    )

    clasificador = RandomForestClassifier(random_state=42)

    pipeline = Pipeline([
        ("preprocesamiento", preprocesador),
        ("modelo", clasificador),
    ])

    hiperparametros = {
        "modelo__n_estimators": [100, 200, 500],
        "modelo__max_depth": [None, 5, 10],
        "modelo__min_samples_split": [2, 5],
        "modelo__min_samples_leaf": [1, 2],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=hiperparametros,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=2,
        refit=True,
    )


def generar_metricas(nombre_ds: str, y_real, y_pred) -> dict:
    return {
        "dataset": nombre_ds,
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
        "recall": recall_score(y_real, y_pred, zero_division=0),
        "f1_score": f1_score(y_real, y_pred, zero_division=0),
    }


def generar_matriz_confusion(nombre_ds: str, y_real, y_pred) -> dict:
    cm = confusion_matrix(y_real, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": nombre_ds,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    # Cargar datos
    dataframes = [depurar_dataframe(df) for df in cargar_zip_a_dfs("files/input")]

    df_test, df_train = dataframes

    X_train, y_train = dividir_features_target(df_train)
    X_test, y_test = dividir_features_target(df_test)

    # Entrenar modelo
    grid = construir_gridsearch()
    grid.fit(X_train, y_train)

    # Guardar modelo
    guardar_objeto_gzip("files/models/model.pkl.gz", grid)

    # Predicciones
    pred_test = grid.predict(X_test)
    pred_train = grid.predict(X_train)

    # Métricas
    metricas_train = generar_metricas("train", y_train, pred_train)
    metricas_test = generar_metricas("test", y_test, pred_test)

    cm_train = generar_matriz_confusion("train", y_train, pred_train)
    cm_test = generar_matriz_confusion("test", y_test, pred_test)

    # Guardar
    salida = Path("files/output")
    salida.mkdir(parents=True, exist_ok=True)

    with open(salida / "metrics.json", "w", encoding="utf-8") as fh:
        for registro in (metricas_train, metricas_test, cm_train, cm_test):
            fh.write(json.dumps(registro) + "\n")


if __name__ == "__main__":
    main()
