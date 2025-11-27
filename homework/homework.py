import gzip
import json
import os
import pickle
import zipfile
from glob import glob
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def leer_zip_a_dfs(directorio: str) -> list[pd.DataFrame]:
    dataframes = []
    for zip_path in sorted(glob(os.path.join(directorio, "*"))):
        with zipfile.ZipFile(zip_path, "r") as zf:
            for miembro in zf.namelist():
                with zf.open(miembro) as fh:
                    dataframes.append(pd.read_csv(fh, sep=",", index_col=0))
    return dataframes


def reiniciar_directorio(ruta: str) -> None:
    if os.path.exists(ruta):
        for f in glob(os.path.join(ruta, "*")):
            try:
                os.remove(f)
            except IsADirectoryError:
                pass
        try:
            os.rmdir(ruta)
        except OSError:
            pass
    os.makedirs(ruta, exist_ok=True)


def guardar_modelo_gz(ruta_salida: str, objeto) -> None:
    reiniciar_directorio(os.path.dirname(ruta_salida))
    with gzip.open(ruta_salida, "wb") as fh:
        pickle.dump(objeto, fh)


def depurar(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    
    # Renombrar columna default
    tmp = tmp.rename(columns={"default payment next month": "default"})
    
    # Eliminar columna ID si existe
    if 'ID' in tmp.columns:
        tmp = tmp.drop(columns=['ID'])
    
    # Filtrar registros con información no disponible
    tmp = tmp.loc[tmp["MARRIAGE"] != 0]
    tmp = tmp.loc[tmp["EDUCATION"] != 0]
    
    # Agrupar niveles superiores de educación en "others" (4)
    tmp["EDUCATION"] = tmp["EDUCATION"].apply(lambda v: 4 if v > 4 else v)
    
    return tmp.dropna()


def separar_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["default"])
    y = df["default"]
    return X, y


def ensamblar_busqueda() -> GridSearchCV:
    # Columnas categóricas para one-hot encoding
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    
    # OneHotEncoder con sparse_output=False para compatibilidad
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    # ColumnTransformer para transformar solo las categóricas
    ct = ColumnTransformer(
        transformers=[("cat", ohe, cat_cols)],
        remainder="passthrough",
    )
    
    # Random Forest con class_weight para datos desbalanceados
    clf = RandomForestClassifier(
        random_state=42,
        class_weight='balanced'  # Importante para default prediction
    )
    
    # Pipeline completo
    pipe = Pipeline(
        steps=[
            ("prep", ct),
            ("rf", clf),
        ]
    )
    
    # Grilla de hiperparámetros ampliada
    grid_params = {
        "rf__n_estimators": [200, 300, 500],
        "rf__max_depth": [10, 15, 20, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__max_features": ['sqrt', 'log2'],
    }
    
    # GridSearchCV con balanced_accuracy
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grid_params,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
        verbose=2,
    )
    return gs


def empaquetar_metricas(etiqueta: str, y_true, y_pred) -> dict:
    return {
        "type": "metrics",  # Campo requerido por el test
        "dataset": etiqueta,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def empaquetar_matriz_conf(etiqueta: str, y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": etiqueta,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }


def main() -> None:
    # Leer y limpiar datasets
    df_list = [depurar(d) for d in leer_zip_a_dfs("files/input")]
    
    # Identificar cuál es train y cuál es test por tamaño
    # El conjunto de entrenamiento suele ser más grande
    if len(df_list[0]) > len(df_list[1]):
        train_df, test_df = df_list[0], df_list[1]
    else:
        train_df, test_df = df_list[1], df_list[0]
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Separar X e y
    X_tr, y_tr = separar_xy(train_df)
    X_te, y_te = separar_xy(test_df)
    
    # Crear y entrenar el modelo
    print("Iniciando búsqueda de hiperparámetros...")
    buscador = ensamblar_busqueda()
    buscador.fit(X_tr, y_tr)
    
    print(f"Mejores parámetros: {buscador.best_params_}")
    print(f"Mejor score (CV): {buscador.best_score_:.4f}")
    
    # Guardar modelo comprimido
    guardar_modelo_gz(os.path.join("files", "models", "model.pkl.gz"), buscador)
    print("Modelo guardado exitosamente")
    
    # Predicciones
    yhat_train = buscador.predict(X_tr)
    yhat_test = buscador.predict(X_te)
    
    # Calcular métricas
    m_train = empaquetar_metricas("train", y_tr, yhat_train)
    m_test = empaquetar_metricas("test", y_te, yhat_test)
    
    # Calcular matrices de confusión
    cm_train = empaquetar_matriz_conf("train", y_tr, yhat_train)
    cm_test = empaquetar_matriz_conf("test", y_te, yhat_test)
    
    # Imprimir métricas
    print("\n--- Métricas de Entrenamiento ---")
    print(f"Precision: {m_train['precision']:.4f}")
    print(f"Balanced Accuracy: {m_train['balanced_accuracy']:.4f}")
    print(f"Recall: {m_train['recall']:.4f}")
    print(f"F1-Score: {m_train['f1_score']:.4f}")
    
    print("\n--- Métricas de Test ---")
    print(f"Precision: {m_test['precision']:.4f}")
    print(f"Balanced Accuracy: {m_test['balanced_accuracy']:.4f}")
    print(f"Recall: {m_test['recall']:.4f}")
    print(f"F1-Score: {m_test['f1_score']:.4f}")
    
    # Guardar métricas en archivo JSON
    Path("files/output").mkdir(parents=True, exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(m_train) + "\n")
        fh.write(json.dumps(m_test) + "\n")
        fh.write(json.dumps(cm_train) + "\n")
        fh.write(json.dumps(cm_test) + "\n")
    
    print("\nMétricas guardadas en files/output/metrics.json")


if __name__ == "__main__":
    main()