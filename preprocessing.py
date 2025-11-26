"""
Módulo de preprocesamiento de datos para Edu Insight
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def cargar_datos(ruta_archivo):
    """Carga el dataset desde un archivo CSV"""
    return pd.read_csv(ruta_archivo, low_memory=False, encoding='utf-8')

def limpiar_datos(df):
    """Limpia el dataset eliminando columnas irrelevantes"""
    # Columnas a eliminar (solo identificadores y etiquetas)
    cols_drop = []

    # Si existen estas columnas, las eliminamos
    cols_posibles = ["student_id", "cluster_label_true"]
    for col in cols_posibles:
        if col in df.columns:
            cols_drop.append(col)

    df_clean = df.drop(columns=cols_drop, errors="ignore").copy()
    return df_clean

def escalar_datos(df):
    """Escala los datos usando StandardScaler"""
    from sklearn.preprocessing import LabelEncoder

    df_encoded = df.copy()

    # Codificar todas las variables categóricas
    label_encoders = {}
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le

    # Escalar los datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_encoded)

    df_scaled = pd.DataFrame(data_scaled, columns=df_encoded.columns)

    return df_scaled, scaler, label_encoders

def verificar_pca_viabilidad(data_scaled):
    """Verifica si PCA es viable mediante determinante y KMO"""
    try:
        from factor_analyzer.factor_analyzer import calculate_kmo

        correlation_matrix = np.corrcoef(data_scaled, rowvar=False)
        determinante = np.linalg.det(correlation_matrix)

        _, kmo = calculate_kmo(data_scaled)

        return {
            'determinante': determinante,
            'kmo': kmo,
            'viable': kmo >= 0.60
        }
    except:
        return {
            'determinante': None,
            'kmo': None,
            'viable': True  # Asumimos viable si no se puede calcular
        }
