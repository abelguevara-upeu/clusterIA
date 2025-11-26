"""
Módulo de modelos de clustering y PCA para Edu Insight
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def aplicar_pca(data_scaled, n_components=None):
    """
    Aplica PCA a los datos escalados

    Args:
        data_scaled: DataFrame con datos escalados
        n_components: Número de componentes (None para calcular óptimo)

    Returns:
        dict con resultados del PCA
    """
    # PCA completo para análisis
    pca = PCA()
    pca.fit(data_scaled)

    autovalores = pca.explained_variance_

    # Si no se especifica, usar criterio de Kaiser (autovalor > 1)
    if n_components is None:
        n_components = sum(autovalores > 1)

    # Tabla de varianza explicada
    var_exp = pd.DataFrame({
        "Componente": [f"C{i+1}" for i in range(len(autovalores))],
        "Autovalor": autovalores,
        "Varianza %": pca.explained_variance_ratio_ * 100,
        "Acumulado %": np.cumsum(pca.explained_variance_ratio_) * 100
    })

    # PCA final con n_components
    pca_final = PCA(n_components=n_components)
    data_pca = pca_final.fit_transform(data_scaled)

    df_pca = pd.DataFrame(
        data_pca,
        columns=[f"C{i+1}" for i in range(n_components)]
    )

    # Matriz de componentes (loadings)
    cargas = pd.DataFrame(
        pca_final.components_.T,
        columns=[f"C{i+1}" for i in range(n_components)],
        index=data_scaled.columns
    )

    return {
        'pca': pca_final,
        'data_pca': df_pca,
        'var_exp': var_exp,
        'cargas': cargas,
        'n_components': n_components,
        'autovalores': autovalores
    }

def encontrar_k_optimo(X, max_k=8):
    """Encuentra el número óptimo de clusters"""
    resultados = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        resultados.append({
            "k": k,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "inercia": kmeans.inertia_
        })

    scores_df = pd.DataFrame(resultados)
    optimal_k = int(scores_df.loc[scores_df["silhouette"].idxmax(), "k"])

    return optimal_k, scores_df

def aplicar_kmeans(data_scaled, n_clusters):
    """
    Aplica K-Means clustering

    Args:
        data_scaled: DataFrame con datos escalados
        n_clusters: Número de clusters

    Returns:
        dict con resultados del clustering
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)

    # Calcular distancias a centroides
    distancias = np.linalg.norm(
        data_scaled.values[:, None] - kmeans.cluster_centers_,
        axis=2
    )
    distancias_min = np.min(distancias, axis=1)

    # Métricas
    metricas = {
        'silhouette': silhouette_score(data_scaled, labels),
        'davies_bouldin': davies_bouldin_score(data_scaled, labels),
        'calinski_harabasz': calinski_harabasz_score(data_scaled, labels),
        'inercia': kmeans.inertia_
    }

    # Centroides como DataFrame
    centroides_df = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=data_scaled.columns,
        index=[f'Cluster {i+1}' for i in range(n_clusters)]
    )

    return {
        'kmeans': kmeans,
        'labels': labels,
        'centroides': centroides_df,
        'distancias': distancias_min,
        'metricas': metricas
    }

def calcular_perfiles_clusters(df_original, labels, centroides_df):
    """Calcula perfiles descriptivos de cada cluster"""
    from sklearn.preprocessing import LabelEncoder

    df_temp = df_original.copy()
    df_temp['Cluster'] = labels + 1

    perfiles = []

    for cluster in range(1, len(centroides_df) + 1):
        df_cluster = df_temp[df_temp['Cluster'] == cluster]
        n_estudiantes = len(df_cluster)

        # Características principales del cluster
        perfil = {
            'cluster': cluster,
            'n_estudiantes': n_estudiantes,
            'porcentaje': (n_estudiantes / len(df_temp)) * 100
        }

        # Agregar estadísticas de variables numéricas
        for col in df_original.columns:
            if df_original[col].dtype in ['int64', 'float64']:
                perfil[f'{col}_mean'] = df_cluster[col].mean()
            elif df_original[col].dtype == 'object':
                # Para categóricas, obtener la moda
                perfil[f'{col}_mode'] = df_cluster[col].mode()[0] if len(df_cluster[col].mode()) > 0 else None

        perfiles.append(perfil)

    return pd.DataFrame(perfiles)
