"""
EDU INSIGHT - Sistema de Análisis de Perfiles Estudiantiles
Aplicación Streamlit con Clustering y PCA
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Importar módulos propios
from preprocessing import cargar_datos, limpiar_datos, escalar_datos, verificar_pca_viabilidad
from models import aplicar_pca, encontrar_k_optimo, aplicar_kmeans, calcular_perfiles_clusters

# Configuración de página
st.set_page_config(
    page_title="EDU INSIGHT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para diseño elegante
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #667eea;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Funciones auxiliares de visualización
def crear_grafico_scree(autovalores):
    """Crea gráfico de sedimentación (Scree Plot)"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1, len(autovalores) + 1)),
        y=autovalores,
        mode='lines+markers',
        name='Autovalores',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10, color='#764ba2')
    ))

    fig.add_hline(y=1, line_dash="dash", line_color="red",
                  annotation_text="Criterio de Kaiser (λ=1)")

    fig.update_layout(
        title="Scree Plot - Selección de Componentes",
        xaxis_title="Componente Principal",
        yaxis_title="Autovalor",
        template="plotly_white",
        height=400
    )

    return fig

def crear_grafico_varianza(var_exp_df):
    """Crea gráfico de varianza explicada"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=var_exp_df['Componente'][:10],
            y=var_exp_df['Varianza %'][:10],
            name='Varianza Individual',
            marker_color='#667eea'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=var_exp_df['Componente'][:10],
            y=var_exp_df['Acumulado %'][:10],
            name='Varianza Acumulada',
            line=dict(color='#f093fb', width=3),
            mode='lines+markers'
        ),
        secondary_y=True
    )

    fig.update_xaxes(title_text="Componente")
    fig.update_yaxes(title_text="Varianza Individual (%)", secondary_y=False)
    fig.update_yaxes(title_text="Varianza Acumulada (%)", secondary_y=True)

    fig.update_layout(
        title="Varianza Explicada por Componente",
        template="plotly_white",
        height=400
    )

    return fig

def crear_grafico_pca_3d(df_pca, labels):
    """Crea visualización 3D de componentes principales"""
    df_plot = df_pca.copy()
    df_plot['Cluster'] = [f'Cluster {i+1}' for i in labels]

    fig = px.scatter_3d(
        df_plot,
        x='C1', y='C2', z='C3',
        color='Cluster',
        title='Visualización 3D - Componentes Principales',
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=600
    )

    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(template="plotly_white")

    return fig

def crear_grafico_pca_2d(df_pca, labels, comp_x='C1', comp_y='C2'):
    """Crea visualización 2D de componentes principales"""
    df_plot = df_pca.copy()
    df_plot['Cluster'] = [f'Cluster {i+1}' for i in labels]

    fig = px.scatter(
        df_plot,
        x=comp_x, y=comp_y,
        color='Cluster',
        title=f'{comp_x} vs {comp_y}',
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=400
    )

    fig.update_traces(marker=dict(size=8, opacity=0.6))
    fig.update_layout(template="plotly_white")

    return fig

def crear_heatmap_cargas(cargas_df, n_vars=10):
    """Crea heatmap de cargas factoriales"""
    # Seleccionar las variables más importantes
    cargas_abs = cargas_df.abs().sum(axis=1).sort_values(ascending=False)
    top_vars = cargas_abs.head(n_vars).index

    fig = px.imshow(
        cargas_df.loc[top_vars].T,
        labels=dict(x="Variable", y="Componente", color="Carga"),
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title=f'Matriz de Cargas - Top {n_vars} Variables',
        height=500
    )

    fig.update_layout(template="plotly_white")

    return fig

def crear_grafico_metricas_k(scores_df):
    """Crea gráfico de métricas para selección de k"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Silhouette Score', 'Davies-Bouldin Index',
                       'Calinski-Harabasz Score', 'Inercia')
    )

    # Silhouette
    fig.add_trace(
        go.Scatter(x=scores_df['k'], y=scores_df['silhouette'],
                  mode='lines+markers', name='Silhouette',
                  line=dict(color='#667eea', width=3)),
        row=1, col=1
    )

    # Davies-Bouldin
    fig.add_trace(
        go.Scatter(x=scores_df['k'], y=scores_df['davies_bouldin'],
                  mode='lines+markers', name='Davies-Bouldin',
                  line=dict(color='#f093fb', width=3)),
        row=1, col=2
    )

    # Calinski-Harabasz
    fig.add_trace(
        go.Scatter(x=scores_df['k'], y=scores_df['calinski_harabasz'],
                  mode='lines+markers', name='Calinski-Harabasz',
                  line=dict(color='#4ecdc4', width=3)),
        row=2, col=1
    )

    # Inercia
    fig.add_trace(
        go.Scatter(x=scores_df['k'], y=scores_df['inercia'],
                  mode='lines+markers', name='Inercia',
                  line=dict(color='#ff6b6b', width=3)),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Número de Clusters (k)")
    fig.update_layout(height=600, showlegend=False, template="plotly_white",
                     title_text="Métricas de Evaluación de Clustering")

    return fig

def crear_grafico_distribucion_clusters(labels):
    """Crea gráfico de distribución de estudiantes por cluster"""
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    cluster_labels = [f'Cluster {i+1}' for i in cluster_counts.index]

    fig = go.Figure(data=[
        go.Bar(
            x=cluster_labels,
            y=cluster_counts.values,
            marker=dict(
                color=cluster_counts.values,
                colorscale='Viridis',
                showscale=True
            ),
            text=cluster_counts.values,
            textposition='auto'
        )
    ])

    fig.update_layout(
        title="Distribución de Estudiantes por Cluster",
        xaxis_title="Cluster",
        yaxis_title="Número de Estudiantes",
        template="plotly_white",
        height=400
    )

    return fig

def crear_radar_chart_centroides(centroides_df, top_n=8):
    """Crea gráfico radar de características de clusters"""
    # Seleccionar top N variables con mayor varianza
    varianzas = centroides_df.var().sort_values(ascending=False)
    top_vars = varianzas.head(top_n).index.tolist()

    fig = go.Figure()

    colors = ['#667eea', '#f093fb', '#4ecdc4', '#ff6b6b', '#45b7d1', '#96ceb4']

    for idx, (cluster, row) in enumerate(centroides_df.iterrows()):
        fig.add_trace(go.Scatterpolar(
            r=row[top_vars].values,
            theta=top_vars,
            fill='toself',
            name=cluster,
            line=dict(color=colors[idx % len(colors)])
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="Perfil de Clusters - Variables Principales",
        height=500
    )

    return fig

# ==================== APLICACIÓN PRINCIPAL ====================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 EDU INSIGHT</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Sistema de Análisis de Perfiles Estudiantiles con Clustering y PCA</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/graduation-cap.png", width=80)
        st.title("⚙️ Configuración")

        # Carga de archivo
        uploaded_file = st.file_uploader(
            "Cargar Dataset CSV",
            type=['csv'],
            help="Sube tu archivo de datos estudiantiles"
        )

        if uploaded_file is None:
            # Opción de usar datos por defecto
            usar_default = st.checkbox("Usar datos sintéticos por defecto", value=True)
            if usar_default:
                ruta_default = "data/sintetic_data.csv"
            else:
                st.info("👆 Por favor, carga un archivo CSV para comenzar")
                st.stop()
        else:
            ruta_default = None

        st.markdown("---")

        # Parámetros
        st.subheader("Parámetros de Análisis")

        max_clusters = st.slider(
            "Máximo de clusters a evaluar",
            min_value=3,
            max_value=10,
            value=8,
            help="Rango para buscar el número óptimo de clusters"
        )

        n_components_manual = st.checkbox("Especificar componentes PCA manualmente")
        if n_components_manual:
            n_components = st.slider("Número de componentes PCA", 2, 10, 3)
        else:
            n_components = None

        analizar = st.button("🚀 Iniciar Análisis", type="primary", use_container_width=True)

    # Área principal
    if not analizar:
        # Vista de bienvenida
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("📊 **Análisis PCA**\n\nReducción de dimensionalidad y visualización de componentes principales")

        with col2:
            st.success("🎯 **Clustering K-Means**\n\nSegmentación de estudiantes en grupos con características similares")

        with col3:
            st.warning("📈 **Visualizaciones**\n\nGráficos interactivos 2D y 3D para explorar los datos")

        st.markdown("---")
        st.markdown("### 🔍 ¿Qué hace EDU INSIGHT?")
        st.markdown("""
        EDU INSIGHT utiliza técnicas avanzadas de Machine Learning para:

        - **Identificar patrones** en el comportamiento estudiantil
        - **Segmentar estudiantes** en grupos homogéneos
        - **Visualizar relaciones** entre múltiples variables académicas
        - **Generar insights** para la toma de decisiones educativas

        👉 Configura los parámetros en el panel lateral y presiona **"Iniciar Análisis"** para comenzar.
        """)

        st.stop()

    # Procesamiento
    with st.spinner("Cargando y procesando datos..."):
        # Cargar datos
        if ruta_default:
            df = cargar_datos(ruta_default)
        else:
            df = pd.read_csv(uploaded_file)

        df_original = df.copy()

        # Limpiar y escalar
        df_clean = limpiar_datos(df)
        df_scaled, scaler, label_encoders = escalar_datos(df_clean)

        # Verificar viabilidad de PCA
        pca_check = verificar_pca_viabilidad(df_scaled.values)

    # Mostrar información del dataset
    st.markdown('<h2 class="sub-header">📋 Información del Dataset</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Registros", f"{len(df):,}")

    with col2:
        st.metric("Variables", f"{len(df_clean.columns)}")

    with col3:
        kmo_value = pca_check['kmo'] if pca_check['kmo'] else 0
        st.metric("KMO Score", f"{kmo_value:.3f}")

    with col4:
        viabilidad = "✅ Viable" if pca_check['viable'] else "⚠️ Limitado"
        st.metric("PCA", viabilidad)

    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis PCA", "🎯 Clustering", "🔍 Exploración", "📥 Exportar"])

    # ==================== TAB 1: PCA ====================
    with tab1:
        st.markdown('<h2 class="sub-header">Análisis de Componentes Principales</h2>', unsafe_allow_html=True)

        with st.spinner("Aplicando PCA..."):
            resultados_pca = aplicar_pca(df_scaled, n_components=n_components)

        # Información de componentes
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Componentes Retenidos", resultados_pca['n_components'])
            var_acum = resultados_pca['var_exp'].iloc[resultados_pca['n_components']-1]['Acumulado %']
            st.metric("Varianza Explicada", f"{var_acum:.2f}%")

        with col2:
            st.plotly_chart(crear_grafico_scree(resultados_pca['autovalores']),
                          use_container_width=True)

        # Tabla de varianza
        st.subheader("Tabla de Varianza Explicada")
        st.dataframe(
            resultados_pca['var_exp'].head(10).style.format({
                'Autovalor': '{:.3f}',
                'Varianza %': '{:.2f}',
                'Acumulado %': '{:.2f}'
            }),
            use_container_width=True
        )

        # Gráfico de varianza
        st.plotly_chart(crear_grafico_varianza(resultados_pca['var_exp']),
                       use_container_width=True)

        # Heatmap de cargas
        st.subheader("Matriz de Cargas Factoriales")
        n_vars_mostrar = st.slider("Número de variables a mostrar", 5, 20, 10)
        st.plotly_chart(crear_heatmap_cargas(resultados_pca['cargas'], n_vars_mostrar),
                       use_container_width=True)

        # Top variables por componente
        st.subheader("Variables Más Influyentes por Componente")

        cols = st.columns(min(3, resultados_pca['n_components']))

        for i, col in enumerate(cols):
            if i < resultados_pca['n_components']:
                comp_name = f"C{i+1}"
                with col:
                    st.markdown(f"**{comp_name}**")
                    top_vars = resultados_pca['cargas'][comp_name].abs().sort_values(ascending=False).head(5)

                    for var, valor in top_vars.items():
                        signo = "+" if resultados_pca['cargas'].loc[var, comp_name] > 0 else "-"
                        st.markdown(f"{signo} {var}: `{abs(valor):.3f}`")

    # ==================== TAB 2: CLUSTERING ====================
    with tab2:
        st.markdown('<h2 class="sub-header">Análisis de Clustering</h2>', unsafe_allow_html=True)

        # Encontrar k óptimo
        with st.spinner("Calculando número óptimo de clusters..."):
            optimal_k, scores_df = encontrar_k_optimo(df_scaled.values, max_k=max_clusters)

        st.success(f"✨ Número óptimo de clusters detectado: **{optimal_k}**")

        # Permitir override manual
        col1, col2 = st.columns([1, 3])
        with col1:
            usar_k_custom = st.checkbox("Usar número personalizado")
            if usar_k_custom:
                k_final = st.number_input("Clusters", min_value=2, max_value=max_clusters, value=optimal_k)
            else:
                k_final = optimal_k

        with col2:
            st.plotly_chart(crear_grafico_metricas_k(scores_df), use_container_width=True)

        # Aplicar clustering
        with st.spinner(f"Aplicando K-Means con {k_final} clusters..."):
            resultados_cluster = aplicar_kmeans(df_scaled, k_final)

        # Métricas de clustering
        st.subheader("Métricas de Calidad del Clustering")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Silhouette Score",
                     f"{resultados_cluster['metricas']['silhouette']:.3f}",
                     help="Mayor es mejor (rango: -1 a 1)")

        with col2:
            st.metric("Davies-Bouldin Index",
                     f"{resultados_cluster['metricas']['davies_bouldin']:.3f}",
                     help="Menor es mejor")

        with col3:
            st.metric("Calinski-Harabasz",
                     f"{resultados_cluster['metricas']['calinski_harabasz']:.0f}",
                     help="Mayor es mejor")

        # Distribución de clusters
        st.plotly_chart(crear_grafico_distribucion_clusters(resultados_cluster['labels']),
                       use_container_width=True)

        # Visualizaciones PCA + Clusters
        st.subheader("Visualización de Clusters en Espacio PCA")

        if resultados_pca['n_components'] >= 3:
            st.plotly_chart(
                crear_grafico_pca_3d(resultados_pca['data_pca'], resultados_cluster['labels']),
                use_container_width=True
            )

        # Gráficos 2D
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                crear_grafico_pca_2d(resultados_pca['data_pca'], resultados_cluster['labels'], 'C1', 'C2'),
                use_container_width=True
            )

        with col2:
            if resultados_pca['n_components'] >= 3:
                st.plotly_chart(
                    crear_grafico_pca_2d(resultados_pca['data_pca'], resultados_cluster['labels'], 'C2', 'C3'),
                    use_container_width=True
                )

        # Radar chart de centroides
        st.subheader("Perfil de Clusters")
        st.plotly_chart(crear_radar_chart_centroides(resultados_cluster['centroides']),
                       use_container_width=True)

        # Tabla de centroides
        with st.expander("Ver Tabla Completa de Centroides"):
            st.dataframe(resultados_cluster['centroides'].T.style.format("{:.3f}"),
                        use_container_width=True)

    # ==================== TAB 3: EXPLORACIÓN ====================
    with tab3:
        st.markdown('<h2 class="sub-header">Exploración Interactiva</h2>', unsafe_allow_html=True)

        # Crear dataset completo
        df_completo = df_original.copy()
        df_completo['Cluster'] = resultados_cluster['labels'] + 1
        df_completo['Distancia_Centroide'] = resultados_cluster['distancias']

        # Agregar componentes PCA
        for i in range(resultados_pca['n_components']):
            df_completo[f'C{i+1}'] = resultados_pca['data_pca'].iloc[:, i]

        # Filtros
        st.subheader("Filtros")

        col1, col2 = st.columns(2)

        with col1:
            clusters_seleccionados = st.multiselect(
                "Seleccionar Clusters",
                options=sorted(df_completo['Cluster'].unique()),
                default=sorted(df_completo['Cluster'].unique())
            )

        with col2:
            # Seleccionar variables numéricas
            vars_numericas = df_original.select_dtypes(include=[np.number]).columns.tolist()
            if vars_numericas:
                var_filtro = st.selectbox("Variable para filtrar", ['Ninguna'] + vars_numericas)

                if var_filtro != 'Ninguna':
                    min_val = float(df_original[var_filtro].min())
                    max_val = float(df_original[var_filtro].max())
                    rango = st.slider(
                        f"Rango de {var_filtro}",
                        min_val, max_val, (min_val, max_val)
                    )

        # Aplicar filtros
        df_filtrado = df_completo[df_completo['Cluster'].isin(clusters_seleccionados)]

        if var_filtro != 'Ninguna':
            df_filtrado = df_filtrado[
                (df_filtrado[var_filtro] >= rango[0]) &
                (df_filtrado[var_filtro] <= rango[1])
            ]

        st.info(f"📊 Mostrando {len(df_filtrado)} de {len(df_completo)} registros")

        # Tabla interactiva
        st.subheader("Datos Filtrados")
        st.dataframe(df_filtrado, use_container_width=True, height=400)

        # Estadísticas por cluster
        st.subheader("Estadísticas por Cluster")

        perfiles = calcular_perfiles_clusters(df_original, resultados_cluster['labels'],
                                              resultados_cluster['centroides'])

        st.dataframe(perfiles.style.format({
            'porcentaje': '{:.2f}%',
            'n_estudiantes': '{:.0f}'
        }), use_container_width=True)

    # ==================== TAB 4: EXPORTAR ====================
    with tab4:
        st.markdown('<h2 class="sub-header">Exportar Resultados</h2>', unsafe_allow_html=True)

        st.markdown("""
        Descarga los resultados del análisis en diferentes formatos:
        """)

        # Dataset completo con clusters y PCA
        df_export = df_original.copy()
        df_export['Cluster'] = resultados_cluster['labels'] + 1
        df_export['Distancia_Centroide'] = resultados_cluster['distancias']

        for i in range(resultados_pca['n_components']):
            df_export[f'Componente_{i+1}'] = resultados_pca['data_pca'].iloc[:, i]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                label="📥 Descargar Dataset Completo",
                data=df_export.to_csv(index=False).encode('utf-8'),
                file_name='edu_insight_completo.csv',
                mime='text/csv',
                use_container_width=True
            )

        with col2:
            st.download_button(
                label="📥 Descargar Centroides",
                data=resultados_cluster['centroides'].to_csv().encode('utf-8'),
                file_name='edu_insight_centroides.csv',
                mime='text/csv',
                use_container_width=True
            )

        with col3:
            st.download_button(
                label="📥 Descargar Matriz PCA",
                data=resultados_pca['cargas'].to_csv().encode('utf-8'),
                file_name='edu_insight_pca_loadings.csv',
                mime='text/csv',
                use_container_width=True
            )

        # Resumen del análisis
        st.subheader("Resumen del Análisis")

        resumen = f"""
        ### Reporte EDU INSIGHT

        **Fecha:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

        #### Datos
        - Total de registros: {len(df_original):,}
        - Variables analizadas: {len(df_clean.columns)}

        #### PCA
        - Componentes retenidos: {resultados_pca['n_components']}
        - Varianza explicada: {resultados_pca['var_exp'].iloc[resultados_pca['n_components']-1]['Acumulado %']:.2f}%
        - KMO Score: {pca_check['kmo']:.3f if pca_check['kmo'] else 'N/A'}

        #### Clustering
        - Número de clusters: {k_final}
        - Silhouette Score: {resultados_cluster['metricas']['silhouette']:.3f}
        - Davies-Bouldin Index: {resultados_cluster['metricas']['davies_bouldin']:.3f}
        - Calinski-Harabasz Score: {resultados_cluster['metricas']['calinski_harabasz']:.0f}

        #### Distribución de Clusters
        """

        for cluster in range(k_final):
            n_estudiantes = sum(resultados_cluster['labels'] == cluster)
            porcentaje = (n_estudiantes / len(df_original)) * 100
            resumen += f"\n- Cluster {cluster + 1}: {n_estudiantes} estudiantes ({porcentaje:.1f}%)"

        st.markdown(resumen)

        # Descargar resumen
        st.download_button(
            label="📥 Descargar Resumen (TXT)",
            data=resumen.encode('utf-8'),
            file_name='edu_insight_resumen.txt',
            mime='text/plain',
            use_container_width=True
        )

if __name__ == "__main__":
    main()
