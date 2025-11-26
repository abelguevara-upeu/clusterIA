# EDU INSIGHT 🎓

Sistema de Análisis de Perfiles Estudiantiles con Clustering y PCA

## Descripción

**EDU INSIGHT** es una aplicación web interactiva construida con Streamlit que permite analizar datos estudiantiles mediante:

- **Análisis de Componentes Principales (PCA)**: Reducción de dimensionalidad para identificar las variables más relevantes
- **Clustering K-Means**: Segmentación automática de estudiantes en grupos con características similares
- **Visualizaciones Interactivas**: Gráficos 2D y 3D para explorar patrones en los datos

## Características

✨ **Interfaz Elegante**: Diseño moderno con gradientes y visualizaciones interactivas
📊 **Análisis Completo**: PCA con scree plots, matriz de cargas y varianza explicada
🎯 **Clustering Inteligente**: Detección automática del número óptimo de clusters
📈 **Gráficos Avanzados**: Plotly para visualizaciones 3D, heatmaps y radar charts
📥 **Exportación**: Descarga resultados en CSV y reportes en texto

## Instalación

1. Clona o descarga este repositorio

2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

1. Ejecuta la aplicación:

```bash
streamlit run app.py
```

2. La aplicación se abrirá en tu navegador (por defecto en `http://localhost:8501`)

3. Configura los parámetros en el panel lateral:
   - Carga tu archivo CSV o usa los datos sintéticos por defecto
   - Ajusta el número máximo de clusters a evaluar
   - Opcionalmente, especifica el número de componentes PCA manualmente

4. Presiona **"Iniciar Análisis"** y explora los resultados en las pestañas:
   - **Análisis PCA**: Componentes principales y variables más influyentes
   - **Clustering**: Segmentación de estudiantes y métricas de calidad
   - **Exploración**: Filtros interactivos y estadísticas por cluster
   - **Exportar**: Descarga datasets y reportes

## Estructura del Proyecto

```
clusterIA/
├── app.py                 # Aplicación principal Streamlit
├── preprocessing.py       # Módulo de preprocesamiento de datos
├── models.py             # Módulo de PCA y Clustering
├── requirements.txt      # Dependencias del proyecto
├── sintetic_data.csv    # Dataset sintético de ejemplo
└── README.md            # Este archivo
```

## Datos de Entrada

El sistema espera un archivo CSV con datos estudiantiles. Puede contener variables categóricas y numéricas como:

- Asistencia (porcentajes)
- Horas de estudio
- Calificaciones
- Hábitos de vida
- Factores socioeconómicos
- Variables psicológicas (estrés, autoeficacia, etc.)

Las variables categóricas se codifican automáticamente usando Label Encoding.

## Tecnologías

- **Streamlit**: Framework para aplicaciones web interactivas
- **Scikit-learn**: Machine Learning (PCA, K-Means, métricas)
- **Plotly**: Visualizaciones interactivas
- **Pandas/NumPy**: Manipulación de datos
- **Seaborn/Matplotlib**: Visualizaciones estáticas

## Métodos y Métricas

### PCA
- **Criterio de Kaiser**: Retiene componentes con autovalor > 1
- **KMO (Kaiser-Meyer-Olkin)**: Evalúa la viabilidad del PCA
- **Scree Plot**: Visualiza la importancia de cada componente

### Clustering
- **Silhouette Score**: Mide la cohesión y separación de clusters (0.5-0.7 = bueno)
- **Davies-Bouldin Index**: Evalúa la compacidad de clusters (< 1.0 = bueno)
- **Calinski-Harabasz**: Mide la densidad y separación (mayor es mejor)
- **Método del Codo**: Analiza la inercia para seleccionar k óptimo

## Autor

Desarrollado para el análisis de datos educativos

## Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.
