import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="GEI Colombia Dashboard", page_icon="🌍", layout="wide")

# Estilos CSS adicionales
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1, h2, h3 {color: #2c3e50;}
    .stButton>button {background-color: #27ae60; color: white; border-radius: 5px;}
    .kpi-card {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;}
    .kpi-value {font-size: 2rem; font-weight: bold; color: #27ae60;}
    .kpi-label {font-size: 1rem; color: #7f8c8d;}
    </style>
""", unsafe_allow_html=True)

# Cargar Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset/processed/dataset_limpio.csv', thousands=',')
    return df

df = load_data()

# Cargar Modelos y Configuración
@st.cache_resource
def load_models():
    model_config = joblib.load('models/model_config.pkl')
    scaler = joblib.load('models/scaler.pkl')
    lr = joblib.load('models/regresion_lineal.pkl')
    dt = joblib.load('models/arbol_decision.pkl')
    rf = joblib.load('models/random_forest.pkl')
    metrics = joblib.load('models/metrics.pkl')
    return model_config, scaler, lr, dt, rf, metrics

try:
    model_config, scaler, lr, dt, rf, metrics = load_models()
except Exception as e:
    st.error(f"Error cargando modelos: {e}. Asegúrate de haber ejecutado train_and_save_models.py")
    st.stop()

# Navegación Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2942/2942784.png", width=100)
st.sidebar.title("Navegación")
menu = st.sidebar.radio("Ir a:", [
    "Inicio", 
    "Dataset", 
    "EDA", 
    "Regresión Lineal", 
    "Árbol de Decisión", 
    "Comparación de Modelos", 
    "Predicción", 
    "Conclusiones"
])

# --- INICIO ---
if menu == "Inicio":
    st.title("🌍 Análisis de Emisiones Netas de GEI en Colombia")
    st.markdown("""
    Bienvenido al Dashboard Interactivo del proyecto de análisis de Gases de Efecto Invernadero (GEI) en los departamentos de Colombia.
    
    ### Objetivos del Proyecto
    - Identificar los departamentos más y menos contaminantes.
    - Analizar las diferencias sectoriales en la emisión y absorción de gases.
    - Construir modelos predictivos para estimar las **Emisiones Netas**.
    - Apoyar decisiones ambientales basadas en datos.
    
    ### Fórmula Principal
    > **Emisiones Netas = Emisiones Totales - Absorciones Totales**
    """)
    
    st.markdown("### KPIs Principales")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df):,}</div><div class="kpi-label">Registros Analizados</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{df["departamento"].nunique()}</div><div class="kpi-label">Departamentos</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{df["sector_principal"].nunique()}</div><div class="kpi-label">Sectores Principales</div></div>', unsafe_allow_html=True)

# --- DATASET ---
elif menu == "Dataset":
    st.title("📊 Exploración del Dataset")
    st.markdown("Visualiza y explora los datos limpios después del proceso de ETL.")
    
    st.dataframe(df.head(100), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(df.describe())
    with col2:
        st.subheader("Tipos de Datos y Nulos")
        info_df = pd.DataFrame({'Tipo': df.dtypes, 'Nulos': df.isnull().sum()})
        st.dataframe(info_df)

# --- EDA ---
elif menu == "EDA":
    st.title("📈 Análisis Exploratorio de Datos (EDA)")
    st.markdown("Selecciona una gráfica generada durante la fase exploratoria:")
    
    eda_images = [f for f in os.listdir('images/eda') if f.endswith('.png')] if os.path.exists('images/eda') else []
    
    if eda_images:
        selected_img = st.selectbox("Elige una visualización:", eda_images)
        image_path = os.path.join('images/eda', selected_img)
        st.image(Image.open(image_path), caption=selected_img, use_column_width=True)
    else:
        st.warning("No se encontraron imágenes en la carpeta images/eda/")

# --- REGRESIÓN LINEAL ---
elif menu == "Regresión Lineal":
    st.title("📉 Modelo: Regresión Lineal")
    st.markdown("""
    La Regresión Lineal asume una relación aditiva y proporcional entre las variables. 
    Se escalaron las variables predictoras (StandardScaler) para poder comparar los coeficientes.
    """)
    
    st.subheader("Métricas de Evaluación (Conjunto de Prueba)")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics['Regresión Lineal']['R2']:.4f}")
    col2.metric("MAE", f"{metrics['Regresión Lineal']['MAE']:,.2f}")
    col3.metric("RMSE", f"{metrics['Regresión Lineal']['RMSE']:,.2f}")
    
    st.markdown("### Visualizaciones del Modelo")
    ml_images = [f for f in os.listdir('images/ml') if f.startswith('01') or f.startswith('02') or f.startswith('03') or f.startswith('04')] if os.path.exists('images/ml') else []
    if ml_images:
        selected_img = st.selectbox("Ver gráfica:", ml_images, key='lr_img')
        st.image(Image.open(os.path.join('images/ml', selected_img)), use_column_width=True)

# --- ÁRBOL DE DECISIÓN ---
elif menu == "Árbol de Decisión":
    st.title("🌳 Modelo: Árbol de Decisión")
    st.markdown("""
    Los árboles de decisión capturan relaciones no lineales y crean reglas lógicas explícitas para predecir las emisiones netas.
    """)
    
    st.subheader("Métricas de Evaluación (Conjunto de Prueba)")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics['Árbol de Decisión']['R2']:.4f}")
    col2.metric("MAE", f"{metrics['Árbol de Decisión']['MAE']:,.2f}")
    col3.metric("RMSE", f"{metrics['Árbol de Decisión']['RMSE']:,.2f}")
    
    st.markdown("### Estructura e Importancia")
    dt_images = [f for f in os.listdir('images/ml') if 'arbol' in f or 'importancia' in f or 'profundidad' in f] if os.path.exists('images/ml') else []
    if dt_images:
        selected_img = st.selectbox("Ver gráfica:", dt_images, key='dt_img')
        st.image(Image.open(os.path.join('images/ml', selected_img)), use_column_width=True)

# --- COMPARACIÓN DE MODELOS ---
elif menu == "Comparación de Modelos":
    st.title("📊 Comparación de Modelos")
    
    df_metrics = pd.DataFrame(metrics).T.reset_index()
    df_metrics.columns = ['Modelo', 'R2', 'MAE', 'RMSE']
    
    st.dataframe(df_metrics.style.highlight_max(subset=['R2'], color='lightgreen').highlight_min(subset=['MAE', 'RMSE'], color='lightgreen'), use_container_width=True)
    
    fig_r2 = px.bar(df_metrics, x='Modelo', y='R2', color='Modelo', title='Comparación de R² (Mayor es mejor)', text_auto='.4f')
    st.plotly_chart(fig_r2, use_container_width=True)
    
    fig_error = go.Figure()
    fig_error.add_trace(go.Bar(x=df_metrics['Modelo'], y=df_metrics['MAE'], name='MAE'))
    fig_error.add_trace(go.Bar(x=df_metrics['Modelo'], y=df_metrics['RMSE'], name='RMSE'))
    fig_error.update_layout(title='Comparación de Errores Absolutos (Menor es mejor)', barmode='group')
    st.plotly_chart(fig_error, use_container_width=True)

# --- PREDICCIÓN ---
elif menu == "Predicción":
    st.title("🔮 Simulador de Predicciones")
    st.markdown("Ingresa los valores hipotéticos para generar una predicción de **Emisiones Netas** utilizando el modelo seleccionado.")
    
    col1, col2 = st.columns(2)
    with col1:
        modelo_seleccionado = st.selectbox("Selecciona el Modelo", ["Regresión Lineal", "Árbol de Decisión", "Random Forest"])
        departamento = st.selectbox("Departamento", sorted(model_config['departamentos']))
        sector = st.selectbox("Sector Principal", sorted(model_config['sectores']))
    with col2:
        co2 = st.number_input("Emisiones de CO2", min_value=0.0, value=100.0)
        ch4 = st.number_input("Emisiones de CH4", min_value=0.0, value=10.0)
        n2o = st.number_input("Emisiones de N2O", min_value=0.0, value=5.0)
    
    if st.button("Generar Predicción", type='primary'):
        # Crear un dataframe con 0s usando las columnas esperadas
        input_data = pd.DataFrame(0, index=[0], columns=model_config['columns'])
        
        # Llenar valores numéricos hipotéticos (simplificado para el ejemplo)
        # En una versión completa habría que solicitar todos los inputs numéricos o llenar con medianas
        for col in input_data.columns:
            if col == 'co2': input_data[col] = co2
            elif col == 'ch4': input_data[col] = ch4
            elif col == 'n2o': input_data[col] = n2o
            # Set other numericals to 0 or median
            
        # Activar las columnas dummy correspondientes
        dept_col = f"departamento_{departamento}"
        sector_col = f"sector_principal_{sector}"
        
        if dept_col in input_data.columns:
            input_data[dept_col] = 1
        if sector_col in input_data.columns:
            input_data[sector_col] = 1
            
        # Seleccionar modelo y predecir
        if modelo_seleccionado == "Regresión Lineal":
            input_scaled = scaler.transform(input_data)
            prediccion = lr.predict(input_scaled)[0]
        elif modelo_seleccionado == "Árbol de Decisión":
            prediccion = dt.predict(input_data)[0]
        else:
            prediccion = rf.predict(input_data)[0]
            
        st.success(f"### Predicción de Emisiones Netas: {prediccion:,.2f} Toneladas")
        st.info("Nota: Las variables numéricas no ingresadas se asumieron como 0 para esta simulación.")

# --- CONCLUSIONES ---
elif menu == "Conclusiones":
    st.title("💡 Conclusiones y Hallazgos")
    
    st.markdown("""
    ### 1. Variables Clave
    Se descubrió que gases equivalentes específicos como el **CO2eq** y la participación de sectores como **Energía** y **AFOLU** dominan la varianza en las emisiones netas.
    
    ### 2. Diferencias Territoriales
    Existen departamentos con altos índices de emisiones vinculados a zonas industriales o alta deforestación, mientras que otros departamentos actúan principalmente como sumideros de carbono, presentando valores de emisiones netas más bajos (o incluso negativos).
    
    ### 3. Rendimiento de Modelos
    - La **Regresión Lineal** logra un ajuste casi perfecto cuando se incluyen todos los gases de forma aditiva.
    - Los **Árboles de Decisión y Random Forest** demostraron ser extremadamente robustos para encontrar umbrales no lineales y aislar dinámicas complejas sectoriales.
    
    ### 🌱 Recomendaciones Ambientales
    1. **Fijar Umbrales (Thresholds):** Basados en las reglas extraídas de los árboles de decisión, se deberían implementar normativas que restrinjan la emisión al llegar a ciertos puntos críticos donde el impacto escala agresivamente.
    2. **Protección de Sumideros:** Es crucial proteger y expandir las áreas forestales y de conservación en departamentos que actualmente mitigan la huella nacional.
    3. **Monitoreo Sectorial:** Focalizar las auditorías de GEI en los sectores señalados por los algoritmos como de mayor importancia predictiva.
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/3233/3233483.png", width=150)
