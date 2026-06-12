# Análisis de Emisiones Netas de GEI en Colombia - Proyecto Talento Tech

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-%23FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)

## 🎯 Objetivos del Proyecto

El objetivo principal de este proyecto es realizar un análisis integral de los datos relacionados con las emisiones de Gases de Efecto Invernadero (GEI) en los departamentos de Colombia, pasando por todo el ciclo de vida de un proyecto de Ciencia de Datos:

1. **Extracción, Transformación y Carga (ETL):** Procesamiento de datos crudos.
2. **Análisis Exploratorio de Datos (EDA):** Identificación de patrones, tendencias y correlaciones.
3. **Machine Learning:** Entrenamiento de modelos predictivos (Regresión Lineal, Árbol de Decisión, Random Forest) para estimar las **Emisiones Netas**.
4. **Capa de Presentación:** Una Landing Page estática y un Dashboard interactivo en Streamlit para la comunicación de resultados.

## 📁 Estructura del Proyecto

```text
.
├── app.py                      # Dashboard Interactivo de Streamlit
├── requirements.txt            # Dependencias del proyecto
├── train_and_save_models.py    # Script para generar y guardar los modelos ML
├── models/                     # Modelos guardados en formato .pkl
├── Dataset/processed/          # Dataset procesado (dataset_limpio.csv)
├── images/                     # Gráficas generadas
│   ├── eda/                    # Imágenes del Análisis Exploratorio
│   └── ml/                     # Gráficas de evaluación de los modelos
├── landing/                    # Archivos de la Landing Page web
│   ├── index.html              
│   ├── css/                    
│   └── js/                     
├── ETL/                        # Notebooks de limpieza
├── EDA/                        # Notebooks de exploración
└── ML/                         # Notebooks de modelado inicial
```

## 🚀 Cómo Ejecutar el Proyecto

### 1. Landing Page
Para visualizar la landing page introductoria del proyecto, simplemente abre el archivo `landing/index.html` en tu navegador web de preferencia.

### 2. Dashboard de Streamlit
Para ejecutar la aplicación interactiva con los modelos de Machine Learning, sigue estos pasos en tu terminal:

1. **Instalar Dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Entrenar y Guardar Modelos (Opcional si ya existen en models/):**
   ```bash
   python train_and_save_models.py
   ```
3. **Lanzar la Aplicación:**
   ```bash
   streamlit run app.py
   ```
   La aplicación se abrirá en `http://localhost:8501`.

## 📈 Hallazgos Principales
* El sector **Energía** y **AFOLU** son los determinantes primarios de las emisiones netas.
* Se construyeron modelos altamente precisos donde el **Árbol de Decisión** y **Random Forest** permitieron aislar umbrales de contaminación y reglas explícitas de negocio de forma superior a los enfoques puramente lineales.

---
*Desarrollado como Proyecto Final para Talento Tech.*
