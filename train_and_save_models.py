import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

print("Iniciando entrenamiento y serialización de modelos...")

# Asegurar que existe el directorio
os.makedirs('models', exist_ok=True)

# 1. Carga del Dataset
df = pd.read_csv('Dataset/processed/dataset_limpio.csv', thousands=',')

# 2. Preparación (Data Leakage y Codificación)
columnas_fuga = ['emisiones_totales', 'abosorciones_totales', 'mod', 'sub', 'nrom', 'categorias']
cols_to_drop = [c for c in columnas_fuga if c in df.columns]
df_modelo = df.drop(columns=cols_to_drop)

cat_cols = ['departamento', 'sector_principal']
cat_cols = [c for c in cat_cols if c in df_modelo.columns]

# Variables para guardar luego y poder reconstruir en Streamlit
departamentos = df['departamento'].unique().tolist() if 'departamento' in df.columns else []
sectores = df['sector_principal'].unique().tolist() if 'sector_principal' in df.columns else []

df_modelo = pd.get_dummies(df_modelo, columns=cat_cols, drop_first=True)
df_modelo = df_modelo.dropna()


X = df_modelo.drop(columns=['emisiones_netas'])
y = df_modelo['emisiones_netas']

# 3. Guardar las columnas esperadas por el modelo para Streamlit
expected_columns = list(X.columns)
joblib.dump({
    'columns': expected_columns,
    'departamentos': departamentos,
    'sectores': sectores
}, 'models/model_config.pkl')

# 4. División Train / Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 5. Escalamiento (Solo entrenamos el scaler en X_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardamos el scaler
joblib.dump(scaler, 'models/scaler.pkl')

def evaluar_modelo(modelo, X_t, y_t, nombre):
    y_pred = modelo.predict(X_t)
    r2 = r2_score(y_t, y_pred)
    mae = mean_absolute_error(y_t, y_pred)
    rmse = np.sqrt(mean_squared_error(y_t, y_pred))
    print(f"--- {nombre} ---")
    print(f"R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    return r2, mae, rmse

# 6. Entrenar y guardar Regresión Lineal
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
r2_lr, mae_lr, rmse_lr = evaluar_modelo(lr, X_test_scaled, y_test, "Regresión Lineal")
joblib.dump(lr, 'models/regresion_lineal.pkl')

# 7. Entrenar y guardar Árbol de Decisión
# Entrenamos con los datos sin escalar (aunque escalar no afecta al árbol, es la práctica estándar)
dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
r2_dt, mae_dt, rmse_dt = evaluar_modelo(dt, X_test, y_test, "Árbol de Decisión")
joblib.dump(dt, 'models/arbol_decision.pkl')

# 8. Entrenar y guardar Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
r2_rf, mae_rf, rmse_rf = evaluar_modelo(rf, X_test, y_test, "Random Forest")
joblib.dump(rf, 'models/random_forest.pkl')

# Guardar métricas comparativas
metrics = {
    'Regresión Lineal': {'R2': r2_lr, 'MAE': mae_lr, 'RMSE': rmse_lr},
    'Árbol de Decisión': {'R2': r2_dt, 'MAE': mae_dt, 'RMSE': rmse_dt},
    'Random Forest': {'R2': r2_rf, 'MAE': mae_rf, 'RMSE': rmse_rf}
}
joblib.dump(metrics, 'models/metrics.pkl')

print("Todos los modelos y configuraciones se han guardado exitosamente en la carpeta models/.")
