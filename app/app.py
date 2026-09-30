"""
Sistema de Tutoría Inteligente (STI) basado en XGBoost
Tesis: Detección Temprana del Riesgo Académico - UPN 2026
"""

import os
import sys

# Asegurar que reconozca los módulos internos
sys.path.append(os.path.dirname(__file__))

from components.alertas import render_modulo_alertas
from components.diagnostico import render_modulo_diagnostico
import streamlit as st

st.set_page_config(
    page_title="Sistema de Tutoría Inteligente - UPN",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Sistema de Tutoría Inteligente (STI) - UPN 2026")
st.caption(
    "Detección Temprana del Riesgo Académico en Estudiantes Universitarios"
    " mediante Árboles Potenciados por Gradiente (XGBoost)"
)

# Menú lateral de navegación
st.sidebar.image(
    "app/assets/logo_upn.png",
    width=200,
)
st.sidebar.title("Inicio")
opcion = st.sidebar.radio(
    "Seleccione el Módulo:",
    [
        "📊 Dashboard de Alertas Tempranas",
        "🔍 Diagnóstico y Plan Individual",
        "📈 Métricas Computacionales",
    ],
)

# Renderizado condicional
if opcion == "📊 Dashboard de Alertas Tempranas":
  render_modulo_alertas()

elif opcion == "🔍 Diagnóstico y Plan Individual":
  render_modulo_diagnostico()

elif opcion == "📈 Métricas Computacionales":
  st.subheader("🔬 Evidencias Computacionales del Modelo XGBoost")
  st.markdown(
      "Evaluación sobre conjunto de prueba independiente ($N = 1,106$"
      " registros de estudiantes):"
  )

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Exactitud (Accuracy)", "84.99%")
  col2.metric("Sensibilidad (Recall - OE01)", "78.87%")
  col3.metric("Precisión (Precision)", "75.47%")
  col4.metric("Área ROC (ROC-AUC)", "0.8974")

  ruta_graficos = os.path.join("reports", "figures", "resultados_xgboost.png")
  if os.path.exists(ruta_graficos):
    st.image(
        ruta_graficos,
        caption=(
            "Figura: Matriz de Confusión e Importancia de Factores de Riesgo"
            " (300 DPI)"
        ),
        use_container_width=True,
    )
  else:
    st.info(
        "El gráfico se encuentra en la carpeta de reportes del repositorio."
    )