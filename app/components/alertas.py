"""
Componente de Alertas Tempranas y Semáforo de Riesgo
Proyecto: Sistema de Tutoría Inteligente basado en XGBoost (UPN 2026)
"""

import os
import joblib
import pandas as pd
import streamlit as st


def render_modulo_alertas():
  st.subheader("📋 Monitoreo de Alertas y Semáforo de Riesgo Académico")
  st.markdown(
      "Detección predictiva a mitad de ciclo (**Semana 7**) orientada a la"
      " intervención tutorial oportuna."
  )

  ruta_data = os.path.join("data", "processed", "dataset_academico_limpio.csv")
  ruta_modelo = os.path.join("models", "modelo_xgboost_riesgo.pkl")

  if not os.path.exists(ruta_data) or not os.path.exists(ruta_modelo):
    st.error("No se encontraron los datos procesados o el modelo entrenado.")
    return

  df = pd.read_csv(ruta_data)
  modelo = joblib.load(ruta_modelo)

  predictores = [
      "edad_matricula",
      "nota_admision",
      "pensiones_al_dia",
      "deudor_financiero",
      "es_becario",
      "cursos_matriculados_sem1",
      "evaluaciones_rendidas_sem1",
      "cursos_aprobados_sem1",
      "promedio_notas_sem1",
      "cursos_sin_evaluar_sem1",
  ]

  # Muestra simulada de una sección de 40 estudiantes para visualización docente
  df_muestra = df.head(40).copy()
  df_muestra["codigo_estudiante"] = [f"EST-2026-{i+1:03d}" for i in range(40)]

  probabilidades = modelo.predict_proba(df_muestra[predictores])[:, 1]
  df_muestra["probabilidad_riesgo"] = (probabilidades * 100).round(1)

  # Asignación del semáforo institucional
  def clasificar_alerta(prob):
    if prob >= 65.0:
      return "🔴 Riesgo Alto"
    elif prob >= 40.0:
      return "🟡 Riesgo Medio"
    else:
      return "🟢 Sin Riesgo"

  df_muestra["alerta_tutorial"] = df_muestra["probabilidad_riesgo"].apply(
      clasificar_alerta
  )

  # Filtro interactivo por nivel de riesgo
  col_filtro1, col_filtro2 = st.columns([2, 1])
  with col_filtro1:
    filtro_alerta = st.multiselect(
        "Filtrar por Nivel de Alerta:",
        options=["🔴 Riesgo Alto", "🟡 Riesgo Medio", "🟢 Sin Riesgo"],
        default=["🔴 Riesgo Alto", "🟡 Riesgo Medio"],
    )
  with col_filtro2:
    st.metric(
        label="Margen de Acción Tutorial",
        value="3 Semanas",
        delta="Anticipación Preventiva",
    )

  df_filtrado = df_muestra[df_muestra["alerta_tutorial"].isin(filtro_alerta)]

  columnas_ver = [
      "codigo_estudiante",
      "alerta_tutorial",
      "probabilidad_riesgo",
      "promedio_notas_sem1",
      "cursos_aprobados_sem1",
      "pensiones_al_dia",
  ]

  df_presentacion = df_filtrado[columnas_ver].rename(
      columns={
          "codigo_estudiante": "Código Alumno",
          "alerta_tutorial": "Nivel de Alerta",
          "probabilidad_riesgo": "Probabilidad (%)",
          "promedio_notas_sem1": "Promedio Sem 1",
          "cursos_aprobados_sem1": "Cursos Aprobados",
          "pensiones_al_dia": "Pensiones al Día (1=Sí)",
      }
  )

  st.dataframe(df_presentacion, use_container_width=True, hide_index=True)

  # Descarga para la bitácora tutorial
  csv_descarga = df_presentacion.to_csv(index=False).encode("utf-8")
  st.download_button(
      label="📥 Exportar Ficha de Derivación Tutorial (CSV)",
      data=csv_descarga,
      file_name="derivacion_estudiantes_riesgo.csv",
      mime="text/csv",
  )