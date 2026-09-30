"""
Componente de Diagnóstico Predictivo Individual y Prescripción Tutorial
Proyecto: Sistema de Tutoría Inteligente basado en XGBoost (UPN 2026)
"""

import os
import joblib
import pandas as pd
import streamlit as st


def render_modulo_diagnostico():
  st.subheader("🔍 Diagnóstico Individual del Estudiante")
  st.markdown(
      "Ingrese el perfil académico y financiero del estudiante para evaluar el"
      " nivel de riesgo y obtener el plan de acción sugerido."
  )

  ruta_modelo = os.path.join("models", "modelo_xgboost_riesgo.pkl")
  if not os.path.exists(ruta_modelo):
    st.error("No se encontró el modelo serializado.")
    return

  modelo = joblib.load(ruta_modelo)

  # Formulario interactivo
  with st.form("form_estudiante"):
    col1, col2 = st.columns(2)

    with col1:
      st.markdown("**Variables Académicas Semestre 1:**")
      cursos_mat = st.number_input(
          "Cursos Matriculados", min_value=1, max_value=10, value=6
      )
      cursos_aprob = st.number_input(
          "Cursos Aprobados", min_value=0, max_value=10, value=2
      )
      eval_rendidas = st.number_input(
          "Evaluaciones Rendidas", min_value=0, max_value=20, value=5
      )
      cursos_sin_eval = st.number_input(
          "Cursos Sin Evaluar (Abandono)", min_value=0, max_value=5, value=1
      )
      promedio_sem1 = st.slider(
          "Promedio Ponderado Semestre 1", 0.0, 20.0, 9.5, 0.1
      )

    with col2:
      st.markdown("**Variables Socioeconómicas y de Ingreso:**")
      nota_admision = st.slider("Nota de Admisión", 0.0, 20.0, 11.5, 0.1)
      edad_matricula = st.number_input(
          "Edad al Matricularse", min_value=16, max_value=60, value=21
      )
      pensiones_al_dia = st.selectbox(
          "Pensiones al Día", options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No"
      )
      deudor = st.selectbox(
          "Deudas Financieras Registradas",
          options=[0, 1],
          format_func=lambda x: "No" if x == 0 else "Sí",
      )
      es_becario = st.selectbox(
          "Cuenta con Beca",
          options=[0, 1],
          format_func=lambda x: "No" if x == 0 else "Sí",
      )

    btn_evaluar = st.form_submit_button("⚡ Evaluar Nivel de Riesgo")

  if btn_evaluar:
    # Preparar vector de entrada
    datos_entrada = pd.DataFrame([{
        "edad_matricula": edad_matricula,
        "nota_admision": nota_admision,
        "pensiones_al_dia": pensiones_al_dia,
        "deudor_financiero": deudor,
        "es_becario": es_becario,
        "cursos_matriculados_sem1": cursos_mat,
        "evaluaciones_rendidas_sem1": eval_rendidas,
        "cursos_aprobados_sem1": cursos_aprob,
        "promedio_notas_sem1": promedio_sem1,
        "cursos_sin_evaluar_sem1": cursos_sin_eval,
    }])

    prob_riesgo = modelo.predict_proba(datos_entrada)[0][1] * 100

    st.markdown("---")
    st.markdown("### 📊 Resultado del Diagnóstico Predictivo (XGBoost)")

    col_res1, col_res2 = st.columns([1, 2])
    with col_res1:
      st.metric(
          label="Probabilidad de Riesgo Académico",
          value=f"{prob_riesgo:.1f}%",
          delta="Riesgo Alto" if prob_riesgo >= 65 else "Riesgo Moderado",
          delta_color="inverse",
      )

    with col_res2:
      if prob_riesgo >= 65:
        st.error(
            "🔴 **Condición Crítica: Riesgo Alto de Reprobación / Abandono.**"
        )
        st.info(
            "**Plan de Acción Tutorial Recomendado:**\n\n"
            "1. Agendar sesión de tutoría individual de nivelación en semanas 8"
            " y 9.\n"
            "2. Notificación prioritaria por inasistencias en cursos no"
            " evaluados.\n"
            "3. Derivación al área de bienestar estudiantil por condición"
            " financiera si corresponde."
        )
      elif prob_riesgo >= 40:
        st.warning("🟡 **Condición Media: Estudiante en Observación.**")
        st.info(
            "**Plan de Acción Tutorial Recomendado:**\n\n"
            "1. Realizar seguimiento al promedio de evaluaciones continuas.\n"
            "2. Recomendar talleres de reforzamiento académico grupal."
        )
      else:
        st.success("🟢 **Condición Regular: Desempeño Estable.**")
        st.info(
            "El estudiante mantiene indicadores favorables. Continuar con el"
            " monitoreo periódico regular."
        )