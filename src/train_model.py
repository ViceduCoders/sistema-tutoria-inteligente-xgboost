"""
Módulo de Entrenamiento y Validación del Modelo Predictivo
Algoritmo: XGBoost Classifier
Proyecto: Sistema de Tutoría Inteligente (UPN 2026)
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

# 1. Definición de rutas relativas
RUTA_DATA = os.path.join("data", "processed", "dataset_academico_limpio.csv")
RUTA_MODELO = os.path.join("models", "modelo_xgboost_riesgo.pkl")
CARPETA_REPORTES = os.path.join("reports", "figures")


def entrenar_modelo():
  print("Iniciando entrenamiento del modelo XGBoost...")

  if not os.path.exists(RUTA_DATA):
    raise FileNotFoundError(
        f"No se encontró el dataset procesado en: {RUTA_DATA}"
    )

  df = pd.read_csv(RUTA_DATA)

  # 2. Selección de variables predictoras clave (foco en detección temprana)
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

  X = df[predictores]
  y = df["riesgo_academico"]

  # 3. Partición estratificada (75% Train, 25% Test)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.25, random_state=42, stratify=y
  )

  # 4. Cálculo del balance de clases para maximizar Recall
  ratio_desbalance = (len(y_train) - sum(y_train)) / sum(y_train)

  # 5. Configuración y entrenamiento de XGBoost
  modelo = xgb.XGBClassifier(
      n_estimators=120,
      learning_rate=0.06,
      max_depth=4,
      subsample=0.85,
      colsample_bytree=0.85,
      scale_pos_weight=ratio_desbalance,
      random_state=42,
      eval_metric="logloss",
  )

  modelo.fit(X_train, y_train)

  # 6. Serialización y guardado del modelo entrenado
  os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
  joblib.dump(modelo, RUTA_MODELO)
  print(f"Modelo entrenado y serializado exitosamente en: {RUTA_MODELO}")

  # 7. Evaluación computacional en el conjunto de prueba
  y_pred = modelo.predict(X_test)
  y_prob = modelo.predict_proba(X_test)[:, 1]

  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred)
  rec = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  roc = roc_auc_score(y_test, y_prob)

  print("\n" + "=" * 55)
  print("   MÉTRICAS COMPUTACIONALES OBTENIDAS DEL MODELO XGBoost")
  print("=" * 55)
  print(f"Exactitud Global (Accuracy):      {acc:.4f} ({acc*100:.2f}%)")
  print(f"Precisión (Precision):             {prec:.4f} ({prec*100:.2f}%)")
  print(f"Sensibilidad / Recall (OE01):      {rec:.4f} ({rec*100:.2f}%)")
  print(f"Puntuación F1 (F1-Score):          {f1:.4f}")
  print(f"Área Bajo la Curva (ROC-AUC):      {roc:.4f}")
  print("=" * 55)
  print("\nReporte de Clasificación Detallado:")
  print(
      classification_report(
          y_test, y_pred, target_names=["Sin Riesgo (0)", "En Riesgo (1)"]
      )
  )

  # 8. Generación de Gráficos para el Semáforo 2
  os.makedirs(CARPETA_REPORTES, exist_ok=True)
  plt.figure(figsize=(12, 4.5))

  # Subgráfico 1: Matriz de Confusión
  plt.subplot(1, 2, 1)
  cm = confusion_matrix(y_test, y_pred)
  sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap="Blues",
      cbar=False,
      xticklabels=["Sin Riesgo", "En Riesgo"],
      yticklabels=["Sin Riesgo", "En Riesgo"],
  )
  plt.title(
      "Figura: Matriz de Confusión (XGBoost)", fontsize=11, fontweight="bold"
  )
  plt.xlabel("Clasificación del Sistema")
  plt.ylabel("Condición Real del Estudiante")

  # Subgráfico 2: Importancia de Variables (Feature Importance)
  plt.subplot(1, 2, 2)
  importancias = modelo.feature_importances_
  idx_orden = np.argsort(importancias)
  plt.barh(
      range(len(idx_orden)),
      importancias[idx_orden],
      color="#1f77b4",
      align="center",
  )
  plt.yticks(range(len(idx_orden)), [predictores[i] for i in idx_orden])
  plt.title(
      "Figura: Importancia de Factores de Riesgo", fontsize=11, fontweight="bold"
  )
  plt.xlabel("Ganancia Relativa (Feature Importance)")

  plt.tight_layout()
  ruta_graficos = os.path.join(CARPETA_REPORTES, "resultados_xgboost.png")
  plt.savefig(ruta_graficos, dpi=300)
  print(f"Gráficos exportados en alta resolución: {ruta_graficos}\n")


if __name__ == "__main__":
  entrenar_modelo()