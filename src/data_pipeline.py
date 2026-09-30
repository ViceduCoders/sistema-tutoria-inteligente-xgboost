"""
Módulo de Procesamiento y Transformación de Datos (ETL)
Proyecto: Sistema de Tutoría Inteligente basado en XGBoost (UPN 2026)
"""

import os
import pandas as pd

# 1. Definición de rutas relativas
RUTA_ORIGINAL = os.path.join("data", "raw", "data.csv")
RUTA_DESTINO = os.path.join("data", "processed", "dataset_academico_limpio.csv")

# 2. Diccionario de traducción inglés -> español
COLUMNAS_ESPANOL = {
    'Marital status': 'estado_civil',
    'Application mode': 'modalidad_postulacion',
    'Application order': 'orden_preferencia_carrera',
    'Course': 'carrera',
    'Daytime/evening attendance\t': 'turno_estudio',
    'Previous qualification': 'grado_educacion_previa',
    'Previous qualification (grade)': 'nota_educacion_previa',
    'Nacionality': 'nacionalidad',
    'Mother\'s qualification': 'nivel_educacion_madre',
    'Father\'s qualification': 'nivel_educacion_padre',
    'Mother\'s occupation': 'ocupacion_madre',
    'Father\'s occupation': 'ocupacion_padre',
    'Admission grade': 'nota_admision',
    'Displaced': 'estudiante_foraneo',
    'Educational special needs': 'necesidades_especiales',
    'Debtor': 'deudor_financiero',
    'Tuition fees up to date': 'pensiones_al_dia',
    'Gender': 'genero',
    'Scholarship holder': 'es_becario',
    'Age at enrollment': 'edad_matricula',
    'International': 'estudiante_internacional',
    'Curricular units 1st sem (credited)': 'cursos_convalidados_sem1',
    'Curricular units 1st sem (enrolled)': 'cursos_matriculados_sem1',
    'Curricular units 1st sem (evaluations)': 'evaluaciones_rendidas_sem1',
    'Curricular units 1st sem (approved)': 'cursos_aprobados_sem1',
    'Curricular units 1st sem (grade)': 'promedio_notas_sem1',
    'Curricular units 1st sem (without evaluations)': 'cursos_sin_evaluar_sem1',
    'Curricular units 2nd sem (credited)': 'cursos_convalidados_sem2',
    'Curricular units 2nd sem (enrolled)': 'cursos_matriculados_sem2',
    'Curricular units 2nd sem (evaluations)': 'evaluaciones_rendidas_sem2',
    'Curricular units 2nd sem (approved)': 'cursos_aprobados_sem2',
    'Curricular units 2nd sem (grade)': 'promedio_notas_sem2',
    'Curricular units 2nd sem (without evaluations)': 'cursos_sin_evaluar_sem2',
    'Unemployment rate': 'tasa_desempleo',
    'Inflation rate': 'tasa_inflacion',
    'GDP': 'pbi',
    'Target': 'condicion_final'
}

def ejecutar_pipeline():
    print("Iniciando pipeline de preprocesamiento de datos...")
    
    if not os.path.exists(RUTA_ORIGINAL):
        raise FileNotFoundError(f"No se encontró el archivo en: {RUTA_ORIGINAL}")

    # Carga del archivo con delimitador ';'
    df = pd.read_csv(RUTA_ORIGINAL, sep=";")
    print(f"Dataset original cargado: {df.shape[0]} filas y {df.shape[1]} columnas.")

    # Renombrado de columnas al español
    df = df.rename(columns=COLUMNAS_ESPANOL)

    # Limpieza de espacios en blanco en la columna objetivo
    df['condicion_final'] = df['condicion_final'].astype(str).str.strip()

    # Creación de la variable objetivo binaria (riesgo_academico)
    # 1 = Dropout (Riesgo Alto), 0 = Regular (Graduate / Enrolled)
    df['riesgo_academico'] = df['condicion_final'].apply(lambda x: 1 if x == 'Dropout' else 0)

    # Verificación de distribución de clases
    distribucion = df['riesgo_academico'].value_counts(normalize=True) * 100
    print(f"Distribución del riesgo académico:")
    print(f" - Sin Riesgo (0): {distribucion[0]:.2f}%")
    print(f" - En Riesgo (1):  {distribucion[1]:.2f}%")

    # Guardar el dataset procesado
    os.makedirs(os.path.dirname(RUTA_DESTINO), exist_ok=True)
    df.to_csv(RUTA_DESTINO, index=False)
    print(f"Dataset procesado guardado exitosamente en: {RUTA_DESTINO}\n")

if __name__ == "__main__":
    ejecutar_pipeline()