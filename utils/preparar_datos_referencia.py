# -*- coding: utf-8 -*-
"""
Script de preparación de datos de referencia (NTNU)
Lee los archivos .txt de datos crudos, aplica SNV (Standard Normal Variate)
a las columnas espectrales y guarda el resultado procesado en formato CSV.
"""

import os
import pandas as pd
import numpy as np

def aplicar_snv(df_espectros):
    """
    Aplica Standard Normal Variate (SNV) a las columnas espectrales.
    SNV = (X - media) / desviación_estándar (fila por fila).
    """
    # Asegurar que todas las columnas espectrales sean numéricas
    df_num = df_espectros.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    media_filas = df_num.mean(axis=1)
    std_filas = df_num.std(axis=1, ddof=1)
    # Evitar división por cero
    std_filas = std_filas.replace(0, 1e-8)
    df_snv = df_num.sub(media_filas, axis=0).div(std_filas, axis=0)
    return df_snv

def preparar_datos():
    # Rutas relativas y absolutas dentro del workspace de la aplicación
    input_path = 'data/raw/ValidationData_NTNU.txt'
    if not os.path.exists(input_path):
        input_path = '../ValidationData_NTNU.txt'
        if not os.path.exists(input_path):
            input_path = 'ValidationData_NTNU.txt'
            if not os.path.exists(input_path):
                input_path = 'data/CalibrationData_NTNU.txt'

    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'muestras_referencia_nir.csv')

    print(f"Leyendo archivo de datos crudos desde: {input_path}")
    
    # Intentar lectura con encoding utf-16 (común en archivos NTNU) y fallback a utf-8 / latin1
    df = None
    for enc in ['utf-16', 'utf-8', 'latin1']:
        try:
            df = pd.read_csv(input_path, sep=r'\s+', header=None, encoding=enc, engine='python', on_bad_lines='skip')
            if not df.empty:
                break
        except Exception:
            try:
                df = pd.read_csv(input_path, sep=r'\s+', header=None, encoding=enc, engine='python', error_bad_lines=False)
                if not df.empty:
                    break
            except Exception:
                continue

    if df is None or df.empty:
        raise ValueError("No se pudo leer el archivo de datos crudos con los encodings soportados.")

    # Asumir que la primera columna es la glucosa_referencia_mM
    df.rename(columns={0: 'glucosa_referencia_mM'}, inplace=True)
    
    # Asegurar que la columna de referencia sea numérica
    glucosa_ref = pd.to_numeric(df['glucosa_referencia_mM'], errors='coerce').fillna(0.0)
    
    # Separar la columna de referencia y las columnas espectrales (restantes)
    columnas_espectrales = [col for col in df.columns if col != 'glucosa_referencia_mM']
    df_espectros = df[columnas_espectrales]

    print("Aplicando transformación Standard Normal Variate (SNV) a las columnas espectrales...")
    df_espectros_snv = aplicar_snv(df_espectros)

    # Reconstruir el DataFrame final con la glucosa y los espectros normalizados
    df_final = pd.concat([glucosa_ref, df_espectros_snv], axis=1)

    # Guardar el resultado procesado como CSV
    print(f"Guardando resultado procesado en: {output_path}")
    df_final.to_csv(output_path, index=False)

    print(f"¡Proceso completado con éxito! Tamaño final del dataset procesado: {df_final.shape}")
    return df_final

if __name__ == '__main__':
    preparar_datos()
