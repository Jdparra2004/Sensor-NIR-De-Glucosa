import pandas as pd
import numpy as np
import os

def preparar_datos_chunked():
    raw_path = 'data/CalibrationData_NTNU.txt'
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    parquet_path = os.path.join(processed_dir, 'muestras_referencia_nir.parquet')
    # 1. Obtener nombres de columnas y longitudes de onda (pequeño, cabe en RAM)
    with open(raw_path, 'r', encoding='utf-16') as f:
        lines = f.readlines()
        # La estructura muestra que los primeros 9 elementos no son longitudes de onda
        # Ajustamos para tener exactamente las 4200 longitudes de onda necesarias
        all_wavelengths = [float(w) for w in lines[1].split()[9:]]

    base_cols = ['glucosa_referencia_mM', 'lactato_mM', 'acetaminophen_mM', 'caffeine_mM', 'ethanol_mM', 
                  'temperatura_C', 'kuvette', 'day', 'run']

    col_names = base_cols + [f'absorbancia_{int(float(wl))}nm_{i}' for i, wl in enumerate(all_wavelengths)]

    
    # 2. Configuración para procesamiento en chunks
    chunk_size = 10000  # Ajustar según la memoria disponible
    target_wavelengths = [1000, 1100, 1200, 1400, 1450, 1600, 1650, 1700]
    selected_cols = ['glucosa_referencia_mM', 'lactato_mM', 'temperatura_C'] + [f'absorbancia_{wl}nm' for wl in target_wavelengths]
    
    # Asegurarse de que las columnas seleccionadas existen en col_names
    # (podría ser necesario ajustar los nombres si hay discrepancias)
    
    first_chunk = True
    
    print("Iniciando procesamiento de datos en chunks...")
    
    # Procesar archivo por partes
    for chunk in pd.read_csv(raw_path, sep=r'\s+', header=None, skiprows=2, 
                             encoding='utf-16', chunksize=chunk_size, names=col_names):
        
        # Filtrar columnas
        df_chunk = chunk[selected_cols].copy()
        
        # 3. Guardar en Parquet (append)
        # Usar pyarrow para manejar el append eficiente
        if first_chunk:
            df_chunk.to_parquet(parquet_path, engine='pyarrow', index=True)
            first_chunk = False
        else:
            df_chunk.to_parquet(parquet_path, engine='pyarrow', index=True, append=True)
            
    print(f"Procesamiento completado. Datos guardados en: {parquet_path}")

if __name__ == '__main__':
    preparar_datos_chunked()
