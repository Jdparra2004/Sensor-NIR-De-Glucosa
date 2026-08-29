import pandas as pd
import numpy as np
import os

def preparar_datos_chunked():
    raw_path = 'data/CalibrationData_NTNU.txt'
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    parquet_path = os.path.join(processed_dir, 'muestras_referencia_nir.parquet')
    
    # 2. Configuración para procesamiento en chunks
    chunk_size = 5000  # Reducido para mayor seguridad
    
    # Usar integer indexing para evitar problemas de columnas complejas
    # Columnas esperadas: 
    # 0: Glucosa, 2: Lactato, 5: Temperatura, 9-end: Absorbancias
    # Las columnas de absorbancia comienzan en el índice 9.
    
    first_chunk = True
    
    print("Iniciando procesamiento de datos en chunks...")
    
    # Procesar archivo por partes, saltando las dos primeras líneas
    # header=None: No tomamos ninguna línea como nombres de columna
    for chunk in pd.read_csv(raw_path, sep=r'\s+', header=None, skiprows=2, 
                             encoding='utf-16', chunksize=chunk_size):
        
        # Seleccionar las columnas por índice numérico
        # 0: Glucosa, 2: Lactato, 5: Temperatura
        # Para las absorbancias, tomaremos las columnas 9 en adelante
        
        # Filtramos y renombramos
        # Columnas: 0 (glucosa), 2 (lactato), 5 (temperatura), 609 (absorbancia 1600nm aprox), 659 (absorbancia 1650nm aprox)
        # Ajuste de índices basado en que empiezan en 9.
        df_chunk = chunk.iloc[:, [0, 2, 5, 609, 659]].copy()
        df_chunk.columns = ['glucosa_referencia_mM', 'lactato_mM', 'temperatura_C', 'absorbancia_1600nm', 'absorbancia_1650nm']
        
        # Nota: Aquí no estamos incluyendo todas las absorbancias por simplicidad y memoria.
        # Si las necesitas, tendrías que seleccionar más columnas numéricas.
        
        # 3. Guardar en Parquet (append)
        if first_chunk:
            df_chunk.to_parquet(parquet_path, engine='pyarrow', index=True)
            first_chunk = False
        else:
            df_chunk.to_parquet(parquet_path, engine='pyarrow', index=True, append=True)
            
    print(f"Procesamiento completado. Datos guardados en: {parquet_path}")

if __name__ == '__main__':
    preparar_datos_chunked()
