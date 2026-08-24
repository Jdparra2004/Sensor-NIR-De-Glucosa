import pandas as pd
import numpy as np
import os

def preparar_datos():
    raw_path = 'data/raw/CalibrationData_NTNU.txt'
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Leer y extraer muestra representativa
    # Usamos codificación utf-16 basada en estructura común de archivos NTNU
    # Asumimos que la estructura es: columnas 0-4 (Concentraciones), 
    # 5 (Temp), 6-8 (Metadatos), 9+ (Absorbancias)
    
    with open(raw_path, 'r', encoding='utf-16') as f:
        lines = f.readlines()
        # La segunda línea contiene las longitudes de onda
        all_wavelengths = [float(w) for w in lines[1].split()[4:]]
    
    # Leer el dataset, saltando las 2 líneas de cabecera
    df = pd.read_csv(raw_path, sep=r'\s+', header=None, skiprows=2, encoding='utf-16')
    
    # Mapear nombres de columnas
    base_cols = ['glucosa_referencia_mM', 'lactato_mM', 'acetaminophen_mM', 'caffeine_mM', 'ethanol_mM', 
                  'temperatura_C', 'kuvette', 'day', 'run']
    
    # Crear diccionario de mapeo
    col_map = {i: name for i, name in enumerate(base_cols)}
    # Mapear el resto a longitudes de onda
    for i, wl in enumerate(all_wavelengths):
        col_map[i + 9] = f'absorbancia_{int(wl)}nm'
    
    df.rename(columns=col_map, inplace=True)
    
    # 2. Selección de columnas para formato ancho estructurado
    target_wavelengths = [1000, 1100, 1200, 1400, 1450, 1600, 1650, 1700]
    selected_cols = ['glucosa_referencia_mM', 'lactato_mM', 'temperatura_C'] + [f'absorbancia_{wl}nm' for wl in target_wavelengths]
    
    # Filtrar y tomar muestra de 300
    df_structured = df[selected_cols].copy().sample(n=300, random_state=42)
    df_structured.reset_index(drop=True, inplace=True)
    df_structured.index.name = 'id_muestra'
    df_structured.reset_index(inplace=True)
    df_structured['id_muestra'] += 1 # Empezar en 1
    
    # Guardar CSV de referencia
    df_structured.to_csv(os.path.join(processed_dir, 'muestras_referencia_nir.csv'), index=False, encoding='utf-8')
    
    # 3. Generar muestras sintéticas de control
    n_synthetic = 100
    data_synth = {
        'id_muestra': range(1, n_synthetic + 1),
        'glucosa_referencia_mM': np.linspace(0.01, 1.0, n_synthetic),
        'lactato_mM': np.random.uniform(0.5, 5.0, n_synthetic),
        'temperatura_C': np.random.uniform(30.0, 37.0, n_synthetic)
    }
    df_synth = pd.DataFrame(data_synth)
    
    for wl in target_wavelengths:
        # Modelo sintético simple: A = eps * C
        df_synth[f'absorbancia_{wl}nm'] = 0.05 * df_synth['glucosa_referencia_mM'] + np.random.normal(0, 0.0005, n_synthetic)
        
    df_synth.to_csv(os.path.join(processed_dir, 'muestras_control_sinteticas.csv'), index=False, encoding='utf-8')
    
    # Verificación final
    print(f"Dimensiones referencia: {df_structured.shape}")
    print(f"Dimensiones sintéticas: {df_synth.shape}")
    print("Datos preparados exitosamente.")

if __name__ == '__main__':
    preparar_datos()
