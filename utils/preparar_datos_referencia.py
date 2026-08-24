import pandas as pd
import numpy as np
import os

def preparar_datos():
    """
    Genera el archivo `data/processed/muestras_referencia_nir.csv`
    en formato tabular estructurado tipo matriz/resumen por muestra,
    100% compatible con Excel.
    """
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, 'muestras_referencia_nir.csv')

    # Bandas críticas solicitadas
    wavelengths = [1000, 1100, 1200, 1300, 1400, 1450, 1550, 1600, 1650, 1700]
    
    # Simulación de datos estructurados (500 muestras)
    n_samples = 500
    ids = range(1, n_samples + 1)
    
    # Generar valores fisiológicos realistas
    data = {
        'id_muestra': ids,
        'glucosa_referencia_mM': np.random.uniform(0.01, 1.0, n_samples),
        'lactato_mM': np.random.uniform(0.5, 5.0, n_samples),
        'temperatura_C': np.random.uniform(30.0, 37.0, n_samples),
    }
    
    # Crear un DataFrame base
    df = pd.DataFrame(data)
    
    # Generar absorbancias sintéticas para cada longitud de onda
    for wl in wavelengths:
        # A = ε * C + ruido
        # ε simplificado para propósitos de demostración
        epsilon_g = 5e-5 # coeficiente ficticio
        A = epsilon_g * df['glucosa_referencia_mM'] + np.random.normal(0, 0.001, n_samples)
        df[f'absorbancia_{wl}nm'] = A
        
    # Reordenar: id, glucosa, lactato, temp, + las absorbancias en orden
    cols = ['id_muestra', 'glucosa_referencia_mM', 'lactato_mM', 'temperatura_C'] + [f'absorbancia_{wl}nm' for wl in wavelengths]
    df = df[cols]
    
    # Guardar CSV ligero
    df.to_csv(output_path, index=False)
    print(f"Archivo generado exitosamente en: {output_path}")
    print(f"Dimensiones: {df.shape}")

if __name__ == '__main__':
    preparar_datos()
