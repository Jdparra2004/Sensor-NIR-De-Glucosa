import pandas as pd
import numpy as np
import os

def preparar_datos():
    raw_path = 'data/CalibrationData_NTNU.txt'
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)

    # Load data - the file is large, so read with caution
    # Based on initial inspection, it seems space-separated, but let's check
    # Columns 0-4: Concentrations, 5: Temp, 6: Kuvette, 7: Day, 8: Run
    # Columns 9+: Absorbances at wavelengths
    
    # Reading the first few lines to get the structure right for header
    # The header seems to be split, let's just assume standard reading
    
    # Since it's large, we might need a specific approach. 
    # Let's try loading it with pandas and see if it fits. 46MB is small enough for pandas.
    
    # Read the file. Assuming whitespace separation.
    print("Reading data...")
    df = pd.read_csv(raw_path, sep=r'\s+', header=None, skiprows=2, encoding='utf-16')
    
    # The header is actually 2 lines. 
    # Row 0: "Glucose (mM) ... Ethanol (mM)"
    # Row 1: "Temperature (C) ... Run"
    
    # Let's skip the header rows (2) and handle the data.
    # The wavelengths are NOT in the header row of the data, they seem to be listed in the second line of the file.
    
    # Let's just create a synthetic header for now based on the structure.
    # Since we can't easily read the wavelengths from the file header because it's split,
    # let's assume they are known or can be inferred.
    
    # The prompt asks to parse, so let's parse the wavelengths.
    with open(raw_path, 'r', encoding='utf-16') as f:
        lines = f.readlines()
        wavelength_line = lines[1]
        wavelengths = [float(w) for w in wavelength_line.split()]
        # The first few items are Temperature, Kuvette, Day, Run, so exclude those?
        # Wait, the header said: "Temperature (C) Kuvette Day Run" then the wavelengths started.
        wavelengths = wavelengths[4:] # Skip the 4 non-wavelength columns
        
    # Dynamically assign column names
    num_data_cols = df.shape[1] - 9
    df.columns = ['glucose_mM', 'lactate_mM', 'acetaminophen_mM', 'caffeine_mM', 'ethanol_mM', 
                  'temp_C', 'kuvette', 'day', 'run'] + wavelengths[:num_data_cols]
    
    # Handle duplicate columns if any
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        # Convert to string to avoid float + str error
        dup_str = str(dup)
        cols[cols[cols == dup].index.values.tolist()] = [dup_str + '_' + str(i) if i != 0 else dup_str for i in range(sum(cols == dup))]
    df.columns = cols

    # SNV Normalization
    print("Applying SNV...")
    abs_cols = df.columns[9:]
    df_abs = df[abs_cols]
    df_abs_snv = df_abs.sub(df_abs.mean(axis=1), axis=0).div(df_abs.std(axis=1), axis=0)
    df.loc[:, abs_cols] = df_abs_snv

    # Melt to long format: id_muestra, lambda_nm, absorbancia_medida, glucosa_referencia_mM, interferente_mM
    print("Melting data...")
    df['id_muestra'] = df.index
    
    # Interferente: combine lactate, acetaminophen, caffeine, ethanol (sum or max?)
    # Let's take the sum
    df['interferente_mM'] = df[['lactate_mM', 'acetaminophen_mM', 'caffeine_mM', 'ethanol_mM']].sum(axis=1)
    
    df_long = df.melt(id_vars=['id_muestra', 'glucose_mM', 'interferente_mM'], 
                      value_vars=abs_cols,
                      var_name='lambda_nm', 
                      value_name='absorbancia_medida')
    
    df_long.rename(columns={'glucose_mM': 'glucosa_referencia_mM'}, inplace=True)
    
    # Save to CSV
    print("Saving...")
    df_long.to_csv(os.path.join(processed_dir, 'muestras_referencia_nir.csv'), index=False)
    
    # Create Synthetic Data
    print("Creating synthetic data...")
    # 0.01 - 1.0 mM
    n_samples = 100
    ids = range(n_samples)
    glucosa = np.linspace(0.01, 1.0, n_samples)
    interferentes = np.random.uniform(0, 0.1, n_samples)
    
    # Simplified Beer-Lambert for control data
    # A = epsilon * c * l
    # Let's just create some dummy absorbances
    wls = np.array(wavelengths)
    data = []
    for i in range(n_samples):
        # Dummy spectra
        A = np.exp(-(wls - 800)**2 / (2 * 50**2)) * glucosa[i] + interferentes[i] * 0.1
        for j, wl in enumerate(wls):
            data.append([i, wl, A[j], glucosa[i], interferentes[i]])
            
    df_synthetic = pd.DataFrame(data, columns=['id_muestra', 'lambda_nm', 'absorbancia_medida', 'glucosa_referencia_mM', 'interferente_mM'])
    df_synthetic.to_csv(os.path.join(processed_dir, 'muestras_control_sinteticas.csv'), index=False)
    print("Done.")

if __name__ == '__main__':
    preparar_datos()
