"""
MÓDULO DE PRUEBAS UNITARIAS: test_sistema.py
PROYECTO: Evaluación paramétrica de detección óptica NIR de glucosa en sudor
"""

import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Asegurar importación del directorio raíz del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.modelo_optico import ModeloBeerLambertNIR, ModeloPLSRegresionNIR, LAMBDA_REFERENCIA_NM, DELTA_W
from core.modelo_microfluido import ModeloMicrofluido
from core.simulacion_parametrica import SimulacionParametrica
from sklearn.cross_decomposition import PLSRegression

class TestSensorNIR(unittest.TestCase):

    def setUp(self):
        """Configuración de entorno aislado antes de cada prueba."""
        self.test_dir = tempfile.mkdtemp()
        self.modelo_optico = ModeloBeerLambertNIR(longitud_optica_mm=1.0, incluir_desplazamiento_agua=True)
        self.modelo_plsr = ModeloPLSRegresionNIR()
        self.modelo_mf = ModeloMicrofluido(ancho_um=200.0, alto_um=50.0, largo_mm=5.0, caudal_nL_min=5.0)

    def tearDown(self):
        """Limpieza del entorno temporal tras cada prueba."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # --- NUEVAS PRUEBAS PARA PLS-R ---

    def test_plsr_channel_masking(self):
        """Verifica que el filtrado de canales sea correcto."""
        # Generar datos simulados con encabezados de longitud de onda
        cols = ["400.0", "600.0", "1100.0", "1600.0", "1900.0", "2400.0"]
        # Canales esperados: 600.0, 1600.0
        # 400 < 500 (descartar), 1100 (crossover 1090-1110 descartar), 1900 (agua 1800-2100 descartar), 2400 (> 2300 descartar)
        
        valid_cols = self.modelo_plsr.obtener_canales_validos(cols)
        self.assertEqual(valid_cols, ["600.0", "1600.0"])

    def test_plsr_training_and_prediction(self):
        """Verifica el entrenamiento y predicción del modelo PLSR."""
        # Crear datos de entrenamiento (X espectral, y concentración)
        X = pd.DataFrame(np.random.rand(20, 100), columns=[str(500.0 + i*10) for i in range(100)])
        y = pd.Series(np.random.rand(20))
        
        self.modelo_plsr.entrenar_calibracion(X, y, max_componentes=5)
        
        # Predecir sobre datos de prueba
        X_test = pd.DataFrame(np.random.rand(5, 100), columns=X.columns)
        predictions = self.modelo_plsr.predecir(X_test)
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(np.all(predictions >= 0))


    def test_absorbancia_computo(self):
        """Verifica que el cálculo de absorbancia retorne un valor numérico válido."""
        C_glucosa = 0.5
        absorbancia = self.modelo_optico.absorbancia(C_glucosa, LAMBDA_REFERENCIA_NM)
        self.assertIsNotNone(absorbancia)
        self.assertIsInstance(float(absorbancia), float)

    def test_efecto_desplazamiento_agua(self):
        """Verifica que la corrección por desplazamiento de agua modifique la absorbancia."""
        C_glucosa = 1.0
        A_con = self.modelo_optico.absorbancia(C_glucosa, LAMBDA_REFERENCIA_NM)
        modelo_sin = ModeloBeerLambertNIR(longitud_optica_mm=1.0, incluir_desplazamiento_agua=False)
        A_sin = modelo_sin.absorbancia(C_glucosa, LAMBDA_REFERENCIA_NM)
        self.assertNotEqual(A_con, A_sin)

    def test_barrido_concentraciones(self):
        """Prueba la respuesta vectorial ante un arreglo de concentraciones."""
        C_vec = np.array([0.1, 0.5, 1.0])
        A_vec = self.modelo_optico.barrido_concentraciones(C_vec, LAMBDA_REFERENCIA_NM)
        self.assertEqual(len(C_vec), len(A_vec))
        self.assertIsInstance(A_vec, np.ndarray)

    def test_concentracion_inversa_matematica(self):
        """Verifica que el motor de inferencia recupere la concentración original."""
        C_original = 0.15
        A_calculada = self.modelo_optico.absorbancia(C_original, LAMBDA_REFERENCIA_NM)
        C_inversa = self.modelo_optico.concentracion_inversa(A_calculada, LAMBDA_REFERENCIA_NM, alpha=1.0, beta=0.0)
        self.assertAlmostEqual(C_original, C_inversa, places=5)

    def test_diagnostico_clinico_tres_casos(self):
        """Evalúa la clasificación en los tres rangos fisiológicos."""
        diag_normal = self.modelo_optico.evaluar_clasificacion_fisiologica(0.10)
        self.assertIn("Normal", diag_normal)
        diag_alerta = self.modelo_optico.evaluar_clasificacion_fisiologica(0.30)
        self.assertIn("Alerta", diag_alerta)
        diag_alto = self.modelo_optico.evaluar_clasificacion_fisiologica(0.80)
        self.assertIn("Elevado", diag_alto)

    def test_diagnostico_valores_extremos(self):
        """Evalúa el manejo de entradas fuera del rango fisiológico analítico."""
        diag_indetectable = self.modelo_optico.evaluar_clasificacion_fisiologica(-0.05)
        self.assertIn("Indetectable", diag_indetectable)

    def test_regimen_laminar(self):
        """Verifica que el número de Reynolds cumpla Re < 1."""
        reynolds = self.modelo_mf.numero_reynolds()
        self.assertLess(reynolds, 1.0)
        self.assertGreater(reynolds, 0.0)
        self.assertTrue(self.modelo_mf.es_flujo_laminar())

    def test_resumen_parametros(self):
        """Verifica que el resumen retorne los parámetros hidrodinámicos."""
        params = self.modelo_mf.resumen_parametros()
        self.assertIsInstance(params, dict)
        self.assertIn("numero_reynolds", params)
        self.assertIn("tiempo_residencia_s", params)

    def test_simulacion_parametrica_ejecucion(self):
        """Verifica la generación de DataFrames en la simulación paramétrica."""
        sim = SimulacionParametrica()
        resultados = sim.ejecutar_todas()
        self.assertIsInstance(resultados, dict)
        self.assertTrue(len(resultados) > 0)
        primer_df = list(resultados.values())[0]
        self.assertIsInstance(primer_df, pd.DataFrame)
        self.assertFalse(primer_df.empty)

    def test_exportar_resultados(self):
        """Verifica la exportación correcta de archivos CSV y JSON."""
        sim = SimulacionParametrica()
        sim.ejecutar_todas()
        carpeta_export = os.path.join(self.test_dir, "export_test")
        sim.exportar_resultados(carpeta=carpeta_export)
        self.assertTrue(os.path.exists(carpeta_export))
        archivos = os.listdir(carpeta_export)
        self.assertTrue(len(archivos) > 0)

    def test_procesamiento_lote_csv(self):
        """Verifica el procesamiento de lotes por inferencia inversa sobre datos tabulares."""
        file_path = Path("data/processed/muestras_referencia_nir.csv")
        if file_path.exists():
            df = pd.read_csv(file_path, low_memory=False)
        else:
            df = pd.DataFrame({
                "glucosa_referencia_mM": [0.05, 0.15, 0.35, 0.70, 0.95],
                "absorbancia_medida": [self.modelo_optico.absorbancia(c, LAMBDA_REFERENCIA_NM) for c in [0.05, 0.15, 0.35, 0.70, 0.95]]
            })
        self.assertFalse(df.empty)
        sample = df.head(5).copy()
        col_abs = "absorbancia_medida" if "absorbancia_medida" in sample.columns else sample.columns[1]
        sample["c_est"] = sample[col_abs].apply(lambda a: self.modelo_optico.concentracion_inversa(float(a), LAMBDA_REFERENCIA_NM, alpha=1.0, beta=0.0))
        self.assertEqual(len(sample), 5)
        self.assertIn("c_est", sample.columns)


if __name__ == '__main__':
    unittest.main()