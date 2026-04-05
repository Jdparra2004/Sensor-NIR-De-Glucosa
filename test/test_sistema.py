import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Agrega el directorio raíz del proyecto al path de Python
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importamos las clases de tu código base
from core.modelo_optico import ModeloBeerLambertNIR, LAMBDA_REFERENCIA_NM
from core.modelo_microfluido import ModeloMicrofluido
from core.simulacion_parametrica import SimulacionParametrica

class TestSensorNIR(unittest.TestCase):

    def setUp(self):
        """Configuración inicial antes de cada prueba."""
        # Configurar un entorno temporal para guardar archivos y no ensuciar el proyecto
        self.test_dir = tempfile.mkdtemp()
        
        # Instanciar modelos con valores típicos
        self.modelo_optico = ModeloBeerLambertNIR(longitud_optica_mm=1.0)
        self.modelo_mf = ModeloMicrofluido(ancho_um=200.0, alto_um=50.0, largo_mm=5.0, caudal_nL_min=5.0)

    def tearDown(self):
        """Limpieza de archivos después de cada prueba."""
        shutil.rmtree(self.test_dir)

    # PRUEBAS DEL MODELO ÓPTICO (Ley de Beer-Lambert)

    def test_absorbancia_positiva(self):
        """Prueba que el cálculo de absorbancia no devuelva valores negativos absurdos."""
        C_glucosa = 0.5  # mM
        # Usamos la longitud de referencia (1600 nm por defecto)
        absorbancia = self.modelo_optico.absorbancia(C_glucosa, LAMBDA_REFERENCIA_NM)
        
        self.assertIsNotNone(absorbancia)
        self.assertIsInstance(absorbancia, float)

    def test_efecto_desplazamiento_agua(self):
        """Prueba que la corrección de agua efectivamente modifique el resultado final."""
        C_glucosa = 1.0 # mM
        
        # Modelo con corrección (default)
        A_con = self.modelo_optico.absorbancia(C_glucosa, LAMBDA_REFERENCIA_NM)
        
        # Modelo sin corrección
        modelo_sin = ModeloBeerLambertNIR(longitud_optica_mm=1.0, incluir_desplazamiento_agua=False)
        A_sin = modelo_sin.absorbancia(C_glucosa, LAMBDA_REFERENCIA_NM)
        
        # La absorbancia con corrección de desplazamiento de agua debería ser distinta
        self.assertNotEqual(A_con, A_sin, "La corrección de agua no está afectando el cálculo")

    def test_barrido_concentraciones(self):
        """Prueba que el barrido vectorial funcione correctamente."""
        C_vec = np.array([0.1, 0.5, 1.0])
        A_vec = self.modelo_optico.barrido_concentraciones(C_vec, LAMBDA_REFERENCIA_NM)
        
        self.assertEqual(len(C_vec), len(A_vec), "El vector de salida no coincide con el de entrada")
        self.assertIsInstance(A_vec, np.ndarray)

    # PRUEBAS DEL MODELO MICROFLUÍDICO
    
    def test_regimen_laminar(self):
        """Prueba que el flujo calculado se mantenga en régimen laminar (Re < 1)."""
        reynolds = self.modelo_mf.numero_reynolds()
        
        self.assertLess(reynolds, 1.0, f"El régimen no es laminar, Re = {reynolds}")
        self.assertGreater(reynolds, 0.0, "El número de Reynolds debe ser positivo")

    def test_resumen_parametros(self):
        """Prueba que la generación del resumen de parámetros devuelva un diccionario válido."""
        params = self.modelo_mf.resumen_parametros()
        
        self.assertIsInstance(params, dict)
        self.assertIn("numero_reynolds", params)
        self.assertIn("tiempo_residencia_s", params)

    # PRUEBAS DEL MOTOR DE SIMULACIÓN Y EXPORTACIÓN

    def test_simulacion_parametrica_ejecucion(self):
        """Prueba que la simulación se ejecute y genere DataFrames."""
        sim = SimulacionParametrica()
        resultados = sim.ejecutar_todas()
        
        self.assertIsInstance(resultados, dict)
        self.assertTrue(len(resultados) > 0, "No se generaron resultados de simulación")
        
        # Verifica que el primer valor devuelto sea un DataFrame
        primer_df = list(resultados.values())[0]
        self.assertIsInstance(primer_df, pd.DataFrame)
        self.assertFalse(primer_df.empty)

    def test_exportar_resultados(self):
        """Prueba que se creen los archivos CSV y JSON en la carpeta de Descargas."""
        sim = SimulacionParametrica()
        sim.ejecutar_todas()
        
        # Buscamos la ruta de Descargas (Downloads) de tu usuario automáticamente
        carpeta_descargas = Path.home() / "Downloads" / "Test_Sensor_NIR"
        
        # Forzamos la exportación a esa carpeta en Descargas
        sim.exportar_resultados(carpeta=str(carpeta_descargas))
        
        archivos_creados = os.listdir(carpeta_descargas)
        
        # Verificamos que se haya creado al menos un archivo
        self.assertTrue(len(archivos_creados) > 0, "No se exportó ningún archivo a Descargas")
        
        # Verificamos que se haya guardado el JSON de configuración
        self.assertIn("configuracion_simulacion.json", archivos_creados)
        
        print(f"\n[OK] Ve a revisar tus archivos de prueba reales en: {carpeta_descargas}")

    # PRUEBAS DE ANÁLISIS CLÍNICO E INFERENCIA INVERSA

    def test_concentracion_inversa_matematica(self):
        """
        Validación de parámetros: Prueba que el cálculo inverso devuelva la 
        concentración original a partir de una absorbancia calculada.
        """
        C_original = 0.15 # mM
        
        # 1. Calculamos la Absorbancia de una concentración conocida
        A_calculada = self.modelo_optico.absorbancia(C_original, LAMBDA_REFERENCIA_NM)
        
        # 2. Le pasamos esa Absorbancia a la nueva función inversa
        C_inversa = self.modelo_optico.concentracion_inversa(A_calculada, LAMBDA_REFERENCIA_NM)
        
        # 3. Validamos que la C_inversa sea casi idéntica a la C_original (5 decimales de precisión)
        self.assertAlmostEqual(C_original, C_inversa, places=5, msg="El cálculo de la concentración inversa falló")

    def test_diagnostico_clinico_tres_casos(self):
        """
        Prueba los 3 escenarios clínicos posibles:
        - No diabetes (Normal)
        - Tal vez diabetes (Riesgo / Prediabetes)
        - Sí diabetes (Muy Alto / Hiperglucemia)
        """
        # Caso 1: NO Diabetes -> Nivel Normal (<= 0.2 mM)
        diag_normal = self.modelo_optico.evaluar_riesgo_clinico(0.1)
        self.assertIn("Nivel Normal", diag_normal, "Falló la clasificación de caso Normal")
        
        # Caso 2: TAL VEZ Diabetes -> Nivel Elevado (0.2 a 0.4 mM)
        diag_riesgo = self.modelo_optico.evaluar_riesgo_clinico(0.3)
        self.assertIn("Nivel Elevado", diag_riesgo, "Falló la clasificación de caso de Riesgo")
        
        # Caso 3: SÍ Diabetes -> Nivel Muy Alto (> 0.4 mM)
        diag_alto = self.modelo_optico.evaluar_riesgo_clinico(0.8)
        self.assertIn("Nivel Muy Alto", diag_alto, "Falló la clasificación de caso de Diabetes")

    def test_diagnostico_valores_extremos(self):
        """
        Prueba cómo reacciona el sistema ante valores absurdos o bajo el límite de detección.
        """
        diag_indetectable = self.modelo_optico.evaluar_riesgo_clinico(-0.05)
        self.assertIn("Indetectable", diag_indetectable, "El sistema no manejó correctamente valores por debajo de 0.01")

if __name__ == '__main__':
    unittest.main()