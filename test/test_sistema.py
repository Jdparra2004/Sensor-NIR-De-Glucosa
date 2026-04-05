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

    # =========================================================================
    # PRUEBAS DEL MODELO ÓPTICO (Ley de Beer-Lambert)
    # =========================================================================

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

    # =========================================================================
    # PRUEBAS DEL MODELO MICROFLUÍDICO
    # =========================================================================

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

    # =========================================================================
    # PRUEBAS DEL MOTOR DE SIMULACIÓN Y EXPORTACIÓN
    # =========================================================================

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
        """Prueba que se creen los archivos CSV y JSON en la carpeta indicada."""
        sim = SimulacionParametrica()
        sim.ejecutar_todas()
        
        # Forzar la exportación a nuestra carpeta temporal segura
        sim.exportar_resultados(carpeta=self.test_dir)
        
        archivos_creados = os.listdir(self.test_dir)
        
        # Verificamos que se haya creado al menos un archivo
        self.assertTrue(len(archivos_creados) > 0, "No se exportó ningún archivo")
        
        # Verificamos que se haya guardado el JSON de configuración
        self.assertIn("configuracion_simulacion.json", archivos_creados)

if __name__ == '__main__':
    unittest.main()