"""
=============================================================================
MÓDULO: simulacion_parametrica.py
PROYECTO: Evaluación paramétrica de detección óptica NIR de glucosa en sudor
=============================================================================

Descripción:
    Motor de simulación paramétrica que permite explorar el espacio de
    parámetros del modelo óptico y microfluídico de forma sistemática.

    Implementa:
    1. Barrido de concentraciones de glucosa (rango fisiológico)
    2. Barrido de longitudes de onda NIR
    3. Barrido de parámetros geométricos del microcanal
    4. Análisis de sensibilidad local (dA/dθ)
    5. Exportación de resultados a CSV

Referencias:
    - Amerov et al. (2004): rangos de absorptividad NIR glucosa
    - Gao et al. (2016): rango fisiológico glucosa en sudor (0.01–1.0 mM)
    - Yin et al. (2025): variabilidad fisiológica del sistema
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

from core.modelo_optico import ModeloBeerLambertNIR, LAMBDA_REFERENCIA_NM
from core.modelo_microfluido import ModeloMicrofluido


# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE SIMULACIÓN POR DEFECTO
# ---------------------------------------------------------------------------

CONFIGURACION_DEFAULT = {
    # Rango fisiológico de glucosa en sudor (Gao et al., 2016)
    "C_glucosa_min_mM": 0.01,
    "C_glucosa_max_mM": 1.0,
    "n_concentraciones": 100,

    # Rango espectral NIR de interés (Yang et al., 2025)
    "lambda_min_nm": 1000,
    "lambda_max_nm": 1700,
    "n_lambdas": 8,

    # Longitud óptica del microcanal [mm]
    "longitud_optica_mm": 1.0,

    # Parámetros microfluídicos por defecto
    "ancho_canal_um": 200.0,
    "alto_canal_um": 50.0,
    "largo_canal_mm": 5.0,
    "caudal_nL_min": 10.0,
}


# ---------------------------------------------------------------------------
# CLASE DE SIMULACIÓN
# ---------------------------------------------------------------------------

class SimulacionParametrica:
    """
    Motor de simulación paramétrica del sistema de detección óptica NIR.

    Permite ejecutar experimentos numéricos variando:
    - Concentración de glucosa
    - Longitud de onda NIR
    - Longitud del camino óptico
    - Parámetros del microcanal

    Todos los resultados se almacenan en DataFrames de Pandas para
    facilitar el análisis y la visualización posterior.

    Parámetros
    ----------
    config : dict, opcional
        Diccionario de configuración. Si no se provee, se usa la
        configuración por defecto.
    """

    def __init__(self, config: dict = None):
        self.config = config or CONFIGURACION_DEFAULT.copy()
        self.resultados = {}

    # =========================================================================
    # SIMULACIÓN 1: Absorbancia vs Concentración (a λ fija)
    # =========================================================================

    def sim_absorbancia_vs_concentracion(
            self,
            longitud_optica_mm: float = None,
            lambda_nm: float = LAMBDA_REFERENCIA_NM,
            incluir_desplazamiento_agua: bool = True
    ) -> pd.DataFrame:
        """
        Simula A(C) a longitud de onda fija.

        Objetivo: verificar linealidad A-C (Beer-Lambert) y evaluar
        sensibilidad ante cambios de concentración en rango fisiológico.

        Retorna
        -------
        pd.DataFrame con columnas: C_mM, A_con_correccion, A_sin_correccion
        """
        L = longitud_optica_mm or self.config["longitud_optica_mm"]

        C_vec = np.linspace(
            self.config["C_glucosa_min_mM"],
            self.config["C_glucosa_max_mM"],
            self.config["n_concentraciones"]
        )

        modelo_con = ModeloBeerLambertNIR(L, incluir_desplazamiento_agua=True)
        modelo_sin = ModeloBeerLambertNIR(L, incluir_desplazamiento_agua=False)

        A_con = modelo_con.barrido_concentraciones(C_vec, lambda_nm)
        A_sin = modelo_sin.barrido_concentraciones(C_vec, lambda_nm)

        df = pd.DataFrame({
            "C_mM": C_vec,
            "A_con_correccion_agua": A_con,
            "A_sin_correccion_agua": A_sin,
            "diferencia_pct": 100 * (A_sin - A_con) / np.where(A_sin != 0, A_sin, 1e-15),
        })

        self.resultados["absorbancia_vs_concentracion"] = df
        return df

    # =========================================================================
    # SIMULACIÓN 2: Espectro NIR (A vs λ) a concentraciones múltiples
    # =========================================================================

    def sim_espectro_nir(
            self,
            concentraciones_mM: list = None,
            longitud_optica_mm: float = None
    ) -> pd.DataFrame:
        """
        Simula el espectro NIR para múltiples concentraciones de glucosa.

        Permite visualizar las bandas de absorción características y
        comparar la señal en función de la concentración.

        Retorna
        -------
        pd.DataFrame con columnas: lambda_nm, A_C=0.01, A_C=0.1, ...
        """
        L = longitud_optica_mm or self.config["longitud_optica_mm"]
        modelo = ModeloBeerLambertNIR(L, incluir_desplazamiento_agua=True)

        if concentraciones_mM is None:
            concentraciones_mM = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

        lambdas = modelo.longitudes_onda
        data = {"lambda_nm": lambdas}

        for C in concentraciones_mM:
            _, A = modelo.espectro_completo(C)
            data[f"A_C{C:.3f}mM"] = A

        df = pd.DataFrame(data)
        self.resultados["espectro_nir"] = df
        return df

    # =========================================================================
    # SIMULACIÓN 3: Sensibilidad paramétrica (dA/dθ)
    # =========================================================================

    def sim_sensibilidad_longitud_optica(
            self,
            longitudes_opticas_mm: np.ndarray = None,
            lambda_nm: float = LAMBDA_REFERENCIA_NM,
            C_ref_mM: float = 0.5
    ) -> pd.DataFrame:
        """
        Analiza cómo varía la sensibilidad del modelo (dA/dC) con la
        longitud del camino óptico.

        Un camino óptico mayor → mayor señal, pero también mayor absorción
        de agua → relación no lineal que este análisis revela.

        Retorna
        -------
        pd.DataFrame con columnas: L_mm, A_en_Cref, sensibilidad_dA_dC
        """
        if longitudes_opticas_mm is None:
            longitudes_opticas_mm = np.linspace(0.1, 10.0, 50)

        filas = []
        for L in longitudes_opticas_mm:
            modelo = ModeloBeerLambertNIR(L, incluir_desplazamiento_agua=True)
            A = modelo.absorbancia(C_ref_mM, lambda_nm)
            s = modelo.sensibilidad(lambda_nm)
            filas.append({
                "L_optica_mm": L,
                "A_en_Cref": A,
                "sensibilidad_dA_dC": s,
            })

        df = pd.DataFrame(filas)
        self.resultados["sensibilidad_longitud_optica"] = df
        return df

    # =========================================================================
    # SIMULACIÓN 4: Parámetros microfluídicos
    # =========================================================================

    def sim_parametros_microfluido(
            self,
            anchos_um: np.ndarray = None,
            caudales_nL_min: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Análisis paramétrico del modelo microfluídico conceptual.

        Evalúa cómo cambian Re, Pe, tiempo de residencia y concentración
        efectiva en función de parámetros geométricos y de flujo.

        Retorna
        -------
        pd.DataFrame con métricas del flujo para cada combinación.
        """
        if anchos_um is None:
            anchos_um = np.array([100, 150, 200, 300, 500])
        if caudales_nL_min is None:
            caudales_nL_min = np.array([1, 5, 10, 20, 50])

        filas = []
        for ancho in anchos_um:
            for Q in caudales_nL_min:
                modelo_mf = ModeloMicrofluido(
                    ancho_um=ancho,
                    alto_um=self.config["alto_canal_um"],
                    largo_mm=self.config["largo_canal_mm"],
                    caudal_nL_min=Q
                )
                params = modelo_mf.resumen_parametros()
                # Concentración efectiva para C_entrada = 0.5 mM (referencia)
                C_ef = modelo_mf.concentracion_efectiva_zona_optica(0.5)
                params["C_efectiva_mM_para_C0_0.5mM"] = C_ef
                filas.append(params)

        df = pd.DataFrame(filas)
        self.resultados["parametros_microfluido"] = df
        return df

    # =========================================================================
    # SIMULACIÓN 5: Barrido completo λ × C
    # =========================================================================

    def sim_mapa_absorbancia(
            self,
            longitud_optica_mm: float = None
    ) -> pd.DataFrame:
        """
        Genera un mapa 2D de A(λ, C) — malla de longitudes de onda y
        concentraciones.

        Útil para identificar ventanas espectrales óptimas para la
        detección según rango de concentración fisiológico.

        Retorna
        -------
        pd.DataFrame en formato long: lambda_nm, C_mM, absorbancia
        """
        L = longitud_optica_mm or self.config["longitud_optica_mm"]
        modelo = ModeloBeerLambertNIR(L, incluir_desplazamiento_agua=True)

        lambdas = modelo.longitudes_onda
        C_vec = np.linspace(
            self.config["C_glucosa_min_mM"],
            self.config["C_glucosa_max_mM"],
            20  # resolución reducida para mapa 2D
        )

        filas = []
        for lam in lambdas:
            for C in C_vec:
                A = modelo.absorbancia(C, lam)
                filas.append({
                    "lambda_nm": lam,
                    "C_mM": round(C, 4),
                    "absorbancia": A
                })

        df = pd.DataFrame(filas)
        self.resultados["mapa_absorbancia"] = df
        return df

    # =========================================================================
    # UTILIDADES
    # =========================================================================

    def exportar_resultados(self, carpeta: str = "outputs"):
        """Exporta todos los DataFrames de resultados a archivos CSV."""
        ruta = Path(carpeta)
        ruta.mkdir(exist_ok=True)

        for nombre, df in self.resultados.items():
            archivo = ruta / f"{nombre}.csv"
            df.to_csv(archivo, index=False)
            print(f"  ✓ Exportado: {archivo}")

        # También guardar la configuración usada
        cfg_path = ruta / "configuracion_simulacion.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Configuración: {cfg_path}")

    def ejecutar_todas(self) -> dict:
        """
        Ejecuta todas las simulaciones disponibles y retorna
        el diccionario de resultados.
        """
        print("Iniciando simulaciones paramétricas...\n")

        print("  [1/5] Absorbancia vs Concentración...")
        self.sim_absorbancia_vs_concentracion()

        print("  [2/5] Espectro NIR a múltiples concentraciones...")
        self.sim_espectro_nir()

        print("  [3/5] Sensibilidad vs Longitud Óptica...")
        self.sim_sensibilidad_longitud_optica()

        print("  [4/5] Parámetros Microfluídicos...")
        self.sim_parametros_microfluido()

        print("  [5/5] Mapa de Absorbancia λ × C...")
        self.sim_mapa_absorbancia()

        print(f"\n✓ {len(self.resultados)} simulaciones completadas.\n")
        return self.resultados
