"""
MÓDULO: modelo_optico.py
PROYECTO: Evaluación paramétrica de detección óptica NIR de glucosa en sudor
DESCRIPCIÓN: Implementación de la Ley de Beer-Lambert modificada con corrección 
             por desplazamiento volumétrico de agua (Amerov et al., 2004).
"""

from typing import Tuple, Union
import numpy as np

# ==============================================================================
# CONSTANTES FÍSICAS Y ESPECTRALES (Amerov et al., 2004; Yang et al., 2025)
# ==============================================================================
LAMBDA_REFERENCIA_NM: float = 1650.0  # Longitud de onda central en ventana óptima (1600-1700 nm)
DELTA_W: float = 6.15                 # Coeficiente de desplazamiento volumétrico de agua (adimensional)
RANGO_LAMBDA_NM: Tuple[float, float] = (1000.0, 1700.0)
RANGO_CONCENTRACION_MM: Tuple[float, float] = (0.01, 1.0)

# Coeficientes de absortividad molar de Glucosa (mM^-1 * mm^-1)
ABSORPTIVIDAD_GLUCOSA = {
    1000: 1.2e-5,
    1100: 2.1e-5,
    1200: 3.5e-5,
    1300: 4.8e-5,
    1400: 3.2e-5,
    1450: 2.5e-5,
    1550: 5.0e-5,
    1600: 6.2e-5,
    1650: 6.8e-5,
    1700: 5.5e-5,
}

# Coeficientes de absortividad de Agua (mm^-1)
ABSORPTIVIDAD_AGUA = {
    1000: 0.0036,
    1100: 0.0048,
    1200: 0.0200,
    1300: 0.0100,
    1400: 0.3500,
    1450: 0.4200,
    1550: 0.0800,
    1600: 0.0600,
    1650: 0.0700,
    1700: 0.1200,
}

# Aliases de compatibilidad
COEF_DESPLAZAMIENTO_AGUA: float = DELTA_W
CONCENTRACION_AGUA: float = 55_500.0


class ModeloBeerLambertNIR:
    """
    Modelo óptico de absorción espectroscópica en el infrarrojo cercano (NIR)
    acoplado con corrección por exclusión volumétrica de solvente.
    """

    def __init__(self, longitud_optica_mm: float = 1.0, incluir_desplazamiento_agua: bool = True):
        self.longitud_optica_mm = float(longitud_optica_mm)
        self.incluir_desplazamiento_agua = bool(incluir_desplazamiento_agua)

    def _interpolar(self, tabla: dict, lam: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Interpola linealmente los coeficientes espectrales."""
        lambdas = np.array(sorted(tabla.keys()), dtype=float)
        valores = np.array([tabla[k] for k in lambdas], dtype=float)
        return np.interp(lam, lambdas, valores)

    def obtener_coeficiente_neto(self, lambda_nm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calcula el coeficiente de absortividad neto: eps_net = eps_g - eps_w * delta_w."""
        eps_g = self._interpolar(ABSORPTIVIDAD_GLUCOSA, lambda_nm)
        if self.incluir_desplazamiento_agua:
            eps_w = self._interpolar(ABSORPTIVIDAD_AGUA, lambda_nm)
            return eps_g - (eps_w * DELTA_W)
        return eps_g

    def absorbancia(self, concentracion_mM: Union[float, np.ndarray], lambda_nm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calcula la absorbancia neta según la ley de Beer-Lambert modificada:
        A_neta = (eps_g - eps_w * delta_w) * C * L
        """
        eps_net = self.obtener_coeficiente_neto(lambda_nm)
        return eps_net * concentracion_mM * self.longitud_optica_mm

    def concentracion_inversa(self, absorbancia: float, lambda_nm: float) -> float:
        """
        Calcula la concentración molar C (mM) a partir de la absorbancia A_neta.
        Maneja magnitudes absolutas para preservar la robustez analítica ante absorbancias negativas.
        """
        L = self.longitud_optica_mm
        eps_net = self.obtener_coeficiente_neto(lambda_nm)
        
        if np.isclose(eps_net, 0.0) or np.isclose(L, 0.0):
            return 0.0
        
        return float(abs(absorbancia) / (abs(eps_net) * L))

    def evaluar_clasificacion_fisiologica(self, concentracion_mM: float) -> str:
        """
        Clasifica la concentración de glucosa en sudor en rangos metabólicos clínicos.
        """
        if concentracion_mM < 0.01 or concentracion_mM > 2.0:
            return "Fuera de rango analítico / Indetectable"
        elif concentracion_mM <= 0.20:
            return "Normal"
        elif concentracion_mM <= 0.40:
            return "Rango de Alerta / Sospecha de Prediabetes"
        else:
            return "Nivel Elevado / Sospecha Hiperglucemia"

    def espectro_completo(self, concentracion_mM: float, puntos: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Genera un barrido espectral continuo entre 1000 y 1700 nm."""
        lambdas = np.linspace(RANGO_LAMBDA_NM[0], RANGO_LAMBDA_NM[1], puntos)
        A = self.absorbancia(concentracion_mM, lambdas)
        return lambdas, A

    def sensibilidad(self, lambda_nm: float) -> float:
        """
        Calcula la sensibilidad local analítica dA/dC = eps_net * L.
        """
        return float(self.obtener_coeficiente_neto(lambda_nm) * self.longitud_optica_mm)
    
    def barrido_concentraciones(self, concentraciones: np.ndarray, lambda_nm: float) -> np.ndarray:
        return np.array([self.absorbancia(c, lambda_nm) for c in concentraciones], dtype=float)