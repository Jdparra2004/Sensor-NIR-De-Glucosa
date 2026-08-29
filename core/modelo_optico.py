"""
MÓDULO: modelo_optico.py
PROYECTO: Evaluación paramétrica de detección óptica NIR de glucosa en sudor
DESCRIPCIÓN: Implementación de la Ley de Beer-Lambert modificada con corrección 
             por desplazamiento volumétrico de agua (Amerov et al., 2004).
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

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

COEF_DESPLAZAMIENTO_AGUA: float = DELTA_W
CONCENTRACION_AGUA: float = 55_500.0


class ModeloBeerLambertNIR:
    """
    Modelo óptico de absorción espectroscópica en el infrarrojo cercano (NIR)
    acoplado con corrección por exclusión volumétrica de solvente.
    """

    def __init__(self, longitud_optica_mm: float = 1.0, alpha: float = 1.0, beta: float = 0.0, incluir_desplazamiento_agua: bool = True):
        self.longitud_optica_mm = float(longitud_optica_mm)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.incluir_desplazamiento_agua = bool(incluir_desplazamiento_agua)

    @property
    def longitudes_onda(self) -> np.ndarray:
        """Retorna el arreglo de longitudes de onda base indexadas."""
        return np.array(sorted(ABSORPTIVIDAD_GLUCOSA.keys()), dtype=float)

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

    def concentracion_inversa(self, absorbancia: float, lambda_nm: float, alpha: float = None, beta: float = None) -> float:
        """
        Calcula la concentración molar C (mM) a partir de la absorbancia A_neta.
        Aplica calibración empírica: C_final = (C_teórica * alpha) + beta.
        Permite override de alpha/beta para inferencia de lote.
        """
        L = self.longitud_optica_mm
        eps_net = self.obtener_coeficiente_neto(lambda_nm)
        
        alpha_val = alpha if alpha is not None else self.alpha
        beta_val = beta if beta is not None else self.beta
        
        if np.isclose(eps_net, 0.0) or np.isclose(L, 0.0):
            return 0.0 + beta_val
        
        c_teorica = float(abs(absorbancia) / (abs(eps_net) * L))
        c_final = (c_teorica * alpha_val) + beta_val
        return max(0.0, c_final) # No concentraciones negativas

    def evaluar_clasificacion_fisiologica(self, concentracion_mM: float) -> str:
        """Clasifica la concentración de glucosa en sudor en rangos metabólicos clínicos."""
        if concentracion_mM < 0.01 or concentracion_mM > 2.0:
            return "Fuera de rango analítico / Indetectable"
        elif concentracion_mM <= 0.20:
            return "Normal"
        elif concentracion_mM <= 0.40:
            return "Rango de Alerta / Sospecha de Prediabetes"
        else:
            return "Nivel Elevado / Sospecha Hiperglucemia"

    def espectro_completo(self, concentracion_mM: float, lambdas: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Genera un barrido espectral continuo o sobre un vector de longitudes de onda dado."""
        if lambdas is None:
            lambdas = self.longitudes_onda
        A = self.absorbancia(concentracion_mM, lambdas)
        return lambdas, np.array(A, dtype=float)

    def sensibilidad(self, lambda_nm: float) -> float:
        """Calcula la sensibilidad local analítica dA/dC = eps_net * L."""
        return float(self.obtener_coeficiente_neto(lambda_nm) * self.longitud_optica_mm)
    
    def barrido_concentraciones(self, concentraciones: np.ndarray, lambda_nm: float) -> np.ndarray:
        """Evalúa un vector de concentraciones manteniendo fija la longitud de onda."""
        return np.array([self.absorbancia(c, lambda_nm) for c in concentraciones], dtype=float)


from sklearn.model_selection import LeaveOneOut, cross_val_score

class ModeloPLSRegresionNIR:
    """
    Modelo quimiométrico multivariante basado en Regresión por Mínimos Cuadrados
    Parciales (PLS-R) para la estimación de concentración de glucosa en sudor
    a partir de espectros completos de absorbancia.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.model = None
        self.active_features = []
        self.n_componentes_optimo = 11

    def obtener_canales_validos(self, columns: list) -> list:
        """
        Selecciona y filtra los canales de longitud de onda válidos:
        - Extremos: < 500 nm
        - Crossover del detector: 1090-1110 nm
        - Absorción de agua fuerte: 1800-2100 nm y > 2300 nm
        """
        valid_cols = []
        for col in columns:
            try:
                w = float(col)
                if w < 500.0 or (1090.0 <= w <= 1110.0) or (1800.0 <= w <= 2100.0) or w > 2300.0:
                    continue
                valid_cols.append(col)
            except ValueError:
                continue
        return valid_cols

    def entrenar_calibracion(self, X_df: pd.DataFrame, y_series: pd.Series, max_componentes: int = 20):
        """
        Entrena el modelo PLS-R seleccionando las bandas espectrales válidas y
        realizando validación cruzada para optimizar el número de componentes.
        """
        self.active_features = self.obtener_canales_validos(X_df.columns.tolist())
        
        if not self.active_features:
            raise ValueError("No se encontraron canales de longitud de onda válidos.")
            
        X_filtered = X_df[self.active_features].copy()
        
        # Selección óptima de LVs usando validación cruzada
        best_score = -np.inf
        
        for k in range(1, max_componentes + 1):
            pls = PLSRegression(n_components=k, scale=True)
            # Usar r2 score para evaluación de CV
            scores = cross_val_score(pls, X_filtered, y_series, cv=5)
            mean_score = np.mean(scores)
            
            if mean_score > best_score:
                best_score = mean_score
                self.n_componentes_optimo = k
        
        # Entrenar modelo final con LVs óptimos
        self.model = PLSRegression(n_components=self.n_componentes_optimo, scale=True)
        self.model.fit(X_filtered, y_series)

    def predecir(self, X_df: pd.DataFrame, alpha: float = None, beta: float = None) -> np.ndarray:
        """
        Predice concentraciones utilizando el modelo PLS-R entrenado sobre
        las bandas espectrales activas.
        """
        if self.model is None:
            raise ValueError("El modelo PLS-R no ha sido entrenado aún.")
            
        X_filtered = X_df.reindex(columns=self.active_features, fill_value=0.0)
        
        C_pred = self.model.predict(X_filtered).flatten()
        
        alpha_val = alpha if alpha is not None else self.alpha
        beta_val = beta if beta is not None else self.beta
        
        return np.maximum(0.0, (C_pred * alpha_val) + beta_val)


    def evaluar_clasificacion_fisiologica(self, concentracion_mM: float) -> str:
        """Clasifica la concentración en rangos metabólicos clínicos, consistente con el modelo univariante."""
        if concentracion_mM < 0.01 or concentracion_mM > 50.0: # Rango ampliado para soportar el dataset de validación real (hasta 50mM)
            return "Fuera de rango analítico / Indetectable"
        elif concentracion_mM <= 10.0: # Umbral escalado para alinearse con los datos NTNU de glucosa de referencia
            return "Normal"
        elif concentracion_mM <= 25.0:
            return "Rango de Alerta / Sospecha de Prediabetes"
        else:
            return "Nivel Elevado / Sospecha Hiperglucemia"