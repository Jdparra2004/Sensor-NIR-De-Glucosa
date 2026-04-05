"""
MÓDULO: modelo_optico.py
PROYECTO: Evaluación paramétrica de detección óptica NIR de glucosa en sudor
PROGRAMA: Bioingeniería - Trabajo de Grado

Descripción:
    Implementa el modelo matemático de detección óptica basado en la
    Ley de Beer-Lambert adaptada a medios biológicos acuosos diluidos.

Principio físico:
    A = ε_glucosa(λ) · C · L  +  ε_agua(λ) · C_agua · L
    donde el término de desplazamiento de agua (water displacement) modifica
    la absorción neta del sistema (Amerov et al., 2004).

Rangos fisiológicos (literatura):
    - Glucosa en sudor: 0.01 – 1.0 mM  (Gao et al., 2016; Yin et al., 2025)
    - Longitudes de onda NIR: 1000 – 1700 nm  (Yang et al., 2025)

Referencias:
    - Amerov, A.K., Chen, J., Arnold, M.A. (2004). Applied Spectroscopy, 58(10).
    - Heise, H.M. et al. (2021). Biosensors, 11(3).
    - Yang, M. et al. (2025). Advanced Sensor Research, 4(3).
"""

import numpy as np

# PARÁMETROS DEL MODELO (valores de referencia desde literatura)

# Coeficientes molares de absorción de la glucosa en NIR [mM^-1 · mm^-1]
# Fuente: Amerov et al. (2004) - primer sobretono y región de combinación
ABSORPTIVIDAD_GLUCOSA = {
    # λ (nm): ε_glucosa [mM^-1·mm^-1]
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

# Coeficientes de absorción del agua en NIR [mm^-1]
# El agua es el principal interferente óptico (Heise et al., 2021)
ABSORPTIVIDAD_AGUA = {
    1000: 0.0036,
    1100: 0.0048,
    1200: 0.0200,
    1300: 0.0100,
    1400: 0.3500,   # banda de absorción fuerte del agua
    1450: 0.4200,
    1550: 0.0800,
    1600: 0.0600,
    1650: 0.0700,
    1700: 0.1200,
}

# Coeficiente de desplazamiento de agua por glucosa [-]
# Fuente: Amerov et al. (2004) — cuando 1 mol glucosa se disuelve,
# desplaza ~6.15 moles de agua (He & Zhu, 2011).
# A concentraciones de sudor (0.01–1.0 mM) el efecto de desplazamiento
# es pequeño comparado con la absorción directa de glucosa.
# Factor de escala para concentraciones en mM (no en M):
COEF_DESPLAZAMIENTO_AGUA = 6.15 / 55500.0  # mol agua desplazada / mol glucosa, normalizado

# Concentración del agua pura [mM]
CONCENTRACION_AGUA = 55_500.0  # ~55.5 M = 55500 mM

# Longitud de onda de referencia del estudio (Yang et al., 2025)
LAMBDA_REFERENCIA_NM = 1600

# CLASE PRINCIPAL DEL MODELO

class ModeloBeerLambertNIR:
    """
    Modelo de absorbancia óptica NIR para detección de glucosa en sudor.

    Implementa la Ley de Beer-Lambert adaptada considerando:
    1. Absorción directa de la glucosa.
    2. Absorción del agua como interferente dominante.
    3. Corrección por desplazamiento de agua (water displacement).

    Ecuación central:
        A_neta(λ, C) = ε_g(λ) · C · L
                        - ε_w(λ) · δ_w · C · L
        donde δ_w es el coeficiente de desplazamiento volumétrico.

    Parámetros
    longitud_optica_mm : float
        Longitud del camino óptico efectivo [mm]. Representa la
        profundidad de interacción de la radiación NIR con el fluido.
        Típico en microfluídica: 0.1 – 10 mm.
    incluir_desplazamiento_agua : bool
        Si True, aplica corrección por desplazamiento de agua
        (modelo más realista según Amerov et al., 2004).
    """

    def __init__(self, longitud_optica_mm: float = 1.0,
                incluir_desplazamiento_agua: bool = True):
        self.longitud_optica_mm = longitud_optica_mm
        self.incluir_desplazamiento_agua = incluir_desplazamiento_agua
        self._longitudes_onda = sorted(ABSORPTIVIDAD_GLUCOSA.keys())

    # Propiedades 

    @property
    def longitudes_onda(self) -> np.ndarray:
        """Array de longitudes de onda disponibles [nm]."""
        return np.array(self._longitudes_onda)

    # Métodos del modelo 

    def absorbancia(self, concentracion_mM: float,
                    lambda_nm: float = None) -> np.ndarray | float:
        """
        Calcula la absorbancia óptica A para una concentración dada.

        Parámetros
        concentracion_mM : float
            Concentración de glucosa en el sudor [mM].
            Rango fisiológico reportado: 0.01 – 1.0 mM.
        lambda_nm : float, opcional
            Longitud de onda específica [nm]. Si es None, calcula
            para todas las longitudes de onda disponibles.

        Retorna
        float o np.ndarray
            Absorbancia A(λ, C) [adimensional, escala log10].
        """
        if lambda_nm is not None:
            return self._calcular_A_en_lambda(concentracion_mM, lambda_nm)

        return np.array([
            self._calcular_A_en_lambda(concentracion_mM, lam)
            for lam in self._longitudes_onda
        ])

    def _calcular_A_en_lambda(self, C: float, lam: float) -> float:
        """Cálculo interno de A en una longitud de onda específica."""
        # Interpolación lineal si λ no está exactamente en la tabla
        eps_g = self._interpolar(ABSORPTIVIDAD_GLUCOSA, lam)
        eps_w = self._interpolar(ABSORPTIVIDAD_AGUA, lam)
        L = self.longitud_optica_mm

        # Término de absorción de glucosa
        A_glucosa = eps_g * C * L

        # Corrección por desplazamiento de agua
        # Cada mol de glucosa desplaza ~6.15 moles de agua
        # Reducción relativa de la absorción del agua:
        # ΔA_agua = ε_w · (C_agua_desplazada) · L
        # C_agua_desplazada [mM] = COEF_DESPL * C [mM]
        if self.incluir_desplazamiento_agua:
            C_agua_desplazada_mM = COEF_DESPLAZAMIENTO_AGUA * C * CONCENTRACION_AGUA
            A_correccion = eps_w * C_agua_desplazada_mM * L
            return A_glucosa - A_correccion
        else:
            return A_glucosa

    def _interpolar(self, tabla: dict, lam: float) -> float:
        """Interpolación lineal en tablas de coeficientes."""
        lambdas = sorted(tabla.keys())
        if lam <= lambdas[0]:
            return tabla[lambdas[0]]
        if lam >= lambdas[-1]:
            return tabla[lambdas[-1]]
        for i in range(len(lambdas) - 1):
            if lambdas[i] <= lam <= lambdas[i + 1]:
                t = (lam - lambdas[i]) / (lambdas[i + 1] - lambdas[i])
                return tabla[lambdas[i]] + t * (tabla[lambdas[i + 1]] - tabla[lambdas[i]])

    def barrido_concentraciones(self, concentraciones_mM: np.ndarray,
                                lambda_nm: float = LAMBDA_REFERENCIA_NM
                                ) -> np.ndarray:
        """
        Calcula A para un vector de concentraciones a λ fija.

        Parámetros
        concentraciones_mM : np.ndarray
            Vector de concentraciones de glucosa [mM].
        lambda_nm : float
            Longitud de onda de análisis [nm].

        Retorna
        np.ndarray
            Vector de absorbancias correspondiente.
        """
        return np.array([self.absorbancia(C, lambda_nm)
                        for C in concentraciones_mM])

    def espectro_completo(self, concentracion_mM: float) -> tuple:
        """
        Retorna el espectro NIR completo para una concentración dada.

        Retorna
        (lambdas_nm, absorbancias) : tuple de np.ndarray
        """
        lambdas = self.longitudes_onda
        A = self.absorbancia(concentracion_mM)
        return lambdas, A

    def sensibilidad(self, lambda_nm: float = LAMBDA_REFERENCIA_NM) -> float:
        """
        Sensibilidad del modelo: dA/dC en una longitud de onda [mM^-1].

        Estimada numéricamente con diferencias finitas centradas.
        """
        dC = 1e-4  # mM
        C_ref = 0.5  # mM (punto de operación central)
        A_mas = self.absorbancia(C_ref + dC, lambda_nm)
        A_menos = self.absorbancia(C_ref - dC, lambda_nm)
        return (A_mas - A_menos) / (2 * dC)
    
    def concentracion_inversa(self, absorbancia: float, lambda_nm: float = LAMBDA_REFERENCIA_NM) -> float:
        """
        Calcula la concentración de glucosa en el sudor [mM] a partir de una absorbancia medida.
        Resuelve analíticamente la ecuación directa usada en _calcular_A_en_lambda.
        """
        L = self.longitud_optica_mm
        
        # Obtenemos los coeficientes usando el método de interpolación de tu clase
        eps_g = self._interpolar(ABSORPTIVIDAD_GLUCOSA, lambda_nm)
        eps_w = self._interpolar(ABSORPTIVIDAD_AGUA, lambda_nm)
        
        if self.incluir_desplazamiento_agua:
            # Despejando C de la fórmula: A = (eps_g * C * L) - (eps_w * COEF * C * C_AGUA * L)
            # Factorizamos C: A = C * [L * (eps_g - eps_w * COEF * C_AGUA)]
            denominador = L * (eps_g - eps_w * COEF_DESPLAZAMIENTO_AGUA * CONCENTRACION_AGUA)
        else:
            # Despejando C de la fórmula clásica: A = eps_g * C * L
            denominador = L * eps_g
            
        if denominador == 0:
            return 0.0
            
        C_calculada = absorbancia / denominador
        return max(0.0, C_calculada) # Evita concentraciones negativas por ruido

    def evaluar_riesgo_clinico(self, concentracion_mM: float) -> str:
        """
        Clasifica el nivel de glucosa en el sudor según umbrales fisiológicos aproximados.
        """
        if concentracion_mM < 0.01:
            return "Indetectable (Fuera de rango o error de lectura)"
        elif concentracion_mM <= 0.2:
            return "Nivel Normal (Normoglucemia probable)"
        elif concentracion_mM <= 0.4:
            return "Nivel Elevado (Riesgo de prediabetes, requiere monitoreo)"
        else:
            return "Nivel Muy Alto (Alta probabilidad de hiperglucemia / Diabetes Tipo II)"
