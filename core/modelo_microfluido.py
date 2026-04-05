"""
MÓDULO: modelo_microfluido.py
PROYECTO: Evaluación paramétrica de detección óptica NIR de glucosa en sudor

Descripción:
    Modelo conceptual simplificado del transporte microfluídico del sudor
    hacia la zona de detección óptica.

    IMPORTANTE: Este módulo NO implementa CFD tridimensional. Representa
    el comportamiento del fluido en régimen laminar (Re << 1) mediante
    relaciones analíticas del flujo de Hagen-Poiseuille, consistente
    con el alcance del proyecto (Squires & Quake, 2005; Whitesides, 2006).

Fenómenos modelados:
    - Flujo laminar en microcanal (régimen viscoso dominante)
    - Número de Reynolds (verificación de régimen laminar)
    - Tiempo de residencia del fluido en la zona de detección
    - Concentración efectiva en la zona óptica (con difusión simplificada)

Referencias:
    - Squires, T.M. & Quake, S.R. (2005). Reviews of Modern Physics, 77(3).
    - Whitesides, G.M. (2006). Nature, 442(7101).
    - Bruus, H. (1997). Theoretical Microfluidics.
"""

import numpy as np

# CONSTANTES FÍSICAS DEL SISTEMA

# Propiedades del sudor (aproximadas a solución acuosa diluida)
VISCOSIDAD_SUDOR_Pa_s = 1.0e-3      # ~agua a 37°C [Pa·s]
DENSIDAD_SUDOR_kg_m3 = 1005.0       # ~agua con solutos [kg/m³]
DIFUSIVIDAD_GLUCOSA_m2_s = 6.7e-10  # coeficiente de difusión glucosa en agua
                                    # a 37°C (Khalil et al., 2006)


class ModeloMicrofluido:
    """
    Modelo conceptual de transporte laminar en microcanal rectangular.

    Representa el confinamiento y conducción controlada del sudor hacia
    la región de detección óptica. Parámetros geométricos son
    modificables para el análisis paramétrico.

    Parámetros
    ancho_um : float
        Ancho del microcanal [µm]. Típico: 50 – 500 µm.
    alto_um : float
        Altura del microcanal [µm]. Típico: 10 – 100 µm.
    largo_mm : float
        Longitud del microcanal hasta la zona de detección [mm].
    caudal_nL_min : float
        Caudal volumétrico del fluido [nL/min].
    """

    def __init__(self,
                ancho_um: float = 200.0,
                alto_um: float = 50.0,
                largo_mm: float = 5.0,
                caudal_nL_min: float = 10.0):

        self.ancho_um = ancho_um
        self.alto_um = alto_um
        self.largo_mm = largo_mm
        self.caudal_nL_min = caudal_nL_min

    # Propiedades derivadas 

    @property
    def ancho_m(self) -> float:
        return self.ancho_um * 1e-6

    @property
    def alto_m(self) -> float:
        return self.alto_um * 1e-6

    @property
    def largo_m(self) -> float:
        return self.largo_mm * 1e-3

    @property
    def area_transversal_m2(self) -> float:
        return self.ancho_m * self.alto_m

    @property
    def caudal_m3_s(self) -> float:
        return self.caudal_nL_min * 1e-9 / 60.0  # nL/min → m³/s

    # Métodos del modelo 

    def velocidad_media_m_s(self) -> float:
        """Velocidad media del fluido en el microcanal [m/s]."""
        return self.caudal_m3_s / self.area_transversal_m2

    def numero_reynolds(self) -> float:
        """
        Número de Reynolds del flujo.

        Re = ρ · v · D_h / μ
        donde D_h es el diámetro hidráulico del canal rectangular.

        Criterio laminar: Re << 1 (típico en microfluídica)
        """
        D_h = 2 * (self.ancho_m * self.alto_m) / (self.ancho_m + self.alto_m)
        v = self.velocidad_media_m_s()
        return DENSIDAD_SUDOR_kg_m3 * v * D_h / VISCOSIDAD_SUDOR_Pa_s

    def tiempo_residencia_s(self) -> float:
        """
        Tiempo de residencia del fluido en el canal hasta la zona de
        detección [s].

        t_r = L / v_media
        """
        v = self.velocidad_media_m_s()
        if v == 0:
            return np.inf
        return self.largo_m / v

    def numero_peclet(self) -> float:
        """
        Número de Péclet para transporte de glucosa.

        Pe = v · L / D_glucosa

        Pe >> 1: transporte dominado por convección
        Pe << 1: transporte dominado por difusión
        """
        v = self.velocidad_media_m_s()
        return v * self.largo_m / DIFUSIVIDAD_GLUCOSA_m2_s

    def longitud_difusion_efectiva_mm(self) -> float:
        """
        Longitud de difusión efectiva de la glucosa en el tiempo
        de residencia [mm].

        δ = sqrt(2 · D · t_r)
        """
        t_r = self.tiempo_residencia_s()
        return np.sqrt(2 * DIFUSIVIDAD_GLUCOSA_m2_s * t_r) * 1e3

    def concentracion_efectiva_zona_optica(self,
                                        concentracion_entrada_mM: float
                                        ) -> float:
        """
        Concentración efectiva estimada en la zona de detección óptica [mM].

        Modelo simplificado: considera una dilución uniforme del analito
        debida a la difusión axial durante el transporte.

        C_ef = C_0 · factor_dilución
        donde factor_dilución ≈ exp(-Pe^-0.5) (aproximación semi-empírica)

        Nota: Para Pe >> 1 (convección dominante), C_ef ≈ C_0.
        """
        Pe = self.numero_peclet()
        if Pe >= 1.0:
            factor = np.exp(-1.0 / np.sqrt(Pe))
        else:
            factor = np.exp(-1.0)  # alta dilución por difusión
        return concentracion_entrada_mM * (1.0 - factor * 0.05)

    def es_regimen_laminar(self) -> bool:
        """Verifica que el flujo sea laminar (Re < 1)."""
        return self.numero_reynolds() < 1.0

    def resumen_parametros(self) -> dict:
        """Retorna un diccionario con los parámetros calculados del flujo."""
        return {
            "ancho_canal_um": self.ancho_um,
            "alto_canal_um": self.alto_um,
            "largo_canal_mm": self.largo_mm,
            "caudal_nL_min": self.caudal_nL_min,
            "velocidad_media_um_s": self.velocidad_media_m_s() * 1e6,
            "numero_reynolds": self.numero_reynolds(),
            "regimen_laminar": self.es_regimen_laminar(),
            "tiempo_residencia_s": self.tiempo_residencia_s(),
            "numero_peclet": self.numero_peclet(),
            "longitud_difusion_mm": self.longitud_difusion_efectiva_mm(),
        }
