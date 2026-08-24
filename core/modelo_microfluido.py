"""
MÓDULO: modelo_microfluido.py
PROYECTO: Evaluación paramétrica de detección óptica NIR de glucosa en sudor
DESCRIPCIÓN: Modelado hidrodinámico laminar en microcanales (Hagen-Poiseuille, Re < 1).
"""

from typing import Tuple
import numpy as np

# ==============================================================================
# CONSTANTES FÍSICAS REOLÓGICAS (SI) (Squires & Quake, 2005; Yin et al., 2025a)
# ==============================================================================
VISCOSIDAD_SUDOR_Pa_s: float = 1.0e-3      # Viscosidad dinámica del sudor [Pa·s]
DENSIDAD_SUDOR_kg_m3: float = 1005.0       # Densidad del biofluido [kg/m³]
DIFUSIVIDAD_GLUCOSA_m2_s: float = 6.7e-10  # Coeficiente de difusión de glucosa [m²/s]
REYNOLDS_LIMITE_LAMINAR: float = 1.0       # Criterio estricto de diseño microfluídico


class ModeloMicrofluido:
    """
    Modelo de transporte de biofluido en microcanales rectangulares.
    Evalúa régimen de flujo laminar, velocidad media y tiempos de residencia.
    """

    def __init__(
        self,
        ancho_um: float = 200.0,
        alto_um: float = 50.0,
        largo_mm: float = 5.0,
        caudal_nL_min: float = 10.0,
    ):
        self.ancho_um = float(ancho_um)
        self.alto_um = float(alto_um)
        self.largo_mm = float(largo_mm)
        self.caudal_nL_min = float(caudal_nL_min)

        # Conversiones al Sistema Internacional (SI)
        self.ancho_m: float = self.ancho_um * 1e-6
        self.alto_m: float = self.alto_um * 1e-6
        self.largo_m: float = self.largo_mm * 1e-3
        self.caudal_m3_s: float = self.caudal_nL_min * 1e-12 / 60.0

    @property
    def area_transversal_m2(self) -> float:
        """Área de la sección transversal Ac = w * h [m²]."""
        return self.ancho_m * self.alto_m

    @property
    def perimetro_humedecido_m(self) -> float:
        """Perímetro de la sección transversal P = 2*(w + h) [m]."""
        return 2.0 * (self.ancho_m + self.alto_m)

    @property
    def diametro_hidraulico_m(self) -> float:
        """Diámetro hidráulico Dh = 4*Ac / P [m]."""
        if self.perimetro_humedecido_m <= 0:
            return 0.0
        return (4.0 * self.area_transversal_m2) / self.perimetro_humedecido_m

    @property
    def volumen_canal_m3(self) -> float:
        """Volumen total de la cámara microfluídica [m³]."""
        return self.area_transversal_m2 * self.largo_m

    @property
    def volumen_canal_nL(self) -> float:
        """Volumen total de la cámara en nanolitros [nL]."""
        return self.volumen_canal_m3 * 1e12

    def velocidad_media_m_s(self) -> float:
        """Velocidad media del fluido v = Q / Ac [m/s]."""
        if self.area_transversal_m2 <= 0:
            return 0.0
        return self.caudal_m3_s / self.area_transversal_m2

    def numero_reynolds(self) -> float:
        """
        Calcula el número adimensional de Reynolds:
        Re = (rho * v * Dh) / mu
        """
        v = self.velocidad_media_m_s()
        dh = self.diametro_hidraulico_m
        if VISCOSIDAD_SUDOR_Pa_s <= 0:
            return 0.0
        return (DENSIDAD_SUDOR_kg_m3 * v * dh) / VISCOSIDAD_SUDOR_Pa_s

    def es_flujo_laminar(self) -> bool:
        """Verifica si el régimen hidrodinámico cumple la restricción Re < 1."""
        return bool(self.numero_reynolds() < REYNOLDS_LIMITE_LAMINAR)

    def tiempo_residencia_s(self) -> float:
        """
        Calcula el tiempo de residencia tr = L / v = V / Q [s]
        dentro de la zona de interrogación óptica.
        """
        v = self.velocidad_media_m_s()
        if v <= 0:
            return float("inf")
        return self.largo_m / v
    
    def resumen_parametros(self) -> dict:
        return {
            "ancho_um": self.ancho_um,
            "alto_um": self.alto_um,
            "largo_mm": self.largo_mm,
            "caudal_nL_min": self.caudal_nL_min,
            "area_transversal_m2": self.area_transversal_m2,
            "diametro_hidraulico_m": self.diametro_hidraulico_m,
            "velocidad_media_m_s": self.velocidad_media_m_s(),
            "numero_reynolds": self.numero_reynolds(),
            "es_laminar": self.es_flujo_laminar(),
            "tiempo_residencia_s": self.tiempo_residencia_s(),
            "volumen_canal_nL": self.volumen_canal_nL,
        }