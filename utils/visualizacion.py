"""
=============================================================================
MÓDULO: visualizacion.py
PROYECTO: Evaluación paramétrica de detección óptica NIR de glucosa en sudor
=============================================================================

Descripción:
    Funciones de visualización para los resultados de simulación.
    Genera gráficas de publicación usando Matplotlib.

    Todas las figuras son reproducibles: usar plt.savefig() para exportar.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL DE ESTILO
# ---------------------------------------------------------------------------

STYLE_CONFIG = {
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

plt.rcParams.update(STYLE_CONFIG)

# Paleta de colores del proyecto
COLORES = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
           "#44BBA4", "#E94F37", "#393E41"]


# ---------------------------------------------------------------------------
# GRÁFICA 1: A vs C (Beer-Lambert)
# ---------------------------------------------------------------------------

def graficar_absorbancia_vs_concentracion(df: pd.DataFrame,
                                          guardar: bool = False,
                                          ruta: str = "outputs"):
    """
    Gráfica de absorbancia vs concentración de glucosa.
    Verifica la linealidad del modelo de Beer-Lambert.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Absorbancia vs Concentración de Glucosa en Sudor\n"
        "Modelo Beer-Lambert NIR (λ = 1600 nm)",
        fontsize=13, fontweight="bold"
    )

    # Panel izquierdo: comparación con/sin corrección
    ax1 = axes[0]
    ax1.plot(df["C_mM"], df["A_con_correccion_agua"],
             color=COLORES[0], lw=2, label="Con corrección (desplaz. agua)")
    ax1.plot(df["C_mM"], df["A_sin_correccion_agua"],
             color=COLORES[1], lw=2, ls="--", label="Sin corrección")
    ax1.set_xlabel("Concentración de glucosa [mM]")
    ax1.set_ylabel("Absorbancia A [u.a.]")
    ax1.set_title("Señal óptica simulada")
    ax1.legend()
    ax1.axvspan(0.01, 1.0, alpha=0.07, color="green",
                label="Rango fisiológico en sudor")

    # Panel derecho: diferencia porcentual (impacto de la corrección)
    ax2 = axes[1]
    ax2.plot(df["C_mM"], df["diferencia_pct"],
             color=COLORES[2], lw=2)
    ax2.set_xlabel("Concentración de glucosa [mM]")
    ax2.set_ylabel("Diferencia [%]")
    ax2.set_title("Impacto del desplazamiento de agua")
    ax2.axhline(y=0, color="gray", ls="--", lw=0.8)

    plt.tight_layout()
    if guardar:
        Path(ruta).mkdir(exist_ok=True)
        fig.savefig(f"{ruta}/absorbancia_vs_concentracion.png",
                    dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# GRÁFICA 2: Espectro NIR
# ---------------------------------------------------------------------------

def graficar_espectro_nir(df: pd.DataFrame,
                          guardar: bool = False,
                          ruta: str = "outputs"):
    """
    Gráfica del espectro NIR para múltiples concentraciones.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    cols_A = [c for c in df.columns if c.startswith("A_C")]
    n_curvas = len(cols_A)
    cmap = cm.get_cmap("plasma", n_curvas)

    for i, col in enumerate(cols_A):
        conc_label = col.replace("A_C", "C=").replace("mM", " mM")
        ax.plot(df["lambda_nm"], df[col],
                color=cmap(i), lw=2, label=conc_label, marker="o",
                markersize=4)

    ax.set_xlabel("Longitud de onda λ [nm]")
    ax.set_ylabel("Absorbancia A [u.a.]")
    ax.set_title(
        "Espectro NIR de Glucosa en Sudor\n"
        "Ley de Beer-Lambert adaptada — múltiples concentraciones",
        fontweight="bold"
    )
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="Concentración")

    # Marcar ventanas espectrales de interés (Yang et al., 2025)
    ax.axvspan(1600, 1700, alpha=0.12, color="orange",
               label="Ventana 1600–1700 nm\n(Yang et al., 2025)")

    plt.tight_layout()
    if guardar:
        Path(ruta).mkdir(exist_ok=True)
        fig.savefig(f"{ruta}/espectro_nir.png", dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# GRÁFICA 3: Sensibilidad vs Longitud Óptica
# ---------------------------------------------------------------------------

def graficar_sensibilidad(df: pd.DataFrame,
                          guardar: bool = False,
                          ruta: str = "outputs"):
    """
    Gráfica de sensibilidad del modelo vs longitud del camino óptico.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Análisis de Sensibilidad — Longitud del Camino Óptico",
        fontsize=13, fontweight="bold"
    )

    ax1 = axes[0]
    ax1.plot(df["L_optica_mm"], df["A_en_Cref"],
             color=COLORES[0], lw=2)
    ax1.set_xlabel("Longitud óptica L [mm]")
    ax1.set_ylabel("Absorbancia A en C=0.5 mM")
    ax1.set_title("Absorbancia vs camino óptico")

    ax2 = axes[1]
    ax2.plot(df["L_optica_mm"], df["sensibilidad_dA_dC"],
             color=COLORES[3], lw=2)
    ax2.set_xlabel("Longitud óptica L [mm]")
    ax2.set_ylabel("Sensibilidad dA/dC [mM⁻¹]")
    ax2.set_title("Sensibilidad vs camino óptico\n(parámetro de diseño clave)")

    plt.tight_layout()
    if guardar:
        Path(ruta).mkdir(exist_ok=True)
        fig.savefig(f"{ruta}/sensibilidad_longitud_optica.png",
                    dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# GRÁFICA 4: Mapa de Absorbancia (heatmap)
# ---------------------------------------------------------------------------

def graficar_mapa_absorbancia(df: pd.DataFrame,
                              guardar: bool = False,
                              ruta: str = "outputs"):
    """
    Mapa 2D de absorbancia en función de λ y C.
    Permite identificar ventanas espectrales óptimas.
    """
    pivot = df.pivot_table(index="C_mM", columns="lambda_nm",
                           values="absorbancia")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[
            pivot.columns.min(), pivot.columns.max(),
            pivot.index.min(), pivot.index.max()
        ]
    )
    plt.colorbar(im, ax=ax, label="Absorbancia A [u.a.]")
    ax.set_xlabel("Longitud de onda λ [nm]")
    ax.set_ylabel("Concentración de glucosa [mM]")
    ax.set_title(
        "Mapa de Absorbancia NIR — A(λ, C)\n"
        "Rango fisiológico de glucosa en sudor",
        fontweight="bold"
    )

    plt.tight_layout()
    if guardar:
        Path(ruta).mkdir(exist_ok=True)
        fig.savefig(f"{ruta}/mapa_absorbancia.png",
                    dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# GRÁFICA 5: Resumen microfluídico
# ---------------------------------------------------------------------------

def graficar_parametros_microfluido(df: pd.DataFrame,
                                    guardar: bool = False,
                                    ruta: str = "outputs"):
    """
    Visualiza el número de Reynolds y tiempo de residencia
    en función del caudal y ancho del canal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Análisis Paramétrico del Modelo Microfluídico Conceptual",
        fontsize=13, fontweight="bold"
    )

    anchos = df["ancho_canal_um"].unique()
    cmap = cm.get_cmap("tab10", len(anchos))

    for i, ancho in enumerate(anchos):
        subset = df[df["ancho_canal_um"] == ancho].sort_values("caudal_nL_min")
        label = f"Ancho={ancho:.0f} µm"

        axes[0].plot(subset["caudal_nL_min"], subset["numero_reynolds"],
                     marker="o", lw=1.5, color=cmap(i), label=label)
        axes[1].plot(subset["caudal_nL_min"], subset["tiempo_residencia_s"],
                     marker="s", lw=1.5, color=cmap(i), label=label)

    axes[0].axhline(y=1.0, color="red", ls="--", lw=1.2, label="Re = 1 (límite laminar)")
    axes[0].set_xlabel("Caudal [nL/min]")
    axes[0].set_ylabel("Número de Reynolds (Re)")
    axes[0].set_title("Régimen de flujo")
    axes[0].legend(fontsize=7.5)

    axes[1].set_xlabel("Caudal [nL/min]")
    axes[1].set_ylabel("Tiempo de residencia [s]")
    axes[1].set_title("Tiempo de residencia en canal")
    axes[1].legend(fontsize=7.5)

    plt.tight_layout()
    if guardar:
        Path(ruta).mkdir(exist_ok=True)
        fig.savefig(f"{ruta}/parametros_microfluido.png",
                    dpi=150, bbox_inches="tight")
    return fig
