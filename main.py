"""
=============================================================================
main.py — Punto de entrada del proyecto
=============================================================================

PROYECTO: Evaluación paramétrica de un modelo matemático para el monitoreo
          no invasivo de glucosa en el sudor, mediante simulación numérica
          computacional basada en detección óptica de infrarrojo cercano (NIR)

PROGRAMA: Bioingeniería — Trabajo de Grado
MODALIDAD: 100% virtual — Producto Tecnológico Digital

=============================================================================
MODOS DE USO
=============================================================================

1. INTERFAZ GRÁFICA (recomendado):
   python main.py

2. SIMULACIÓN EN CONSOLA (sin GUI):
   python main.py --consola

3. SIMULACIÓN RÁPIDA + exportar CSV:
   python main.py --exportar

=============================================================================
"""

import sys
import argparse
from pathlib import Path

# Asegurar que los módulos del proyecto estén accesibles
sys.path.insert(0, str(Path(__file__).parent))


def modo_gui():
    """Lanza la interfaz gráfica interactiva."""
    from gui.interfaz_grafica import IniciarGUI
    print("Iniciando interfaz gráfica...")
    IniciarGUI()


def modo_consola():
    """Ejecuta las simulaciones en consola y muestra resultados básicos."""
    import numpy as np
    from core.modelo_optico import ModeloBeerLambertNIR
    from core.modelo_microfluido import ModeloMicrofluido
    from core.simulacion_parametrica import SimulacionParametrica

    print("=" * 60)
    print("SIMULADOR NIR-GLUCOSA — MODO CONSOLA")
    print("Evaluación paramétrica del modelo óptico y microfluídico")
    print("=" * 60)

    # --- Modelo óptico básico ---
    print("\n[1] Modelo Óptico — Beer-Lambert NIR")
    modelo = ModeloBeerLambertNIR(longitud_optica_mm=1.0,
                                  incluir_desplazamiento_agua=True)

    concentraciones = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    lambda_ref = 1600  # nm

    print(f"\n  Longitud de onda: {lambda_ref} nm  |  L óptica: 1.0 mm")
    print(f"  {'C [mM]':>10}  {'A (con corr.)':>15}  {'A (sin corr.)':>15}")
    print(f"  {'-'*45}")

    modelo_sin = ModeloBeerLambertNIR(1.0, incluir_desplazamiento_agua=False)
    for C in concentraciones:
        A_con = modelo.absorbancia(C, lambda_ref)
        A_sin = modelo_sin.absorbancia(C, lambda_ref)
        print(f"  {C:>10.3f}  {A_con:>15.6e}  {A_sin:>15.6e}")

    S = modelo.sensibilidad(lambda_ref)
    print(f"\n  Sensibilidad dA/dC en λ={lambda_ref}nm: {S:.4e} mM⁻¹")

    # --- Modelo microfluídico ---
    print("\n[2] Modelo Microfluídico — Canal conceptual")
    mf = ModeloMicrofluido(ancho_um=200, alto_um=50,
                           largo_mm=5.0, caudal_nL_min=10.0)
    params = mf.resumen_parametros()
    print()
    for k, v in params.items():
        print(f"  {k}: {v}")

    # --- Simulación completa ---
    print("\n[3] Ejecutando simulación paramétrica completa...")
    sim = SimulacionParametrica()
    resultados = sim.ejecutar_todas()

    for nombre, df in resultados.items():
        print(f"  → '{nombre}': {df.shape[0]} filas × {df.shape[1]} columnas")

    print("\nSimulación completada. Use --exportar para guardar CSV.")
    return sim


def modo_exportar():
    """Ejecuta todas las simulaciones y exporta a CSV."""
    sim = modo_consola()
    print("\n[4] Exportando resultados...")
    sim.exportar_resultados("outputs")
    print("\n✓ Archivos guardados en la carpeta 'outputs/'")


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulador NIR-Glucosa — Bioingeniería UPB"
    )
    parser.add_argument(
        "--consola", action="store_true",
        help="Ejecutar en modo consola (sin GUI)"
    )
    parser.add_argument(
        "--exportar", action="store_true",
        help="Ejecutar simulaciones y exportar resultados a CSV"
    )
    args = parser.parse_args()

    if args.exportar:
        modo_exportar()
    elif args.consola:
        modo_consola()
    else:
        modo_gui()
