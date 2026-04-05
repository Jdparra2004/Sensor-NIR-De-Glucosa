"""
=============================================================================
MÓDULO: interfaz_grafica.py
PROYECTO: Evaluación paramétrica de detección óptica NIR de glucosa en sudor
=============================================================================

Descripción:
    Interfaz gráfica (GUI) desarrollada con Tkinter + Matplotlib.
    Permite modificar parámetros de entrada e visualizar resultados
    de simulación en tiempo real sin necesidad de modificar el código.

    Requerimiento RC-03: "Permitir la modificación de parámetros dentro
    del entorno ejecutable."

Uso:
    python interfaz_grafica.py
    o desde main.py:
        from gui.interfaz_grafica import IniciarGUI
        IniciarGUI()
=============================================================================
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.modelo_optico import ModeloBeerLambertNIR
from core.modelo_microfluido import ModeloMicrofluido
from core.simulacion_parametrica import SimulacionParametrica


# ---------------------------------------------------------------------------
# COLORES Y ESTILOS DE LA INTERFAZ
# ---------------------------------------------------------------------------

COLOR_BG = "#F4F6F8"
COLOR_PANEL = "#FFFFFF"
COLOR_ACCENT = "#2E86AB"
COLOR_BTN = "#2E86AB"
COLOR_BTN_FG = "white"
COLOR_TITULO = "#1A2E40"

FONT_TITULO = ("Helvetica", 14, "bold")
FONT_SUBTITULO = ("Helvetica", 11, "bold")
FONT_LABEL = ("Helvetica", 9)
FONT_SMALL = ("Helvetica", 8)


# ---------------------------------------------------------------------------
# VENTANA PRINCIPAL
# ---------------------------------------------------------------------------

class AppGlucosaNIR(tk.Tk):
    """
    Aplicación principal: simulador paramétrico de detección óptica NIR
    de glucosa en sudor.
    """

    def __init__(self):
        super().__init__()
        self.title("Simulador NIR-Glucosa — Bioingeniería")
        self.geometry("1200x750")
        self.configure(bg=COLOR_BG)
        self.resizable(True, True)

        self._construir_ui()

    # =========================================================================
    # CONSTRUCCIÓN DE LA INTERFAZ
    # =========================================================================

    def _construir_ui(self):
        """Construye los widgets principales de la interfaz."""

        # --- Barra de título --------------------------------------------------
        frame_titulo = tk.Frame(self, bg=COLOR_ACCENT, height=55)
        frame_titulo.pack(fill="x")
        tk.Label(
            frame_titulo,
            text="  Evaluación Paramétrica — Detección óptica NIR de glucosa en sudor",
            font=FONT_TITULO, bg=COLOR_ACCENT, fg="white", anchor="w"
        ).pack(side="left", padx=10, pady=12)

        tk.Label(
            frame_titulo,
            text="Bioingeniería UPB  ",
            font=FONT_SMALL, bg=COLOR_ACCENT, fg="#BDE0F0", anchor="e"
        ).pack(side="right", padx=10)

        # --- Contenido principal: panel izquierdo + área de gráficas ----------
        frame_main = tk.Frame(self, bg=COLOR_BG)
        frame_main.pack(fill="both", expand=True, padx=8, pady=8)

        # Panel de parámetros (izquierda)
        self.frame_params = tk.Frame(frame_main, bg=COLOR_PANEL,
                                     width=280, relief="groove", bd=1)
        self.frame_params.pack(side="left", fill="y", padx=(0, 6))
        self.frame_params.pack_propagate(False)

        self._construir_panel_parametros()

        # Área de gráficas (derecha)
        self.frame_graficas = tk.Frame(frame_main, bg=COLOR_BG)
        self.frame_graficas.pack(side="left", fill="both", expand=True)

        self._construir_area_graficas()

        # --- Barra de estado --------------------------------------------------
        self.var_estado = tk.StringVar(value="Listo. Configure los parámetros y ejecute una simulación.")
        tk.Label(self, textvariable=self.var_estado, font=FONT_SMALL,
                 bg="#E8EBF0", anchor="w", relief="sunken",
                 bd=1).pack(fill="x", side="bottom")

        # Ejecutar simulación inicial
        self._ejecutar_simulacion()

    def _construir_panel_parametros(self):
        """Panel lateral de parámetros con widgets de entrada."""
        p = self.frame_params

        tk.Label(p, text="PARÁMETROS DEL MODELO",
                 font=FONT_SUBTITULO, bg=COLOR_PANEL,
                 fg=COLOR_TITULO).pack(pady=(14, 2), padx=10, anchor="w")
        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=10, pady=4)

        # --- Sección: Parámetros ópticos --------------------------------------
        tk.Label(p, text="⬤  Modelo Óptico (Beer-Lambert)",
                 font=("Helvetica", 9, "bold"), bg=COLOR_PANEL,
                 fg=COLOR_ACCENT).pack(anchor="w", padx=12, pady=(6, 2))

        self.var_lambda = self._slider_param(
            p, "Longitud de onda λ [nm]", 1000, 1700, 1600, step=50)

        self.var_longitud_optica = self._slider_param(
            p, "Longitud óptica L [mm]", 0.1, 10.0, 1.0, step=0.1,
            es_float=True)

        self.var_C_min = self._slider_param(
            p, "C mínima [mM]", 0.01, 0.5, 0.01, step=0.01, es_float=True)

        self.var_C_max = self._slider_param(
            p, "C máxima [mM]", 0.1, 2.0, 1.0, step=0.1, es_float=True)

        self.var_correccion_agua = tk.BooleanVar(value=True)
        tk.Checkbutton(
            p, text="Corrección por desplazamiento de agua",
            variable=self.var_correccion_agua,
            font=FONT_SMALL, bg=COLOR_PANEL,
            command=self._ejecutar_simulacion
        ).pack(anchor="w", padx=14, pady=2)

        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=10, pady=6)

        # --- Sección: Microfluídica -------------------------------------------
        tk.Label(p, text="⬤  Modelo Microfluídico",
                 font=("Helvetica", 9, "bold"), bg=COLOR_PANEL,
                 fg=COLOR_ACCENT).pack(anchor="w", padx=12, pady=(2, 2))

        self.var_ancho_canal = self._slider_param(
            p, "Ancho canal [µm]", 50, 500, 200, step=10)

        self.var_alto_canal = self._slider_param(
            p, "Alto canal [µm]", 10, 200, 50, step=5)

        self.var_caudal = self._slider_param(
            p, "Caudal [nL/min]", 1, 100, 10, step=1)

        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=10, pady=6)

        # --- Botones ----------------------------------------------------------
        tk.Button(
            p, text="▶  Ejecutar Simulación",
            bg=COLOR_BTN, fg=COLOR_BTN_FG,
            font=("Helvetica", 10, "bold"),
            relief="flat", padx=8, pady=6,
            command=self._ejecutar_simulacion
        ).pack(fill="x", padx=12, pady=(4, 2))

        tk.Button(
            p, text="💾  Exportar Resultados",
            bg="#44BBA4", fg="white",
            font=("Helvetica", 9, "bold"),
            relief="flat", padx=8, pady=5,
            command=self._exportar
        ).pack(fill="x", padx=12, pady=2)

        tk.Button(
            p, text="Restablecer valores por defecto",
            bg="#E8EBF0", fg="#555",
            font=FONT_SMALL,
            relief="flat", padx=4, pady=3,
            command=self._restablecer
        ).pack(fill="x", padx=12, pady=2)

    def _slider_param(self, parent, label: str, vmin, vmax, default,
                      step=1, es_float=False):
        """Crea un slider + etiqueta de valor para un parámetro."""
        frame = tk.Frame(parent, bg=COLOR_PANEL)
        frame.pack(fill="x", padx=12, pady=2)

        tk.Label(frame, text=label, font=FONT_SMALL,
                 bg=COLOR_PANEL, anchor="w").pack(anchor="w")

        var = tk.DoubleVar(value=default)
        var_display = tk.StringVar(value=str(default))

        row = tk.Frame(frame, bg=COLOR_PANEL)
        row.pack(fill="x")

        slider = ttk.Scale(
            row, from_=vmin, to=vmax, variable=var,
            orient="horizontal",
            command=lambda v: (
                var_display.set(f"{float(v):.2f}" if es_float else f"{int(float(v))}"),
                self._ejecutar_simulacion()
            )
        )
        slider.pack(side="left", fill="x", expand=True)

        tk.Label(row, textvariable=var_display, font=FONT_SMALL,
                 bg=COLOR_PANEL, width=6, anchor="e").pack(side="right")

        return var

    def _construir_area_graficas(self):
        """Área de gráficas con pestañas."""
        self.notebook = ttk.Notebook(self.frame_graficas)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: A vs C
        self.tab1 = tk.Frame(self.notebook, bg=COLOR_BG)
        self.notebook.add(self.tab1, text="  A vs C  ")

        # Tab 2: Espectro NIR
        self.tab2 = tk.Frame(self.notebook, bg=COLOR_BG)
        self.notebook.add(self.tab2, text="  Espectro NIR  ")

        # Tab 3: Sensibilidad
        self.tab3 = tk.Frame(self.notebook, bg=COLOR_BG)
        self.notebook.add(self.tab3, text="  Sensibilidad  ")

        # Tab 4: Microfluídica
        self.tab4 = tk.Frame(self.notebook, bg=COLOR_BG)
        self.notebook.add(self.tab4, text="  Microfluídica  ")

        # Figuras de Matplotlib para cada tab
        self._fig1, self._ax1 = self._nueva_figura(self.tab1)
        self._fig2, self._ax2 = self._nueva_figura(self.tab2)
        self._fig3, self._axes3 = self._nueva_figura(self.tab3, subplots=(1, 2))
        self._fig4, self._axes4 = self._nueva_figura(self.tab4, subplots=(1, 2))

    def _nueva_figura(self, parent, subplots=None):
        """Crea e incrusta una figura Matplotlib en un Frame de Tkinter."""
        fig = Figure(figsize=(8, 5), dpi=100, facecolor=COLOR_BG)

        if subplots:
            axes = fig.subplots(*subplots)
        else:
            ax = fig.add_subplot(111)
            axes = ax

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        canvas.draw()

        fig._canvas = canvas  # referencia para draw()
        return fig, axes

    # =========================================================================
    # LÓGICA DE SIMULACIÓN
    # =========================================================================

    def _ejecutar_simulacion(self, *args):
        """Recoge parámetros y actualiza todas las gráficas."""
        try:
            self.var_estado.set("Calculando simulaciones...")
            self.update_idletasks()

            lambda_nm = float(self.var_lambda.get())
            L = float(self.var_longitud_optica.get())
            C_min = float(self.var_C_min.get())
            C_max = float(self.var_C_max.get())
            corr_agua = bool(self.var_correccion_agua.get())
            ancho = float(self.var_ancho_canal.get())
            alto = float(self.var_alto_canal.get())
            caudal = float(self.var_caudal.get())

            if C_min >= C_max:
                C_max = C_min + 0.1

            # Modelos
            modelo_op = ModeloBeerLambertNIR(L, incluir_desplazamiento_agua=corr_agua)
            modelo_mf = ModeloMicrofluido(ancho, alto, 5.0, caudal)

            # --- Tab 1: A vs C ---
            C_vec = np.linspace(C_min, C_max, 100)
            A_vec = modelo_op.barrido_concentraciones(C_vec, lambda_nm)
            modelo_op2 = ModeloBeerLambertNIR(L, incluir_desplazamiento_agua=False)
            A_vec2 = modelo_op2.barrido_concentraciones(C_vec, lambda_nm)

            self._fig1.clf()
            ax = self._fig1.add_subplot(111)
            ax.plot(C_vec, A_vec, color="#2E86AB", lw=2,
                    label=f"Con corr. agua — L={L:.1f} mm")
            ax.plot(C_vec, A_vec2, color="#C73E1D", lw=2, ls="--",
                    label="Sin corrección agua")
            ax.set_xlabel("Concentración glucosa [mM]", fontsize=9)
            ax.set_ylabel("Absorbancia A [u.a.]", fontsize=9)
            ax.set_title(f"Absorbancia vs Concentración  (λ={lambda_nm:.0f} nm)",
                         fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axvspan(0.01, 1.0, alpha=0.06, color="green")
            self._fig1._canvas.draw()

            # --- Tab 2: Espectro NIR ---
            lambdas = modelo_op.longitudes_onda
            concs = [C_min, (C_min + C_max) / 2, C_max]
            import matplotlib.cm as cm
            cmap = cm.plasma

            self._fig2.clf()
            ax2 = self._fig2.add_subplot(111)
            for i, C in enumerate(concs):
                _, A = modelo_op.espectro_completo(C)
                ax2.plot(lambdas, A, color=cmap(0.2 + i * 0.35),
                         lw=2, marker="o", ms=5, label=f"C={C:.3f} mM")
            ax2.set_xlabel("Longitud de onda λ [nm]", fontsize=9)
            ax2.set_ylabel("Absorbancia A [u.a.]", fontsize=9)
            ax2.set_title("Espectro NIR simulado", fontsize=10)
            ax2.axvspan(1600, 1700, alpha=0.12, color="orange")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            self._fig2._canvas.draw()

            # --- Tab 3: Sensibilidad ---
            L_vec = np.linspace(0.1, 10.0, 50)
            sens = []
            A_Cref = []
            for Li in L_vec:
                m = ModeloBeerLambertNIR(Li, corr_agua)
                sens.append(m.sensibilidad(lambda_nm))
                A_Cref.append(m.absorbancia(0.5, lambda_nm))

            self._fig3.clf()
            a3a, a3b = self._fig3.subplots(1, 2)
            a3a.plot(L_vec, A_Cref, color="#2E86AB", lw=2)
            a3a.set_xlabel("L óptica [mm]", fontsize=9)
            a3a.set_ylabel("A en C=0.5 mM", fontsize=9)
            a3a.set_title("Absorbancia vs camino óptico", fontsize=9)
            a3a.axvline(x=L, color="red", ls="--", lw=1.2, label=f"L={L:.1f}")
            a3a.legend(fontsize=8)
            a3a.grid(True, alpha=0.3)

            a3b.plot(L_vec, sens, color="#A23B72", lw=2)
            a3b.set_xlabel("L óptica [mm]", fontsize=9)
            a3b.set_ylabel("dA/dC [mM⁻¹]", fontsize=9)
            a3b.set_title("Sensibilidad vs camino óptico", fontsize=9)
            a3b.axvline(x=L, color="red", ls="--", lw=1.2, label=f"L={L:.1f}")
            a3b.legend(fontsize=8)
            a3b.grid(True, alpha=0.3)
            self._fig3.tight_layout()
            self._fig3._canvas.draw()

            # --- Tab 4: Microfluídica ---
            params = modelo_mf.resumen_parametros()
            Q_vec = np.linspace(1, 100, 50)
            Re_vec = []
            tr_vec = []
            for Q in Q_vec:
                m = ModeloMicrofluido(ancho, alto, 5.0, Q)
                Re_vec.append(m.numero_reynolds())
                tr_vec.append(m.tiempo_residencia_s())

            self._fig4.clf()
            a4a, a4b = self._fig4.subplots(1, 2)
            a4a.plot(Q_vec, Re_vec, color="#F18F01", lw=2)
            a4a.axhline(y=1.0, color="red", ls="--", lw=1,
                        label="Re=1 (límite laminar)")
            a4a.axvline(x=caudal, color="#2E86AB", ls=":", lw=1.5,
                        label=f"Q={caudal:.0f} nL/min")
            a4a.set_xlabel("Caudal [nL/min]", fontsize=9)
            a4a.set_ylabel("Número de Reynolds", fontsize=9)
            a4a.set_title(f"Régimen de flujo\n(ancho={ancho:.0f} µm)", fontsize=9)
            a4a.legend(fontsize=7.5)
            a4a.grid(True, alpha=0.3)

            a4b.plot(Q_vec, tr_vec, color="#44BBA4", lw=2)
            a4b.axvline(x=caudal, color="#2E86AB", ls=":", lw=1.5,
                        label=f"Q={caudal:.0f} nL/min")
            a4b.set_xlabel("Caudal [nL/min]", fontsize=9)
            a4b.set_ylabel("Tiempo de residencia [s]", fontsize=9)
            a4b.set_title("Tiempo de residencia en canal", fontsize=9)
            a4b.legend(fontsize=7.5)
            a4b.grid(True, alpha=0.3)

            # Métricas en el panel
            Re = params["numero_reynolds"]
            estado_laminar = "✓ Laminar" if Re < 1 else "✗ No laminar"
            self._fig4.text(
                0.5, 0.01,
                f"Re={Re:.4f}  |  Pe={params['numero_peclet']:.1f}  "
                f"|  t_r={params['tiempo_residencia_s']:.1f}s  |  {estado_laminar}",
                ha="center", fontsize=8, color="#333"
            )
            self._fig4.tight_layout(rect=[0, 0.04, 1, 1])
            self._fig4._canvas.draw()

            self.var_estado.set(
                f"Simulación completada  |  λ={lambda_nm:.0f} nm  |  "
                f"L={L:.2f} mm  |  C∈[{C_min:.3f}, {C_max:.2f}] mM  |  Re={Re:.4f}"
            )

        except Exception as e:
            self.var_estado.set(f"Error: {e}")
            messagebox.showerror("Error de simulación", str(e))

    def _exportar(self):
        """Exporta todas las simulaciones a CSV."""
        try:
            sim = SimulacionParametrica()
            sim.ejecutar_todas()
            sim.exportar_resultados("outputs")
            messagebox.showinfo("Exportación exitosa",
                                "Resultados guardados en la carpeta 'outputs/'")
            self.var_estado.set("Resultados exportados en outputs/")
        except Exception as e:
            messagebox.showerror("Error de exportación", str(e))

    def _restablecer(self):
        """Restablece todos los parámetros a sus valores por defecto."""
        self.var_lambda.set(1600)
        self.var_longitud_optica.set(1.0)
        self.var_C_min.set(0.01)
        self.var_C_max.set(1.0)
        self.var_correccion_agua.set(True)
        self.var_ancho_canal.set(200)
        self.var_alto_canal.set(50)
        self.var_caudal.set(10)
        self._ejecutar_simulacion()


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------

def IniciarGUI():
    """Lanza la interfaz gráfica."""
    app = AppGlucosaNIR()
    app.mainloop()


if __name__ == "__main__":
    IniciarGUI()
