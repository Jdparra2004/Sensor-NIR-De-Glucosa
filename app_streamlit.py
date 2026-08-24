"""
app_streamlit.py — Interfaz interactiva del Biosensor NIR
PROYECTO: Simulación paramétrica de detección óptica NIR de glucosa en sudor
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.modelo_optico import ModeloBeerLambertNIR
from core.modelo_microfluido import ModeloMicrofluido

st.set_page_config(page_title="Biosensor NIR - Glucosa en Sudor", layout="wide")

# Configuración estándar para colocar la leyenda abajo en todas las figuras
LEYENDA_INFERIOR = dict(
    orientation="h",
    yanchor="top",
    y=-0.25,
    xanchor="center",
    x=0.5
)

# --- SIDEBAR ---
st.sidebar.header("Parámetros de Diseño y Simulación")
st.sidebar.subheader("Óptica NIR")
lambda_nm = st.sidebar.slider("Longitud de onda (λ) [nm]", 1000, 1700, 1600, 1)
L_mm = st.sidebar.slider("Camino óptico (L) [mm]", 0.1, 2.0, 1.0, 0.1)
c_sim = st.sidebar.slider("Concentración de glucosa (C) [mM]", 0.01, 1.0, 0.20, 0.01)
noise_instrumental = st.sidebar.checkbox("Inyectar ruido fotométrico instrumental (±5%)", value=False)

st.sidebar.subheader("Microfluídica")
Q_nlmin = st.sidebar.slider("Caudal volumétrico (Q) [nL/min]", 1.0, 10.0, 5.0, 0.1)
w_um = st.sidebar.slider("Ancho del canal (w) [µm]", 50, 500, 200, 10)
h_um = st.sidebar.slider("Alto del canal (h) [µm]", 10, 200, 50, 10)
largo_mm = st.sidebar.slider("Largo de celda [mm]", 0.5, 5.0, 1.0, 0.5)

# --- MODELOS ---
modelo_optico = ModeloBeerLambertNIR(longitud_optica_mm=L_mm)
modelo_micro = ModeloMicrofluido(w_um, h_um, largo_mm, Q_nlmin)

def aplicar_ruido(valor):
    return valor * np.random.uniform(0.95, 1.05) if noise_instrumental else valor

# --- PÁGINAS ---
st.title("Biosensor NIR: Simulación Integrada")
tab1, tab2, tab3, tab4 = st.tabs(["Óptica NIR", "Microfluídica", "Sensibilidad", "Inferencia Clínica"])

# Tab 1: Óptica
with tab1:
    c_range = np.linspace(0.01, 1.0, 100)
    abs_vals = [aplicar_ruido(modelo_optico.absorbancia(c, lambda_nm)) for c in c_range]
    
    col1, col2 = st.columns(2)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=c_range, y=abs_vals, mode="lines", name="Absorbancia neta (Beer-Lambert corregido)"))
    fig1.update_layout(
        title="Absorbancia neta vs Concentración",
        xaxis_title="C [mM]",
        yaxis_title="A [u.a.]",
        showlegend=True,
        legend=LEYENDA_INFERIOR
    )
    col1.plotly_chart(fig1)

    lambdas, spect = modelo_optico.espectro_completo(c_sim, lambdas=np.linspace(1000, 1700, 100))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=lambdas, y=spect, mode="lines", name=f"Espectro NIR (C = {c_sim} mM)"))
    fig2.add_vrect(x0=1600, x1=1700, fillcolor="lightgray", opacity=0.3, annotation_text="Ventana 1600-1700 nm")
    fig2.update_layout(
        title="Espectro NIR",
        xaxis_title="λ [nm]",
        yaxis_title="A [u.a.]",
        showlegend=True,
        legend=LEYENDA_INFERIOR
    )
    col2.plotly_chart(fig2)

    st.latex(r"A_{\text{neta}}(\lambda) = (\epsilon_g(\lambda) - \epsilon_w(\lambda) \cdot \delta_w) \cdot C \cdot L")
    A_actual = modelo_optico.absorbancia(c_sim, lambda_nm)
    st.info(
        rf"A λ = {lambda_nm} nm, L = {L_mm} mm y C = {c_sim} mM, la absorbancia neta es de **{A_actual:.5f} u.a.** "
        rf"La pendiente negativa responde al desplazamiento volumétrico del agua ($\delta_w \approx 6.15$): "
        rf"al disolverse glucosa, se excluye solvente, disminuyendo la absorción del agua."
    )

# Tab 2: Microfluídica
with tab2:
    Q_range = np.linspace(1.0, 10.0, 50)
    Re_vals = [ModeloMicrofluido(w_um, h_um, largo_mm, q).numero_reynolds() for q in Q_range]
    tr_vals = [ModeloMicrofluido(w_um, h_um, largo_mm, q).tiempo_residencia_s() for q in Q_range]
    
    col1, col2 = st.columns(2)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=Q_range, y=Re_vals, mode="lines", name="Número de Reynolds (Re)"))
    fig3.add_trace(go.Scatter(x=[Q_range[0], Q_range[-1]], y=[1.0, 1.0], mode="lines", line=dict(dash="dash", color="red"), name="Límite laminar (Re = 1.0)"))
    fig3.update_layout(
        title="Número de Reynolds vs Caudal",
        xaxis_title="Q [nL/min]",
        yaxis_title="Re",
        showlegend=True,
        legend=LEYENDA_INFERIOR
    )
    col1.plotly_chart(fig3)
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=Q_range, y=tr_vals, mode="lines", name="Tiempo de residencia (t_r)"))
    fig4.update_layout(
        title="Tiempo de residencia vs Caudal",
        xaxis_title="Q [nL/min]",
        yaxis_title="t_r [s]",
        showlegend=True,
        legend=LEYENDA_INFERIOR
    )
    col2.plotly_chart(fig4)
    
    st.latex(r"Re = \frac{2\rho Q}{\mu(w+h)} \quad \text{y} \quad t_r = \frac{V_{\text{celda}}}{Q}")
    re_actual = modelo_micro.numero_reynolds()
    st.info(
        f"Con Q={Q_nlmin} nL/min, Re = **{re_actual:.6f}** "
        f"({'Régimen Laminar' if re_actual < 1 else 'Flujo No Laminar'}). "
        f"Velocidad media: **{modelo_micro.velocidad_media_m_s()*1e6:.2f} µm/s**. "
        f"Tiempo de residencia: **{modelo_micro.tiempo_residencia_s():.2f} s**."
    )

# Tab 3: Sensibilidad
with tab3:
    L_range = np.linspace(0.1, 2.0, 50)
    sens_vals = [ModeloBeerLambertNIR(L).sensibilidad(lambda_nm) for L in L_range]
    abs_vals_L = [ModeloBeerLambertNIR(L).absorbancia(c_sim, lambda_nm) for L in L_range]
    
    col1, col2 = st.columns(2)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=L_range, y=sens_vals, mode="lines", name="Sensibilidad local (dA/dC)"))
    fig5.update_layout(
        title="Sensibilidad vs Camino óptico",
        xaxis_title="L [mm]",
        yaxis_title="dA/dC [mM⁻¹]",
        showlegend=True,
        legend=LEYENDA_INFERIOR
    )
    col1.plotly_chart(fig5)
    
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=L_range, y=abs_vals_L, mode="lines", name=f"Absorbancia (C = {c_sim} mM)"))
    fig6.update_layout(
        title="Absorbancia vs Camino óptico",
        xaxis_title="L [mm]",
        yaxis_title="A [u.a.]",
        showlegend=True,
        legend=LEYENDA_INFERIOR
    )
    col2.plotly_chart(fig6)
    
    st.latex(r"\text{Sensibilidad} = \frac{\partial A}{\partial C} = (\epsilon_g(\lambda) - \epsilon_w(\lambda) \cdot \delta_w) \cdot L")
    st.info("La sensibilidad aumenta linealmente con el camino óptico L, permitiendo amplificar la señal neta sin afectar la selectividad del desplazamiento del solvente.")

# Tab 4: Inferencia
with tab4:
    st.subheader("Inferencia Clínica")
    A_med = st.number_input("Absorbancia medida (A)", value=-0.05, step=0.001, format="%.5f")
    if st.button("Estimar"):
        c_est = modelo_optico.concentracion_inversa(A_med, lambda_nm)
        st.write(f"Concentración estimada: **{c_est:.4f} mM** — Clasificación: **{modelo_optico.evaluar_clasificacion_fisiologica(c_est)}**")
        
    st.markdown("---")
    st.subheader("Procesamiento por Lotes (CSV)")
    uploaded = st.file_uploader("Subir archivo CSV de muestras", type=["csv", "txt"])
    if uploaded is not None:
        try:
            # Intento de lectura con C engine rápido
            try:
                uploaded.seek(0)
                df_lote = pd.read_csv(uploaded)
            except Exception:
                # Fallback con motor Python sin pasar el parámetro low_memory
                uploaded.seek(0)
                df_lote = pd.read_csv(uploaded, sep=None, engine="python")
            
            # Detección flexible de la columna de absorbancia
            col_abs = None
            candidatos_abs = ['absorbancia_1600nm', 'absorbancia_1650nm', 'absorbancia_medida', 'absorbancia', 'Absorbance', 'A']
            for cand in candidatos_abs:
                if cand in df_lote.columns:
                    col_abs = cand
                    break
            
            if col_abs is None:
                for col in df_lote.columns:
                    if 'abs' in col.lower():
                        col_abs = col
                        break

            if col_abs is not None:
                valores_abs = pd.to_numeric(df_lote[col_abs], errors="coerce")
                df_lote["Glucosa_Estimada_mM"] = valores_abs.apply(
                    lambda a: modelo_optico.concentracion_inversa(float(a), lambda_nm) if pd.notna(a) else np.nan
                ).round(4)
                
                candidatos_ref = ['glucosa_referencia_mM', 'Glucosa_Real_mM', 'glucosa_mM', 'glucose_mM', 'C_real']
                col_ref = next((c for c in candidatos_ref if c in df_lote.columns), None)
                if col_ref is not None:
                    c_real = pd.to_numeric(df_lote[col_ref], errors="coerce")
                    df_lote["Error_%"] = (np.abs(df_lote["Glucosa_Estimada_mM"] - c_real) / np.where(c_real != 0, c_real, 1e-12) * 100).round(2)
                
                df_lote["Clasificación_Metabólica"] = df_lote["Glucosa_Estimada_mM"].apply(
                    lambda c: modelo_optico.evaluar_clasificacion_fisiologica(c) if pd.notna(c) else "Indeterminado"
                )
                
                # Mostrar primeras filas si el archivo es muy grande para no sobrecargar el navegador
                st.dataframe(df_lote.head(1000))
                if len(df_lote) > 1000:
                    st.caption(f"Mostrando las primeras 1,000 filas de un total de {len(df_lote):,} registros procesados.")
                
                st.download_button(
                    label="Descargar CSV Procesado",
                    data=df_lote.to_csv(index=False).encode("utf-8"),
                    file_name="resultados_procesamiento_lote.csv",
                    mime="text/csv"
                )
            else:
                st.error("No se detectó una columna de absorbancia válida en el archivo CSV subido.")
        except Exception as e:
            st.error(f"Error al procesar el archivo CSV: {e}")