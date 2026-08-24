"""
MÓDULO: app.py
PROYECTO: SIMULACIÓN PARAMÉTRICA Y CARACTERIZACIÓN DE DESEMPEÑO DE UN SENSOR ÓPTICO NIR Y TRANSPORTE MICROFLUIDÍCO PARA EL MONITOREO DE GLUCOSA EN SUDOR
PROGRAMA: Bioingeniería - Trabajo de Grado

Descripción:
    Aplicación interactiva construida con Streamlit para la simulación,
    análisis paramétrico y validación clínica del biosensor.
    Conecta los modelos físicos de óptica NIR y dinámica de fluidos.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple, List

# Importaciones de los módulos core del proyecto
from core.modelo_optico import ModeloBeerLambertNIR
from core.modelo_microfluido import ModeloMicrofluido

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Biosensor NIR - Glucosa en Sudor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR: CONTROLES ---
st.sidebar.header("⚙️ Configuración del Sistema")

st.sidebar.subheader("Óptica NIR")
lambda_nm = st.sidebar.slider("Longitud de onda (λ) [nm]", 1000, 1700, 1600, step=1)
L_mm = st.sidebar.slider("Camino óptico (L) [mm]", 0.1, 2.0, 1.0, step=0.1)
noise_instrumental = st.sidebar.checkbox("Inyectar ruido fotométrico (+/- 5%)")

st.sidebar.subheader("Microfluídica")
Q_nlmin_fixed = st.sidebar.slider("Caudal base (Q) [nL/min]", 1.0, 10.0, 5.0, step=0.1)
w_um = st.sidebar.slider("Ancho canal (w) [µm]", 50, 500, 200, step=10)
h_um = st.sidebar.slider("Alto canal (h) [µm]", 10, 200, 50, step=10)

st.sidebar.subheader("Simulación Clínica")
c_sim = st.sidebar.number_input("Concentración glucosa [mM]", 0.01, 1.0, 0.2, step=0.01)

# --- INSTANCIACIÓN DE MODELOS ---
modelo_optico = ModeloBeerLambertNIR(longitud_optica_mm=L_mm, incluir_desplazamiento_agua=True)
modelo_micro = ModeloMicrofluido(ancho_um=w_um, alto_um=h_um, caudal_nL_min=Q_nlmin_fixed)

# --- FUNCIÓN PARA APLICAR RUIDO ---
def aplicar_ruido(valor: float) -> float:
    if noise_instrumental:
        return valor * np.random.uniform(0.95, 1.05)
    return valor

# --- PÁGINA PRINCIPAL ---
st.title("🔬 Caracterización de Biosensor NIR para Glucosa en Sudor")

tab1, tab2, tab3, tab4 = st.tabs([
    "Óptica NIR", "Microfluídica", "Sensibilidad y Análisis", "Inferencia Clínica y Lotes"
])

# Tab 1: Óptica
with tab1:
    st.subheader("Caracterización Óptica")
    col1, col2 = st.columns(2)
    
    # Gráfica 1: Absorbancia vs C
    c_range = np.linspace(0.01, 1.0, 100)
    abs_range = [aplicar_ruido(modelo_optico.absorbancia(c, lambda_nm)) for c in c_range]
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=c_range, y=abs_range, mode='lines', name='Absorbancia Neta'))
    fig1.update_layout(title="Absorbancia neta vs Concentración", xaxis_title="C [mM]", yaxis_title="A", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
    col1.plotly_chart(fig1, use_container_width=True)
    
    # Gráfica 2: Espectro
    lambdas, abs_spectrum = modelo_optico.espectro_completo(c_sim)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=lambdas, y=abs_spectrum, mode='lines', name='Espectro'))
    fig2.add_vrect(x0=1600, x1=1700, fillcolor="lightgray", opacity=0.3, line_width=0)
    fig2.update_layout(title="Espectro NIR", xaxis_title="λ [nm]", yaxis_title="Absorbancia", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
    col2.plotly_chart(fig2, use_container_width=True)

# Tab 2: Microfluídica
with tab2:
    st.subheader("Análisis de Transporte Microfluídico")
    Q_range = np.linspace(1.0, 10.0, 50)
    # FIX: Correct constructor usage (w, h, largo, Q)
    Re_vals = [ModeloMicrofluido(w_um, h_um, largo_mm=5.0, caudal_nL_min=q).numero_reynolds() for q in Q_range]
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=Q_range, y=Re_vals, name='Re'))
    fig3.add_hline(y=1.0, line_dash="dash", line_color="red", name="Límite Laminar (Re=1)")
    fig3.update_layout(title="Número de Reynolds vs Caudal", xaxis_title="Q [nL/min]", yaxis_title="Re", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
    st.plotly_chart(fig3, use_container_width=True)

# Tab 3: Sensibilidad
with tab3:
    st.subheader("Análisis de Sensibilidad Paramétrica")
    L_range = np.linspace(0.1, 2.0, 50)
    # FIX: Create instance per L
    sens_vals = [ModeloBeerLambertNIR(longitud_optica_mm=L).sensibilidad(lambda_nm) for L in L_range]
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=L_range, y=sens_vals, name='dA/dC'))
    fig4.update_layout(title="Sensibilidad (dA/dC) vs Camino óptico (L)", xaxis_title="L [mm]", yaxis_title="Sensibilidad [mM^-1]", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
    st.plotly_chart(fig4, use_container_width=True)

# Tab 4: Inferencia
with tab4:
    st.subheader("Inferencia Clínica y Lotes")
    
    # Inferencia única
    A_medida = st.number_input("Introducir Absorbancia medida (A)", -1.0, 5.0, 0.05)
    if st.button("Estimar Concentración"):
        c_est = modelo_optico.concentracion_inversa(A_medida, lambda_nm)
        clas = modelo_optico.evaluar_clasificacion_fisiologica(c_est)
        st.metric("Concentración Estimada", f"{c_est:.4f} mM")
        st.write(f"**Clasificación:** {clas}")
        
    # Lotes
    st.markdown("---")
    st.subheader("Simulación por Lotes (CSV)")
    uploaded_file = st.file_uploader("Subir archivo CSV con columna de Absorbancia", type="csv")
    if uploaded_file:
        df_lote = pd.read_csv(uploaded_file)
        # Detectar columnas posibles
        col_abs = next((c for c in df_lote.columns if c.lower() in ['absorbancia_medida', 'absorbancia', 'a']), None)
        
        if col_abs:
            df_lote["Glucosa_estimada_mM"] = df_lote[col_abs].apply(lambda a: modelo_optico.concentracion_inversa(a, lambda_nm))
            df_lote["Clasificación_Metabólica"] = df_lote["Glucosa_estimada_mM"].apply(modelo_optico.evaluar_clasificacion_fisiologica)
            
            # Error si existe referencia
            col_ref = next((c for c in df_lote.columns if 'referencia' in c.lower()), None)
            if col_ref:
                df_lote["Error_mM"] = df_lote["Glucosa_estimada_mM"] - df_lote[col_ref]
            
            st.dataframe(df_lote)
            st.download_button("Descargar Resultados", df_lote.to_csv(index=False), "resultados.csv")
        else:
            st.error("No se detectó una columna de absorbancia válida (p. ej. 'Absorbancia', 'A', 'absorbancia_medida').")
