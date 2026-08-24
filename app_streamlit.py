"""
MÓDULO: app_streamlit.py
PROYECTO: SIMULACIÓN PARAMÉTRICA Y CARACTERIZACIÓN DE DESEMPEÑO DE UN SENSOR ÓPTICO NIR Y TRANSPORTE MICROFLUIDÍCO
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.modelo_optico import ModeloBeerLambertNIR
from core.modelo_microfluido import ModeloMicrofluido

st.set_page_config(page_title="Biosensor NIR - Glucosa en Sudor", layout="wide")

# Sidebar
st.sidebar.header("⚙️ Configuración del Sistema")
lambda_nm = st.sidebar.slider("λ [nm]", 1000, 1700, 1600, step=10)
L_mm = st.sidebar.slider("L [mm]", 0.1, 2.0, 1.0, step=0.1)

# Models
modelo_optico = ModeloBeerLambertNIR(longitud_optica_mm=L_mm)
modelo_micro = ModeloMicrofluido(ancho_um=200, alto_um=50, caudal_nL_min=5.0)

# Main Title
st.title("🔬 Biosensor NIR: Glucosa en Sudor")

tab1, tab2, tab3, tab4 = st.tabs(["Óptica NIR", "Microfluídica", "Sensibilidad", "Inferencia Clínica"])

# Tab 1: Óptica
with tab1:
    st.subheader("Caracterización Óptica")
    col1, col2 = st.columns(2)
    c_range = np.linspace(0.01, 1.0, 100)
    
    # Gráfica A vs C
    abs_range = [modelo_optico.absorbancia(c, lambda_nm) for c in c_range]
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=c_range, y=abs_range, name='A neta'))
    fig1.update_layout(title="Absorbancia neta vs Concentración", xaxis_title="C [mM]", yaxis_title="A")
    col1.plotly_chart(fig1, use_container_width=True)
    
    # Gráfica Espectro
    lambdas, abs_spectrum = modelo_optico.espectro_completo(0.5)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=lambdas, y=abs_spectrum, name='Absorbancia'))
    fig2.add_vrect(x0=1600, x1=1700, fillcolor="lightgray", opacity=0.3)
    fig2.update_layout(title="Espectro NIR", xaxis_title="λ [nm]", yaxis_title="A")
    col2.plotly_chart(fig2, use_container_width=True)
    
    st.latex(r"A_{\text{neta}}(\lambda) = (\epsilon_g(\lambda) - \epsilon_w(\lambda)\cdot\delta_w)\cdot C\cdot L")
    st.info("La linealidad responde a la Ley de Beer-Lambert. La pendiente negativa en 1600-1700nm se debe al desplazamiento volumétrico de agua.")

# Tab 2: Microfluídica
with tab2:
    st.subheader("Análisis Microfluídico")
    Q_range = np.linspace(1.0, 10.0, 50)
    Re_vals = [ModeloMicrofluido(200, 50, q).numero_reynolds() for q in Q_range]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=Q_range, y=Re_vals, name='Re'))
    fig3.add_hline(y=1.0, line_dash="dash", line_color="red")
    st.plotly_chart(fig3, use_container_width=True)
    
    st.latex(r"Re = \frac{2\rho Q}{\mu (w + h)}, \quad t_r = \frac{w \cdot h \cdot largo}{Q}")
    st.markdown("Para caudales fisiológicos, $Re \ll 1$, garantizando régimen laminar estricto.")

# Tab 3: Sensibilidad
with tab3:
    st.subheader("Análisis de Sensibilidad")
    L_range = np.linspace(0.1, 2.0, 50)
    sens_vals = [ModeloBeerLambertNIR(L).sensibilidad(lambda_nm) for L in L_range]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=L_range, y=sens_vals, name='dA/dC'))
    st.plotly_chart(fig4, use_container_width=True)
    
    st.latex(r"\text{Sensibilidad} = \frac{\partial A}{\partial C} = (\epsilon_g(\lambda) - \epsilon_w(\lambda)\cdot\delta_w)\cdot L")
    st.info("La sensibilidad analítica aumenta linealmente con L, optimizando la relación señal-ruido.")

# Tab 4: Inferencia
with tab4:
    st.subheader("Procesamiento por Lotes")
    uploaded_file = st.file_uploader("Subir CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'absorbancia_1600nm' in df.columns:
            df["Glucosa_Estimada_mM"] = df['absorbancia_1600nm'].apply(lambda a: modelo_optico.concentracion_inversa(a, 1600))
            df["Error_Relativo_%"] = ((df["Glucosa_Estimada_mM"] - df["glucosa_referencia_mM"]) / df["glucosa_referencia_mM"]) * 100
            df["Clasificación_Metabólica"] = df["Glucosa_Estimada_mM"].apply(modelo_optico.evaluar_clasificacion_fisiologica)
            st.dataframe(df)
            st.download_button("Descargar Resultados", df.to_csv(index=False), "resultados_procesados.csv")
        else:
            st.error("CSV debe tener 'absorbancia_1600nm'")
