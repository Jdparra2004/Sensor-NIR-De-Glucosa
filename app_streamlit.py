import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.modelo_optico import ModeloBeerLambertNIR
from core.modelo_microfluido import ModeloMicrofluido

st.set_page_config(page_title="Biosensor NIR - Glucosa en Sudor", layout="wide")

# --- SIDEBAR COMPLETO ---
st.sidebar.header("⚙️ Parámetros de Diseño y Simulación")
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
st.title("🔬 Biosensor NIR: Simulación Integrada")
tab1, tab2, tab3, tab4 = st.tabs(["Óptica NIR", "Microfluídica", "Sensibilidad", "Inferencia Clínica"])

# Tab 1: Óptica
with tab1:
    c_range = np.linspace(0.01, 1.0, 100)
    abs_vals = [aplicar_ruido(modelo_optico.absorbancia(c, lambda_nm)) for c in c_range]
    
    col1, col2 = st.columns(2)
    fig1 = go.Figure().add_trace(go.Scatter(x=c_range, y=abs_vals, name='Absorbancia neta con corrección'))
    fig1.update_layout(title="Absorbancia neta vs Concentración", xaxis_title="C [mM]", yaxis_title="A")
    col1.plotly_chart(fig1)

    lambdas, spect = modelo_optico.espectro_completo(c_sim)
    fig2 = go.Figure().add_trace(go.Scatter(x=lambdas, y=spect, name='Espectro'))
    fig2.add_vrect(x0=1600, x1=1700, fillcolor="lightgray", opacity=0.3)
    fig2.update_layout(title="Espectro NIR", xaxis_title="λ [nm]", yaxis_title="A")
    col2.plotly_chart(fig2)

    st.latex(r"A_{\text{neta}}(\lambda) = (\epsilon_g(\lambda) - \epsilon_w(\lambda) \cdot \delta_w) \cdot C \cdot L")
    A_actual = modelo_optico.absorbancia(c_sim, lambda_nm)
    st.info(f"A λ = {lambda_nm} nm, L = {L_mm} mm y C = {c_sim} mM, la absorbancia neta es de **{A_actual:.5f} u.a.** La pendiente negativa responde al desplazamiento volumétrico del agua ($\delta_w \approx 6.15$): al disolverse glucosa, se excluye solvente, disminuyendo la absorción del agua.")

# Tab 2: Microfluídica
with tab2:
    Q_range = np.linspace(1.0, 10.0, 50)
    Re_vals = [ModeloMicrofluido(w_um, h_um, largo_mm, q).numero_reynolds() for q in Q_range]
    tr_vals = [ModeloMicrofluido(w_um, h_um, largo_mm, q).tiempo_residencia_s() for q in Q_range]
    
    col1, col2 = st.columns(2)
    fig3 = go.Figure().add_trace(go.Scatter(x=Q_range, y=Re_vals, name='Re'))
    fig3.add_hline(y=1.0, line_dash="dash", line_color="red")
    fig3.update_layout(title="Número de Reynolds vs Caudal", xaxis_title="Q [nL/min]", yaxis_title="Re")
    col1.plotly_chart(fig3)
    
    fig4 = go.Figure().add_trace(go.Scatter(x=Q_range, y=tr_vals, name='t_r'))
    fig4.update_layout(title="Tiempo de residencia vs Caudal", xaxis_title="Q [nL/min]", yaxis_title="t_r [s]")
    col2.plotly_chart(fig4)
    
    st.latex(r"Re = \frac{2\rho Q}{\mu(w+h)} \quad \text{y} \quad t_r = \frac{V_{\text{celda}}}{Q}")
    re_actual = modelo_micro.numero_reynolds()
    st.info(f"Con Q={Q_nlmin} nL/min, Re = **{re_actual:.6f}** ({'Régimen Laminar' if re_actual < 1 else 'Flujo No Laminar'}). Velocidad media: **{modelo_micro.velocidad_media_m_s()*1e6:.2f} µm/s**. Tiempo de residencia: **{modelo_micro.tiempo_residencia_s():.2f} s**.")

# Tab 3: Sensibilidad
with tab3:
    L_range = np.linspace(0.1, 2.0, 50)
    sens_vals = [ModeloBeerLambertNIR(L).sensibilidad(lambda_nm) for L in L_range]
    abs_vals_L = [ModeloBeerLambertNIR(L).absorbancia(c_sim, lambda_nm) for L in L_range]
    
    col1, col2 = st.columns(2)
    fig5 = go.Figure().add_trace(go.Scatter(x=L_range, y=sens_vals, name='dA/dC'))
    fig5.update_layout(title="Sensibilidad vs Camino óptico", xaxis_title="L [mm]", yaxis_title="dA/dC")
    col1.plotly_chart(fig5)
    
    fig6 = go.Figure().add_trace(go.Scatter(x=L_range, y=abs_vals_L, name='Absorbancia'))
    fig6.update_layout(title="Absorbancia vs Camino óptico", xaxis_title="L [mm]", yaxis_title="A")
    col2.plotly_chart(fig6)
    
    st.latex(r"\text{Sensibilidad} = \frac{\partial A}{\partial C} = (\epsilon_g(\lambda) - \epsilon_w(\lambda) \cdot \delta_w) \cdot L")
    st.info("La sensibilidad aumenta linealmente con el camino óptico L, permitiendo amplificar la señal neta sin afectar la selectividad del desplazamiento del solvente.")

# Tab 4: Inferencia
with tab4:
    st.subheader("Inferencia Clínica")
    A_med = st.number_input("Absorbancia medida (A)", value=-0.05, step=0.001)
    if st.button("Estimar"):
        c_est = modelo_optico.concentracion_inversa(A_med, lambda_nm)
        st.write(f"Concentración: {c_est:.4f} mM - {modelo_optico.evaluar_clasificacion_fisiologica(c_est)}")
        
    st.markdown("---")
    uploaded = st.file_uploader("Subir CSV de lotes", type="csv")
    if uploaded:
        df_lote = pd.read_csv(uploaded)
        if 'absorbancia_1600nm' in df_lote.columns:
            df_lote["Glucosa_Estimada_mM"] = df_lote['absorbancia_1600nm'].apply(lambda a: modelo_optico.concentracion_inversa(a, 1600))
            if 'glucosa_referencia_mM' in df_lote.columns:
                df_lote["Error_%"] = ((df_lote["Glucosa_Estimada_mM"] - df_lote['glucosa_referencia_mM']) / df_lote['glucosa_referencia_mM']) * 100
            df_lote["Clasificación_Metabólica"] = df_lote["Glucosa_Estimada_mM"].apply(modelo_optico.evaluar_clasificacion_fisiologica)
            st.dataframe(df_lote)
            st.download_button("Descargar CSV", df_lote.to_csv(index=False), "resultados.csv")
