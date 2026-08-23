import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from core.modelo_optico import ModeloBeerLambertNIR
from core.modelo_microfluido import ModeloMicrofluido

# Configuración de la página
st.set_page_config(layout="wide", page_title="Simulador NIR Glucosa")

st.title("🔬 Simulador de Biosensor NIR para Glucosa en Sudor")

# --- SIDEBAR: CONTROLES ---
st.sidebar.header("⚙️ Configuración")

st.sidebar.subheader("Óptica")
lambda_nm = st.sidebar.slider("Longitud de onda (nm)", 1000, 1700, 1600)
L_mm = st.sidebar.slider("Camino óptico (mm)", 0.1, 10.0, 1.0)
incluir_agua = st.sidebar.checkbox("Corrección por desplazamiento de agua", True)

st.sidebar.subheader("Microfluídica")
w_um = st.sidebar.slider("Ancho del canal (µm)", 50, 500, 200)
h_um = st.sidebar.slider("Alto del canal (µm)", 10, 200, 50)
Q_nlmin = st.sidebar.slider("Caudal (nL/min)", 1, 100, 10)

# Instanciación de Modelos
modelo_optico = ModeloBeerLambertNIR(longitud_optica_mm=L_mm, incluir_desplazamiento_agua=incluir_agua)
modelo_micro = ModeloMicrofluido(ancho_um=w_um, alto_um=h_um, caudal_nL_min=Q_nlmin)

# Métricas rápidas
st.sidebar.subheader("Métricas del Canal")
re = modelo_micro.numero_reynolds()
tr = modelo_micro.tiempo_residencia_s()
st.sidebar.metric("Número de Reynolds (Re)", f"{re:.4f}")
st.sidebar.metric("Tiempo de residencia (s)", f"{tr:.2f}")
if modelo_micro.es_regimen_laminar():
    st.sidebar.success("Régimen: Laminar")
else:
    st.sidebar.warning("Régimen: No laminar")

# --- CUERPO PRINCIPAL: TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Calibración", "Espectro NIR", "Sensibilidad", 
    "Análisis Hidrodinámico", "Diagnóstico Clínico", "Procesamiento Lote"
])

# Tab 1: Curva de Calibración
with tab1:
    st.subheader("Curva de Calibración (A vs C)")
    c_range = np.linspace(0.01, 1.0, 50)
    a_values = modelo_optico.barrido_concentraciones(c_range, lambda_nm)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c_range, y=a_values, mode='lines', name='Absorbancia'))
    fig.update_layout(xaxis_title="Concentración Glucosa [mM]", yaxis_title="Absorbancia")
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Espectro NIR Completo
with tab2:
    st.subheader("Espectro NIR Completo")
    fig, ax = plt.subplots()
    for c in [0.1, 0.5, 1.0]:
        lambdas, A = modelo_optico.espectro_completo(c)
        ax.plot(lambdas, A, label=f"C={c} mM")
    ax.set_xlabel("Longitud de onda [nm]")
    ax.set_ylabel("Absorbancia")
    ax.legend()
    st.pyplot(fig)

# Tab 3: Sensibilidad
with tab3:
    st.subheader("Sensibilidad y Camino Óptico")
    # ... (Implementación de visualización)
    st.write("Visualización pendiente de la sensibilidad dA/dC.")

# Tab 4: Análisis Hidrodinámico
with tab4:
    st.subheader("Análisis Hidrodinámico")
    # ... (Implementación de visualización de Re y Tr vs Q)
    st.write("Visualización pendiente del análisis hidrodinámico.")

# Tab 5: Diagnóstico
with tab5:
    st.subheader("Diagnóstico Clínico Individual")
    A_medida = st.number_input("Absorbancia medida (A)", 0.0, 1.0, 0.05)
    if A_medida > 0:
        c_calc = modelo_optico.concentracion_inversa(A_medida, lambda_nm)
        riesgo = modelo_optico.evaluar_riesgo_clinico(c_calc)
        st.write(f"Concentración estimada: {c_calc:.3f} mM")
        st.write(f"Riesgo: {riesgo}")

# Tab 6: Procesamiento de Dataset
with tab6:
    st.subheader("Procesamiento de Muestras en Lote")
    uploaded_file = st.file_uploader("Subir CSV", type="csv")
    
    # Datos precargados
    df_base = pd.DataFrame({"Absorbancia": [0.02, 0.05, 0.08, 0.15]})
    
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = st.data_editor(df_base)
        
    if st.button("Procesar Muestras"):
        resultados = []
        for _, row in df_input.iterrows():
            c = modelo_optico.concentracion_inversa(row["Absorbancia"], lambda_nm)
            resultados.append({"Absorbancia": row["Absorbancia"], "Glucosa [mM]": c})
            
        df_res = pd.DataFrame(resultados)
        st.dataframe(df_res)
        st.download_button("Descargar Reporte", df_res.to_csv(index=False), "reporte.csv")
