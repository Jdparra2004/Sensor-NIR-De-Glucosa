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
lambda_nm = st.sidebar.slider("Longitud de onda (nm)", 1000, 1700, 1600, step=50)
L_mm = st.sidebar.slider("Camino óptico (mm)", 0.1, 10.0, 1.0, step=0.1)
incluir_agua = st.sidebar.checkbox("Corrección por desplazamiento de agua", True)
c_min = st.sidebar.number_input("C_min (mM)", 0.01, 1.0, 0.01, step=0.01)
c_max = st.sidebar.number_input("C_max (mM)", 0.01, 1.0, 1.0, step=0.01)

st.sidebar.subheader("Microfluídica")
w_um = st.sidebar.slider("Ancho del canal (µm)", 50, 500, 200, step=10)
h_um = st.sidebar.slider("Alto del canal (µm)", 10, 200, 50, step=10)
Q_nlmin = st.sidebar.slider("Caudal (nL/min)", 1, 100, 10, step=1)

# Instanciación de Modelos
modelo_optico = ModeloBeerLambertNIR(longitud_optica_mm=L_mm, incluir_desplazamiento_agua=incluir_agua)
modelo_micro = ModeloMicrofluido(ancho_um=w_um, alto_um=h_um, caudal_nL_min=Q_nlmin)

# Métricas rápidas
st.sidebar.subheader("Métricas del Canal")
re = modelo_micro.numero_reynolds()
tr = modelo_micro.tiempo_residencia_s()
vm = modelo_micro.velocidad_media_m_s()
st.sidebar.metric("Número de Reynolds (Re)", f"{re:.4f}")
st.sidebar.metric("Velocidad media (m/s)", f"{vm:.6f}")
st.sidebar.metric("Tiempo de residencia (s)", f"{tr:.2f}")

if re < 1.0:
    st.sidebar.success("Régimen: Laminar (Re < 1.0)")
else:
    st.sidebar.warning("Régimen: No laminar (Re >= 1.0)")

# --- CUERPO PRINCIPAL: TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Calibración", "Espectro NIR", "Sensibilidad", 
    "Hidrodinámica", "Corroboración Fisiológica", "Procesamiento Lote"
])

# Tab 1: Curva de Calibración
with tab1:
    st.subheader("Curva de Calibración (A vs C)")
    c_range = np.linspace(0.01, 1.0, 50)
    
    # Comparativa agua
    modelo_sin_agua = ModeloBeerLambertNIR(longitud_optica_mm=L_mm, incluir_desplazamiento_agua=False)
    a_con = modelo_optico.barrido_concentraciones(c_range, lambda_nm)
    a_sin = modelo_sin_agua.barrido_concentraciones(c_range, lambda_nm)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c_range, y=a_con, mode='lines', name='Con corrección agua'))
    fig.add_trace(go.Scatter(x=c_range, y=a_sin, mode='lines', name='Sin corrección agua'))
    fig.update_layout(xaxis_title="Concentración Glucosa [mM]", yaxis_title="Absorbancia")
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Espectro NIR Completo
with tab2:
    st.subheader("Espectro NIR Completo")
    fig, ax = plt.subplots(figsize=(10, 5))
    concentraciones = [c_min, (c_min+c_max)/2, c_max]
    for c in concentraciones:
        lambdas, A = modelo_optico.espectro_completo(c)
        ax.plot(lambdas, A, label=f"C={c:.2f} mM")
    
    ax.axvspan(1600, 1700, color='gray', alpha=0.3, label='Banda de interés (1600-1700nm)')
    ax.set_xlabel("Longitud de onda [nm]")
    ax.set_ylabel("Absorbancia")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# Tab 3: Sensibilidad
with tab3:
    st.subheader("Análisis de Sensibilidad")
    L_range = np.linspace(0.1, 10.0, 50)
    C_ref = 0.5
    
    A_vs_L = [modelo_optico.absorbancia(C_ref, lambda_nm) for _ in L_range] # Placeholder lógica
    # Sensibilidad aproximada: dA/dC = eps_g * L (si despreciamos agua)
    sens_vs_L = [modelo_optico.sensibilidad(lambda_nm) for _ in L_range]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    ax1.plot(L_range, A_vs_L)
    ax1.axvline(L_mm, color='r', linestyle='--')
    ax1.set_title("Absorbancia vs Camino óptico")
    
    ax2.plot(L_range, sens_vs_L)
    ax2.axvline(L_mm, color='r', linestyle='--')
    ax2.set_title("Sensibilidad dA/dC [mM^-1]")
    
    st.pyplot(fig)
    plt.close(fig)

# Tab 4: Análisis Hidrodinámico
with tab4:
    st.subheader("Análisis Hidrodinámico")
    Q_range = np.linspace(1, 100, 50)
    Re_range = [ModeloMicrofluido(w_um, h_um, q).numero_reynolds() for q in Q_range]
    Tr_range = [ModeloMicrofluido(w_um, h_um, q).tiempo_residencia_s() for q in Q_range]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    ax1.plot(Q_range, Re_range)
    ax1.axhline(1.0, color='r', linestyle='-')
    ax1.axvline(Q_nlmin, color='g', linestyle='--')
    ax1.set_title("Re vs Caudal")
    
    ax2.plot(Q_range, Tr_range)
    ax2.axvline(Q_nlmin, color='g', linestyle='--')
    ax2.set_title("Tiempo de residencia vs Caudal")
    
    st.pyplot(fig)
    plt.close(fig)

# Tab 5: Corroboración Fisiológica
with tab5:
    st.subheader("Corroboración Fisiológica de Muestra")
    A_medida = st.number_input("Absorbancia medida (A)", 0.0, 2.0, 0.05)
    
    if A_medida > 0:
        c_calc = modelo_optico.concentracion_inversa(A_medida, lambda_nm)
        clasificacion = modelo_optico.evaluar_clasificacion_fisiologica(c_calc)
        
        st.metric("Concentración Estimada", f"{c_calc:.4f} mM")
        st.info(f"Clasificación: {clasificacion}")
        
        # Gráfica semáforo
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.barh([0], [c_calc], color='green' if c_calc <= 0.2 else 'orange' if c_calc <= 0.4 else 'red')
        ax.set_xlim(0, 1.0)
        st.pyplot(fig)
        plt.close(fig)

# Tab 6: Procesamiento de Dataset
with tab6:
    st.subheader("Procesamiento de Muestras en Lote / Validación In Silico")
    uploaded_file = st.file_uploader("Subir CSV de muestras (formato: id_muestra, lambda_nm, absorbancia_medida, glucosa_referencia_mM, interferente_mM)", type="csv")
    
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        
        if st.button("Procesar Lote"):
            # Procesamiento fila a fila
            resultados = []
            
            # Agrupar por muestra si hay múltiples lambda por muestra
            for muestra_id, grupo in df_input.groupby('id_muestra'):
                # Usar la primera lambda disponible para el cálculo (o la lambda configurada)
                # Opcional: promediar o considerar múltiples lambda si el modelo lo soportara
                row = grupo.iloc[0]
                A = row['absorbancia_medida']
                c_ref = row['glucosa_referencia_mM']
                
                # Inversa del modelo físico
                c_est = modelo_optico.concentracion_inversa(A, lambda_nm)
                
                resultados.append({
                    "id_muestra": muestra_id,
                    "Glucosa Referencia [mM]": c_ref,
                    "Glucosa Estimada [mM]": c_est,
                    "Error Relativo": abs(c_ref - c_est) / c_ref if c_ref != 0 else 0,
                    "Clasificación": modelo_optico.evaluar_clasificacion_fisiologica(c_est)
                })
            
            df_res = pd.DataFrame(resultados)
            
            # Métricas
            rmse = np.sqrt(((df_res["Glucosa Referencia [mM]"] - df_res["Glucosa Estimada [mM]"])**2).mean())
            st.metric("RMSE [mM]", f"{rmse:.4f}")
            
            st.dataframe(df_res)
            
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar Reporte Validación", csv, "reporte_validacion.csv", "text/csv")
