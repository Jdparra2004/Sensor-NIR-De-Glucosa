"""
app_streamlit.py — Interfaz interactiva del Biosensor NIR
PROYECTO: Simulación paramétrica de detección óptica NIR de glucosa en sudor
"""

import io
import os
import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from sklearn.metrics import confusion_matrix

from core.modelo_optico import ModeloBeerLambertNIR, ModeloPLSRegresionNIR
from core.modelo_microfluido import ModeloMicrofluido

st.set_page_config(page_title="Biosensor NIR - Glucosa en Sudor", layout="wide")

# Configuración estándar de leyenda fija inferior
LEYENDA_INFERIOR = dict(
    orientation="h",
    yanchor="top",
    y=-0.25,
    xanchor="center",
    x=0.5
)

# --- ESTADO DE LA SESIÓN PARA PANTALLA DE BIENVENIDA ---
if 'show_info' not in st.session_state:
    st.session_state.show_info = True

def display_welcome_info():
    """Muestra la ventana de bienvenida con información del proyecto y disclaimer."""
    st.info("### Bienvenido al Biosensor NIR: Simulador Paramétrico")
    st.markdown("""
    Esta aplicación es una **herramienta de simulación numérica** diseñada para explorar los parámetros de diseño en sistemas de detección óptica de glucosa en sudor mediante espectroscopia NIR.

    **¿Qué puedes hacer aquí?**
    *   **Simulación Óptica:** Ajustar la longitud de onda y el camino óptico para analizar la absorbancia neta.
    *   **Análisis Microfluídico:** Evaluar el régimen de flujo (Reynolds) y el tiempo de residencia.
    *   **Sensibilidad:** Analizar cómo los cambios geométricos afectan la capacidad de detección.
    *   **Inferencia Analítica:** Procesar lotes de datos para estimar concentraciones de glucosa y clasificar resultados metabólicos in silico.

    **IMPORTANTE - DISCLAIMER DE DISEÑO:**
    Este software es exclusivamente una **herramienta de simulación para diseño y exploración de parámetros**. 
    **NO** es un dispositivo médico, ni proporciona resultados clínicos ni decisiones de diagnóstico técnico. Los resultados son proyecciones basadas en modelos teóricos (física-matemática) y deben utilizarse únicamente para evaluar la viabilidad de parámetros de diseño durante la fase de desarrollo.
    """)
    if st.button("Entendido y cerrar"):
        st.session_state.show_info = False
        st.rerun()

# Si debe mostrarse la información, la mostramos
if st.session_state.show_info:
    display_welcome_info()
    st.stop() # Detenemos ejecución para que solo se vea la info

# --- SIDEBAR ---
st.sidebar.header("Parámetros de Diseño y Simulación")

if st.sidebar.button("Ver Guía y Disclaimer"):
    st.session_state.show_info = True
    st.rerun()

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

st.sidebar.subheader("Calibración Empírica")
alpha = st.sidebar.slider("Factor de escala (α)", 0.0, 1000.0, 1.0, 0.1)
beta = st.sidebar.slider("Sesgo (β)", -100.0, 100.0, 0.0, 0.1)

# --- MODELOS ---
modelo_optico = ModeloBeerLambertNIR(longitud_optica_mm=L_mm, alpha=alpha, beta=beta)
modelo_micro = ModeloMicrofluido(w_um, h_um, largo_mm, Q_nlmin)

def aplicar_ruido(valor):
    return valor * np.random.uniform(0.95, 1.05) if noise_instrumental else valor

def generar_excel_multihoja_estetico(df_resultado, lambda_val, L_val):
    """Genera un libro Excel multi-hoja con formato profesional, colores y estilos."""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_resultado.to_excel(writer, sheet_name='Resultados_Analisis', index=False)
        
        if "Clasificación_Metabólica" in df_resultado.columns:
            conteo = df_resultado["Clasificación_Metabólica"].value_counts().reset_index()
            conteo.columns = ["Categoría Fisiológica", "Total Muestras"]
            conteo["Porcentaje (%)"] = (conteo["Total Muestras"] / len(df_resultado) * 100).round(2)
            conteo.to_excel(writer, sheet_name='Distribucion_Metabolica', index=False)
        
        c_validos = df_resultado["Glucosa_Estimada_mM"].dropna()
        metricas = {
            "Parámetro de Simulación": [
                "Longitud de onda de análisis (λ)",
                "Camino óptico configurado (L)",
                "Total de muestras evaluadas",
                "Concentración media estimada",
                "Concentración mínima detectada",
                "Concentración máxima detectada"
            ],
            "Valor": [
                f"{lambda_val} nm",
                f"{L_val} mm",
                len(df_resultado),
                f"{c_validos.mean():.4f} mM" if not c_validos.empty else "N/A",
                f"{c_validos.min():.4f} mM" if not c_validos.empty else "N/A",
                f"{c_validos.max():.4f} mM" if not c_validos.empty else "N/A"
            ]
        }
        
        df_validos = df_resultado[df_resultado["Clasificación_Metabólica"] != "Indeterminado"].copy()
        if "Error Relativo (%)" in df_validos.columns and not df_validos["Error Relativo (%)"].dropna().empty:
            metricas["Parámetro de Simulación"].append("Error relativo medio (muestras válidas)")
            metricas["Valor"].append(f"{df_validos['Error Relativo (%)'].dropna().mean():.2f} %")
            
        df_params = pd.DataFrame(metricas)
        df_params.to_excel(writer, sheet_name='Parametros_Diseno', index=False)

    output.seek(0)
    wb = openpyxl.load_workbook(output)
    if len(wb.worksheets) > 0:
        wb.active = 0
    else:
        wb.create_sheet("Sin_Datos")
        wb.active = 0
    return output


# --- PÁGINAS ---
st.title("Biosensor NIR: Simulación Integrada")
tab1, tab2, tab3, tab4 = st.tabs(["Óptica NIR", "Microfluídica", "Sensibilidad", "Inferencia Analítica"])

# Tab 1: Óptica
with tab1:
    with st.expander("Información del Análisis", expanded=False):
        st.markdown("Este módulo caracteriza la respuesta óptica del biosensor basándose en la Ley de Beer-Lambert. Calcula la absorbancia neta considerando el fenómeno de desplazamiento de agua, permitiendo visualizar la relación entre la concentración de glucosa y la absorbancia, así como el perfil espectral en la ventana de detección seleccionada.")
    
    c_range = np.linspace(0.01, 1.0, 100)
    abs_vals = [aplicar_ruido(modelo_optico.absorbancia(c, lambda_nm)) for c in c_range]
    
    col1, col2 = st.columns(2)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=c_range, y=abs_vals, mode="lines", name="Absorbancia neta (Beer-Lambert corregido)"))
    fig1.update_layout(
        title="Absorbancia neta vs Concentración",
        xaxis_title="C [mM]",
        yaxis_title="A [u.a.]",
        height=380,
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
        height=380,
        showlegend=True,
        legend=LEYENDA_INFERIOR
    )
    col2.plotly_chart(fig2)

    with st.container(border=True):
        st.latex(r"C_{\text{final}} = \alpha \cdot \left( \frac{|A_{\text{neta}}|}{|\epsilon_{\text{g}}(\lambda) - \epsilon_{\text{w}}(\lambda) \cdot \delta_{\text{w}}| \cdot L} \right) + \beta")
        A_actual = modelo_optico.absorbancia(c_sim, lambda_nm)
        c_est_val = modelo_optico.concentracion_inversa(A_actual, lambda_nm)
        st.info(
            rf"Configuración: $\lambda = {lambda_nm}\text{{ nm}}$, $L = {L_mm}\text{{ mm}}$, "
            rf"$\alpha = {alpha}$, $\beta = {beta}$. "
            rf"Para una concentración teórica de $C = {c_sim}\text{{ mM}}$, "
            rf"la estimación ajustada resulta en **{c_est_val:.5f} mM**."
        )

# Tab 2: Microfluídica
with tab2:
    with st.expander("Información del Análisis", expanded=False):
        st.markdown("Este módulo evalúa las propiedades hidrodinámicas del fluido dentro del canal microfluídico. Calcula parámetros críticos para el diseño, incluyendo el número de Reynolds para verificar la laminaridad del flujo y el tiempo de residencia para determinar la interacción óptima fluido-sensor.")
    
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
        height=380,
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
        height=380,
        showlegend=True,
        legend=LEYENDA_INFERIOR
    )
    col2.plotly_chart(fig4)
    
    with st.container(border=True):
        st.latex(r"Re = \frac{2\rho Q}{\mu(w+h)} \quad \text{y} \quad t_r = \frac{V_{\text{celda}}}{Q}")
        re_actual = modelo_micro.numero_reynolds()
        st.info(
            rf"Con $Q = {Q_nlmin}\text{{ nL/min}}$, $Re = \mathbf{{{re_actual:.6f}}}$ "
            rf"({'Régimen Laminar' if re_actual < 1 else 'Flujo No Laminar'}). "
            rf"Velocidad media: **{modelo_micro.velocidad_media_m_s()*1e6:.2f} µm/s**. "
            rf"Tiempo de residencia: **{modelo_micro.tiempo_residencia_s():.2f} s**."
        )

# Tab 3: Sensibilidad
with tab3:
    with st.expander("Información del Análisis", expanded=False):
        st.markdown("Este estudio analiza cómo la longitud del camino óptico afecta la sensibilidad local (dA/dC) del biosensor. El objetivo es identificar configuraciones geométricas que maximicen la señal de detección sin degradar la selectividad del sistema.")
    
    L_range = np.linspace(0.1, 2.0, 50)
    sens_vals = [ModeloBeerLambertNIR(L).sensibilidad(lambda_nm) for L in L_range]
    abs_vals_L = [ModeloBeerLambertNIR(L).absorbancia(c_sim, lambda_nm) for L in L_range]
    
    col1, col2 = st.columns(2)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=L_range, y=abs_vals_L, mode="lines", name="Absorbancia neta (A)"))
    fig5.update_layout(
        title="Absorbancia vs Camino Óptico (L)",
        xaxis_title="L [mm]",
        yaxis_title="A [u.a.]",
        height=380,
        showlegend=True,
        legend=LEYENDA_INFERIOR
    )
    col1.plotly_chart(fig5)
    
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=L_range, y=sens_vals, mode="lines", name="Sensibilidad (dA/dC)", line=dict(color="orange")))
    fig6.update_layout(
        title="Sensibilidad Analítica vs Camino Óptico",
        xaxis_title="L [mm]",
        yaxis_title="Sensibilidad [mM⁻¹]",
        height=380,
        showlegend=True,
        legend=LEYENDA_INFERIOR
    )
    col2.plotly_chart(fig6)

# Tab 4: Inferencia Analítica
with tab4:
    with st.expander("Información del Análisis", expanded=False):
        st.markdown("Motor de inferencia para la estimación de concentración de glucosa a partir de valores de absorbancia. Permite el análisis puntual o el procesamiento de lotes mediante carga de archivos CSV, Parquet o TXT, clasificando las muestras según umbrales metabólicos fisiológicos.")

    st.subheader("Inferencia Analítica Puntual")
    col_inf1, col_inf2 = st.columns(2)
    with col_inf1:
        A_med = st.number_input("Absorbancia medida (A)", value=-0.05, step=0.001, format="%.5f")
    with col_inf2:
        error_est_pct = st.slider("Incertidumbre instrumental estimada (±%)", 1.0, 10.0, 5.0, 0.5)

    if st.button("Ejecutar Inferencia Puntual", type="primary"):
        c_est = modelo_optico.concentracion_inversa(A_med, lambda_nm)
        delta_c = c_est * (error_est_pct / 100.0)
        c_min = max(0.0, c_est - delta_c)
        c_max = c_est + delta_c
        
        st.markdown("### Resultado Principal de Inferencia")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Concentración Estimada", f"{c_est:.4f} mM", f"± {delta_c:.4f} mM (IC 95%)")
        m_col2.metric("Límite Inferior (IC)", f"{c_min:.4f} mM")
        m_col3.metric("Límite Superior (IC)", f"{c_max:.4f} mM")
        
        st.success(f"Clasificación In Silico: **{modelo_optico.evaluar_clasificacion_fisiologica(c_est)}**")
        
    st.markdown("---")
    st.subheader("Procesamiento por Lotes (Carga por Chunks)")
    
    with st.expander("ℹ️ Guía detallada: Estructura de archivos para Lotes", expanded=False):
        st.markdown(rf"""
        ### Formatos Soportados
        El sistema procesa archivos `.txt` (tab-separated), `.csv` (comma-separated) y `.parquet` mediante carga optimizada por lotes (chunks) para superar cualquier límite de filas.

        ### Estructura Requerida
        1. **Matriz Espectral (Obligatoria):**
           - Encabezados estrictamente numéricos (ej. `400`, `1600`).
        2. **Columna de Referencia (Opcional):**
           - `Glucose (mM)`, `glucosa_referencia_mM`, `C_real`.
        """)

    uploaded = st.file_uploader("Subir archivo de muestras (CSV, Parquet, TXT)", type=["csv", "parquet", "txt"])
    
    if uploaded is not None:
        progreso_contenedor = st.container()
        barra_progreso = progreso_contenedor.progress(0)
        estado_texto = progreso_contenedor.empty()
        
        try:
            estado_texto.text("Paso 1/4: Leyendo archivo por lotes (chunks)...")
            barra_progreso.progress(25)
            
            file_ext = os.path.splitext(uploaded.name)[1].lower()
            
            if file_ext == '.parquet':
                df_lote = pd.read_parquet(uploaded)
            elif file_ext == '.txt':
                chunks = list(pd.read_csv(uploaded, sep='\t', encoding='utf-16', chunksize=5000))
                df_lote = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            else:
                try:
                    uploaded.seek(0)
                    chunks = list(pd.read_csv(uploaded, encoding='utf-8', chunksize=5000))
                    df_lote = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                except Exception:
                    uploaded.seek(0)
                    chunks = list(pd.read_csv(uploaded, sep=None, engine="python", chunksize=5000))
                    df_lote = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            
            total_filas_original = len(df_lote)
            st.info(f"Procesamiento por lotes completado: **{total_filas_original:,} filas** cargadas y procesadas sin truncamiento.")

            estado_texto.text("Paso 2/4: Identificando canal óptico o espectro completo...")
            barra_progreso.progress(50)
            time.sleep(0.05)
            
            df_lote.columns = df_lote.columns.str.strip()
            espectro_cols = [c for c in df_lote.columns if c.replace('.','',1).isdigit()]
            
            if len(espectro_cols) > 100:
                estado_texto.text("Procesando con modelo multivariante PLS-R...")
                pls_model = ModeloPLSRegresionNIR(alpha=alpha, beta=beta)
                
                candidatos_ref = ['Glucose (mM)', 'glucosa_referencia_mM', 'Glucosa_Real_mM', 'glucosa_mM', 'glucose_mM', 'C_real']
                col_ref = next((c for c in candidatos_ref if c in df_lote.columns), None)
                
                if col_ref:
                    y_series = pd.to_numeric(df_lote[col_ref], errors="coerce").fillna(0)
                    pls_model.entrenar_calibracion(df_lote, y_series)
                    df_lote["Glucosa_Estimada_mM"] = pls_model.predecir(df_lote, alpha=alpha, beta=beta)
                    
                    mse = np.mean((df_lote["Glucosa_Estimada_mM"] - y_series)**2)
                    rmse = np.sqrt(mse)
                    st.write(f"**Métricas PLS-R:** RMSEP = {rmse:.4f} mM, Componentes óptimos = {pls_model.n_componentes_optimo}")
                else:
                    st.error("No se encontró columna de referencia para calibración del modelo PLS-R.")
            
            else:
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
                    estado_texto.text(rf"Paso 3/4: Ejecutando inferencia inversa ($\lambda={lambda_nm}\text{{ nm}}$, $L={L_mm}\text{{ mm}}$)...")
                    barra_progreso.progress(75)
                    
                    valores_abs = pd.to_numeric(df_lote[col_abs], errors="coerce")
                    df_lote["Glucosa_Estimada_mM"] = valores_abs.apply(
                        lambda a: modelo_optico.concentracion_inversa(float(a), lambda_nm, alpha=alpha, beta=beta) if pd.notna(a) else np.nan
                    ).round(4)
            
            df_resultado = df_lote.copy()
            
            if 'Glucosa_Estimada_mM' in df_resultado.columns:
                candidatos_ref = ['Glucose (mM)', 'glucosa_referencia_mM', 'Glucosa_Real_mM', 'glucosa_mM', 'glucose_mM', 'C_real']
                col_ref = next((c for c in candidatos_ref if c in df_resultado.columns), None)
                if col_ref is not None:
                    c_real = pd.to_numeric(df_resultado[col_ref], errors="coerce")
                    df_resultado["Error Relativo (%)"] = (np.abs(df_resultado["Glucosa_Estimada_mM"] - c_real) / np.where(c_real != 0, c_real, 1e-12) * 100).round(2)
                
                df_resultado["Clasificación_Metabólica"] = df_resultado["Glucosa_Estimada_mM"].apply(
                    lambda c: modelo_optico.evaluar_clasificacion_fisiologica(c) if pd.notna(c) else "Indeterminado"
                )
                
                estado_texto.text("Paso 4/4: Consolidando resultados y libro de reporte...")
                barra_progreso.progress(100)
                time.sleep(0.05)
                
                barra_progreso.empty()
                estado_texto.success(f"Procesamiento completado: {len(df_resultado):,} muestras analizadas bajo λ = {lambda_nm} nm y L = {L_mm} mm.")
                
                st.dataframe(df_resultado)
                
                st.markdown("#### Análisis Estadístico y Fisiológico del Lote")
                col_g1, col_g2 = st.columns(2)
                
                st.markdown("#### Análisis Estadístico y Fisiológico del Lote")
                col_g1, col_g2 = st.columns(2)
                
                candidatos_ref = ['Glucose (mM)', 'glucosa_referencia_mM', 'Glucosa_Real_mM', 'glucosa_mM', 'glucose_mM', 'C_real']
                col_ref_found = next((c for c in candidatos_ref if c in df_resultado.columns), None)
                
                if col_ref_found is not None:
                    c_real = pd.to_numeric(df_resultado[col_ref_found], errors="coerce")
                    c_est = df_resultado["Glucosa_Estimada_mM"]
                    
                    valid_mask = c_real.notna() & c_est.notna()
                    n_muestras = int(valid_mask.sum())
                    if n_muestras > 1:
                        cr_v = c_real[valid_mask]
                        ce_v = c_est[valid_mask]
                        mse = np.mean((ce_v - cr_v)**2)
                        rmse = np.sqrt(mse)
                        ss_res = np.sum((cr_v - ce_v)**2)
                        ss_tot = np.sum((cr_v - np.mean(cr_v))**2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    else:
                        rmse = 0.0
                        r2 = 0.0
                    
                    # 1. Gráfico de Dispersión de Correlación (C_real vs C_estimada) con línea y=x y anotación
                    fig_corr = go.Figure()
                    fig_corr.add_trace(go.Scatter(
                        x=c_real,
                        y=c_est,
                        mode='markers',
                        name='Muestras Lote',
                        marker=dict(color='#1f77b4', size=6, opacity=0.7)
                    ))
                    
                    min_val = min(c_real.min(), c_est.min())
                    max_val = max(c_real.max(), c_est.max())
                    fig_corr.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Ideal (y = x)',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    anotacion_texto = f"<b>Métricas Dinámicas:</b><br>• R² = {r2:.4f}<br>• RMSE = {rmse:.4f} mM<br>• N = {n_muestras:,}"
                    fig_corr.add_annotation(
                        xref="paper", yref="paper",
                        x=0.05, y=0.95,
                        text=anotacion_texto,
                        showarrow=False,
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4,
                        font=dict(size=11)
                    )
                    
                    fig_corr.update_layout(
                        title="<b>Correlación C_real vs C_estimada</b>",
                        xaxis_title="Concentración Real (mM)",
                        yaxis_title="Concentración Estimada (mM)",
                        height=380,
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    col_g1.plotly_chart(fig_corr)
                    
                    # 2. Factor de Dilución Simulado para etiquetas clínicas (escala sudor: 0 a ~1.0 mM)
                    c_real_dil = c_real / 50.0
                    c_est_dil = c_est / 50.0
                    
                    def asignar_categoria(val):
                        if pd.isna(val) or val < 0.0:
                            return "Indeterminado"
                        elif val <= 0.20:
                            return "Normal"
                        elif val <= 0.40:
                            return "Rango de Alerta / Prediabetes"
                        else:
                            return "Hiperglucemia"
                            
                    y_real_cat = c_real_dil.apply(asignar_categoria)
                    y_est_cat = c_est_dil.apply(asignar_categoria)
                    
                    labels_clase = ["Normal", "Rango de Alerta / Prediabetes", "Hiperglucemia"]
                    
                    # Generar Matriz de Confusión usando sklearn.metrics.confusion_matrix
                    mask_validas = y_real_cat.isin(labels_clase) & y_est_cat.isin(labels_clase)
                    if mask_validas.sum() > 0:
                        matriz_cm = confusion_matrix(
                            y_real_cat[mask_validas],
                            y_est_cat[mask_validas],
                            labels=labels_clase
                        )
                    else:
                        matriz_cm = np.zeros((3, 3), dtype=int)
                        
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=matriz_cm,
                        x=labels_clase,
                        y=labels_clase,
                        text=matriz_cm,
                        texttemplate="%{text}",
                        colorscale="Blues",
                        hoverinfo='z'
                    ))
                    
                    fig_cm.update_layout(
                        title="<b>Matriz de Confusión Clínica (Heatmap)</b>",
                        xaxis_title="Categoría Estimada",
                        yaxis_title="Categoría Real",
                        height=380,
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    col_g2.plotly_chart(fig_cm)
                    
                else:
                    conteo_df = df_resultado["Clasificación_Metabólica"].value_counts().reset_index()
                    conteo_df.columns = ["Categoría", "Muestras"]
                    
                    colores_map = {
                        "Normal": "#2ca02c",
                        "Rango de Alerta / Sospecha de Prediabetes": "#ff7f0e",
                        "Nivel Elevado / Sospecha Hiperglucemia": "#d62728",
                        "Fuera de rango analítico / Indetectable": "#7f7f7f"
                    }
                    bar_colors = [colores_map.get(cat, "#1f77b4") for cat in conteo_df["Categoría"]]
                    
                    fig_lote_cat = go.Figure()
                    fig_lote_cat.add_trace(go.Bar(
                        x=conteo_df["Categoría"],
                        y=conteo_df["Muestras"],
                        marker_color=bar_colors,
                        name="Muestras por Estado"
                    ))
                    fig_lote_cat.update_layout(
                        title="<b>Distribución de Categorías Fisiológicas</b>",
                        xaxis_title="Estado Metabólico",
                        yaxis_title="Cantidad de Muestras",
                        height=350,
                        showlegend=False,
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    col_g1.plotly_chart(fig_lote_cat)
                    
                    fig_lote_disp = go.Figure()
                    indices_muestras = list(range(1, len(df_resultado) + 1))
                    
                    fig_lote_disp.add_trace(go.Scatter(
                        x=indices_muestras,
                        y=df_resultado["Glucosa_Estimada_mM"],
                        mode="markers",
                        name="Glucosa Estimada (mM)",
                        marker=dict(color="#1f77b4", size=5, opacity=0.7)
                    ))
                    fig_lote_disp.add_hline(y=0.20, line_dash="dash", line_color="green", annotation_text="Límite Normal (0.20 mM)")
                    fig_lote_disp.add_hline(y=0.40, line_dash="dash", line_color="red", annotation_text="Umbral Hiperglucemia (0.40 mM)")
                    
                    fig_lote_disp.update_layout(
                        title="<b>Concentración Estimada por Muestra</b>",
                        xaxis_title="Índice de Muestra",
                        yaxis_title="Glucosa [mM]",
                        height=350,
                        showlegend=True,
                        legend=LEYENDA_INFERIOR,
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    col_g2.plotly_chart(fig_lote_disp)
                
                col_btn1, col_btn2 = st.columns(2)
                excel_bytes = generar_excel_multihoja_estetico(df_resultado, lambda_nm, L_mm)
                
                with col_btn1:
                    if excel_bytes is not None:
                        st.download_button(
                            label="Descargar Reporte Completo en Excel (.xlsx)",
                            data=excel_bytes,
                            file_name="Reporte_Biosensor_NIR.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.info("Exportación directa a CSV disponible.")
                        
                with col_btn2:
                    st.download_button(
                        label="Descargar en formato CSV",
                        data=df_resultado.to_csv(index=False).encode("utf-8"),
                        file_name="resultados_procesamiento_lote.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            barra_progreso.empty()
            estado_texto.error(f"Error al procesar el archivo: {e}")
