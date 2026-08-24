"""
app_streamlit.py — Interfaz interactiva del Biosensor NIR
PROYECTO: Simulación paramétrica de detección óptica NIR de glucosa en sudor
"""

import io
import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.modelo_optico import ModeloBeerLambertNIR
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

def generar_excel_multihoja(df_resultado, lambda_val, L_val):
    """Genera un archivo Excel estructurado en memoria con múltiples hojas de forma segura."""
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Hoja 1: Datos Detallados
            df_resultado.to_excel(writer, sheet_name='Datos_Procesados', index=False)
            
            # Hoja 2: Resumen de Clasificación
            if "Clasificación_Metabólica" in df_resultado.columns:
                conteo = df_resultado["Clasificación_Metabólica"].value_counts().reset_index()
                conteo.columns = ["Categoría Fisiológica", "Total Muestras"]
                conteo["Porcentaje (%)"] = (conteo["Total Muestras"] / len(df_resultado) * 100).round(2)
                conteo.to_excel(writer, sheet_name='Resumen_Clasificacion', index=False)
            
            # Hoja 3: Parámetros del Ensayo Óptico
            c_validos = df_resultado["Glucosa_Estimada_mM"].dropna()
            metricas = {
                "Parámetro": [
                    "Longitud de onda aplicada (λ)",
                    "Camino óptico aplicado (L)",
                    "Total de muestras analizadas",
                    "Concentración media estimada",
                    "Concentración mínima",
                    "Concentración máxima"
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
            if "Error_%" in df_resultado.columns and not df_resultado["Error_%"].dropna().empty:
                metricas["Parámetro"].append("Error relativo promedio")
                metricas["Valor"].append(f"{df_resultado['Error_%'].dropna().mean():.2f} %")
                
            df_params = pd.DataFrame(metricas)
            df_params.to_excel(writer, sheet_name='Parametros_Simulacion', index=False)
            
        output.seek(0)
        return output.getvalue()
    except Exception:
        return None

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
        st.latex(r"A_{\text{neta}}(\lambda) = \left( \epsilon_g(\lambda) - \epsilon_w(\lambda) \cdot \delta_w \right) \cdot C \cdot L")
        A_actual = modelo_optico.absorbancia(c_sim, lambda_nm)
        st.info(
            rf"A $\lambda = {lambda_nm}\text{{ nm}}$, $L = {L_mm}\text{{ mm}}$ y $C = {c_sim}\text{{ mM}}$, la absorbancia neta es de **{A_actual:.5f} u.a.** "
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
        height=380,
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
        height=380,
        showlegend=True,
        legend=LEYENDA_INFERIOR
    )
    col2.plotly_chart(fig6)
    
    with st.container(border=True):
        st.latex(r"\text{Sensibilidad} = \frac{\partial A}{\partial C} = \left(\epsilon_g(\lambda) - \epsilon_w(\lambda) \cdot \delta_w\right) \cdot L")
        st.info("La sensibilidad aumenta linealmente con el camino óptico L, permitiendo amplificar la señal neta sin afectar la selectividad del desplazamiento del solvente.")

# Tab 4: Inferencia
with tab4:
    st.subheader("Inferencia Clínica")
    A_med = st.number_input("Absorbancia medida (A)", value=-0.05, step=0.001, format="%.5f")
    if st.button("Estimar"):
        c_est = modelo_optico.concentracion_inversa(A_med, lambda_nm)
        st.write(f"Concentración estimada: **{c_est:.4f} mM** — Clasificación: **{modelo_optico.evaluar_clasificacion_fisiologica(c_est)}**")
        
    st.markdown("---")
    st.subheader("Procesamiento por Lotes")
    
    with st.expander("ℹ️ Guía de formato para el archivo CSV y parámetros de análisis", expanded=False):
        st.markdown(rf"""
        **Condiciones de Análisis Activas:**
        * Los datos del archivo se evalúan en tiempo real con la **Longitud de onda ($\lambda = {lambda_nm}\text{{ nm}}$)** y el **Camino óptico ($L = {L_mm}\text{{ mm}}$)** configurados en la barra lateral (*sidebar*).
        
        **Estructura y Unidades Requeridas en el CSV:**
        * **Columna de absorbancia (Obligatoria):** Encabezados válidos: `absorbancia_medida`, `absorbancia_1600nm`, `absorbancia_1650nm`, `absorbancia` o `A`. Valores en **u.a.** (unidades de absorbancia neta).
        * **Columna de referencia (Opcional):** Encabezados válidos: `glucosa_referencia_mM`, `Glucosa_Real_mM`, `glucosa_mM` o `C_real`. Valores en **mM** (milimolar) para el cálculo de error porcentual.
        * **Identificador (Opcional):** `id_muestra` (alfanumérico).

        ⚠️ **Recomendación de Rendimiento:** Se aconseja cargar archivos con un máximo de **1,000 filas** para asegurar una respuesta fluida. Si se sube un lote mayor, el sistema procesará y graficará automáticamente las primeras 1,000 muestras.
        """)

    uploaded = st.file_uploader("Subir archivo CSV de muestras", type=["csv", "txt"])
    
    if uploaded is not None:
        progreso_contenedor = st.container()
        barra_progreso = progreso_contenedor.progress(0)
        estado_texto = progreso_contenedor.empty()
        
        try:
            estado_texto.text("Paso 1/4: Leyendo archivo en memoria...")
            barra_progreso.progress(25)
            
            try:
                uploaded.seek(0)
                df_lote = pd.read_csv(uploaded)
            except Exception:
                uploaded.seek(0)
                df_lote = pd.read_csv(uploaded, sep=None, engine="python")
            
            # Validación y limitación a 1,000 filas para rendimiento óptimo
            total_filas_original = len(df_lote)
            if total_filas_original > 1000:
                df_lote = df_lote.head(1000).copy()
                st.warning(f"El archivo contiene {total_filas_original:,} filas. Para garantizar la fluidez de la interfaz, se procesan y analizan las primeras 1,000 muestras.")

            estado_texto.text("Paso 2/4: Identificando canal óptico de absorbancia...")
            barra_progreso.progress(50)
            time.sleep(0.05)
            
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
                
                estado_texto.text("Paso 4/4: Consolidando resultados y visualizaciones...")
                barra_progreso.progress(100)
                time.sleep(0.05)
                
                barra_progreso.empty()
                estado_texto.success(f"Procesamiento completado: {len(df_lote):,} muestras analizadas bajo λ = {lambda_nm} nm y L = {L_mm} mm.")
                
                # Tabla de resultados
                st.dataframe(df_lote)
                
                # --- GRÁFICOS DE ANÁLISIS DEL LOTE ---
                st.markdown("#### Análisis Estadístico y Fisiológico del Lote")
                col_g1, col_g2 = st.columns(2)
                
                # Gráfico 1: Conteo por Categoría Fisiológica
                conteo_df = df_lote["Clasificación_Metabólica"].value_counts().reset_index()
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
                
                # Gráfico 2: Dispersión de Concentración Estimada con Umbrales
                fig_lote_disp = go.Figure()
                indices_muestras = list(range(1, len(df_lote) + 1))
                
                fig_lote_disp.add_trace(go.Scatter(
                    x=indices_muestras,
                    y=df_lote["Glucosa_Estimada_mM"],
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
                
                # Exportación
                col_btn1, col_btn2 = st.columns(2)
                excel_bytes = generar_excel_multihoja(df_lote, lambda_nm, L_mm)
                
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
                        data=df_lote.to_csv(index=False).encode("utf-8"),
                        file_name="resultados_procesamiento_lote.csv",
                        mime="text/csv"
                    )
            else:
                barra_progreso.empty()
                estado_texto.error("No se detectó una columna de absorbancia válida en el archivo cargado.")
        except Exception as e:
            barra_progreso.empty()
            estado_texto.error(f"Error al procesar el archivo: {e}")