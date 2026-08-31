# Manual Técnico: Biosensor NIR

Este documento proporciona una guía técnica sobre el uso y las capacidades de la interfaz web del **Biosensor NIR**, desarrollada en **Streamlit**.

---

## 📋 Introducción
El Biosensor NIR es una herramienta de simulación numérica avanzada para la evaluación de parámetros en sistemas de detección óptica de glucosa en sudor mediante espectroscopia infrarroja cercana (NIR).

> **⚠️ AVISO LEGAL:** Esta aplicación es **exclusivamente una herramienta de simulación para diseño y exploración**. **NO** es un dispositivo médico y no proporciona diagnósticos clínicos.

---

## 🚀 Puesta en Marcha

1. **Requisitos:** Asegúrese de tener instalado Python 3.10+ y las dependencias listadas en el proyecto.
2. **Instalación:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Ejecución:**
   ```bash
   streamlit run app_streamlit.py
   ```

---

## 🖥️ Interfaz de Usuario (Dashboard)

La aplicación está organizada en cuatro pestañas principales:

### 1. Óptica NIR
Analiza la respuesta óptica basada en la Ley de Beer-Lambert corregida por desplazamiento de agua. Permite ajustar la longitud de onda ($\lambda$), el camino óptico ($L$) y la concentración de glucosa ($C$), visualizando la absorbancia y el espectro resultante.

### 2. Microfluídica
Evalúa la hidrodinámica del canal. Calcula el **Número de Reynolds** (para confirmar flujo laminar) y el **Tiempo de residencia** del fluido.

### 3. Sensibilidad
Estudia la relación entre la geometría del biosensor ($L$) y la sensibilidad local ($dA/dC$). Ayuda a identificar el diseño geométrico óptimo para la detección.

### 4. Inferencia Clínica
Motor de procesamiento para la estimación de concentración de glucosa a partir de valores de absorbancia (puntual o por lotes).

---

## 📊 Procesamiento de Datos por Lotes (CSV/TXT)

Para analizar múltiples muestras:
1. Asegúrese de que su archivo esté en formato `.csv`, `.parquet` o `.txt` (tab-separated).
2. **Estructura Requerida:**
   - **Matriz Espectral:** Columnas con encabezados numéricos representando las longitudes de onda (ej. `400`, `1600`).
   - **Columna de Referencia (Opcional):** Encabezados reconocidos como `Glucose (mM)`, `glucosa_referencia_mM`, etc.
3. Suba el archivo en la pestaña "Inferencia Clínica" y el sistema aplicará automáticamente el modelo de regresión multivariante (**PLS-R**) si detecta una matriz espectral completa, o el modelo univariante si solo detecta columnas de absorbancia.
4. Descargue el reporte final en formato Excel (`.xlsx`) o CSV.

---

## 🛠️ Detalles de Implementación
* **Frontend:** Streamlit, Plotly.
* **Backend:** NumPy, Pandas, Scikit-learn (internamente en los modelos).
* **Formatos de exportación:** Excel multi-hoja con métricas de diseño y distribución metabólica.
