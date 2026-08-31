# Simulador NIR-Glucosa — Bioingeniería 

**Evaluación paramétrica de un modelo matemático para el monitoreo no invasivo de glucosa en el sudor, mediante simulación numérica computacional basada en detección óptica de infrarrojo cercano (NIR).**

Este software es una herramienta de simulación avanzada que integra la física de la espectroscopia NIR, la dinámica de fluidos en microcanales y un motor de inferencia clínica para el diagnóstico preventivo de diabetes.

---

## 🚀 Características Principales

* **Modelo Óptico de Precisión:** Implementación de la Ley de Beer-Lambert adaptada con el fenómeno de **desplazamiento de agua** (Water Displacement), fundamental para la detección en medios acuosos diluidos.
* **Análisis Microfluídico:** Validación de régimen laminar mediante el cálculo automático de los números de **Reynolds** y **Péclet**, además del tiempo de residencia en la cámara de detección.
* **Dashboard Clínico Avanzado (Web):**
  * Interfaz interactiva mediante **Streamlit**.
  * **Inferencia Inversa:** Cálculo de la concentración de glucosa [$mM$] a partir de la absorbancia medida por el sensor.
  * **Semáforo de Riesgo:** Clasificación visual inmediata (Normal, Prediabetes, Hiperglucemia) según rangos fisiológicos de literatura.
  * **Análisis de Incertidumbre:** Modelado de la confianza del diagnóstico basado en el ruido fotométrico del hardware (Distribución Gaussiana).
* **Exportación de Datos:** Generación de reportes técnicos en formatos CSV y JSON con selección de ruta personalizada.
* **QA Integrado:** Suite de pruebas unitarias que garantizan la integridad de los cálculos físicos y lógicos.

---

## 🔬 Fundamento Físico-Matemático

El simulador se basa en la absorción neta del sistema, donde la señal detectada es la diferencia entre la absorción de la glucosa y la reducción de la absorción del agua desplazada:

$$
A(\lambda, C) = \epsilon_{g}(\lambda) \cdot C \cdot L - \epsilon_{w}(\lambda) \cdot \delta_{w} \cdot C \cdot L
$$

* **$C$**: Concentración de glucosa [$mM$].
* **$L$**: Longitud del camino óptico [$mm$].
* **$\delta_{w}$**: Coeficiente de desplazamiento volumétrico ($\approx 6.15$ para glucosa).

---

### Arquitectura del Proyecto

El proyecto está diseñado bajo un paradigma modular para separar la lógica de simulación, las pruebas y la interfaz.

#### Estructura de Directorios

```text
Sensor-NIR-De-Glucosa/
├── core/                # Modelos físicos fundamentales
├── data/                # Datos de entrenamiento y validación
├── outputs/             # Resultados exportados (CSV/JSON)
├── test/                # Suite de validación y pruebas unitarias
├── utils/               # Módulos auxiliares (datos y visualización)
├── app_streamlit.py     # Interfaz web (Streamlit)
├── main.py              # Punto de entrada (CLI)
└── README.md            # Documentación del proyecto
```

#### Flujo de Dependencias
El sistema sigue una dirección de flujo clara para mantener la coherencia y facilitar las pruebas:
`Tests (test/)` → `Motor de Simulación (core/simulacion_parametrica.py)` → `Modelos Nucleares (core/modelo_optico.py, core/modelo_microfluido.py)`

Esta separación asegura que los cambios en los modelos nucleares sean automáticamente validados por las pruebas y reflejados en el motor de simulación.

---


### Requisitos previos

* Python 3.10 o superior.
* Librerías: `numpy`, `pandas`, `matplotlib`, `streamlit`, `plotly`.

### Instalación

```bash
pip install -r requirements.txt
```

### Ejecución

#### Opción 1: Interfaz Web (Recomendado)
```bash
streamlit run app_streamlit.py
```

#### Opción 2: Línea de comandos (Análisis técnico)
```bash
# Ejecutar simulaciones y mostrar resultados
python main.py --consola

# Ejecutar y exportar resultados a CSV
python main.py --exportar
```
