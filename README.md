
# Simulador NIR-Glucosa — Bioingeniería UPB

**Evaluación paramétrica de un modelo matemático para el monitoreo no invasivo de glucosa en el sudor, mediante simulación numérica computacional basada en detección óptica de infrarrojo cercano (NIR).**

Este software es una herramienta de simulación avanzada que integra la física de la espectroscopia NIR, la dinámica de fluidos en microcanales y un motor de inferencia clínica para el diagnóstico preventivo de diabetes.

---

## 🚀 Características Principales

* **Modelo Óptico de Precisión:** Implementación de la Ley de Beer-Lambert adaptada con el fenómeno de **desplazamiento de agua** (Water Displacement), fundamental para la detección en medios acuosos diluidos.
* **Análisis Microfluídico:** Validación de régimen laminar mediante el cálculo automático de los números de **Reynolds** y **Péclet**, además del tiempo de residencia en la cámara de detección.
* **Dashboard Clínico Avanzado:**
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

## 🛠️ Instalación y Uso

### Requisitos previos

* Python 3.10 o superior.
* Librerías: `numpy`, `pandas`, `matplotlib`.

### Instalación

```bash
pip install -r requirements.txt
```
