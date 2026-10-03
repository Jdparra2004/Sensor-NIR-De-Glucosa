# Auditoría y Documentación del Método Numérico - Biosensor NIR de Glucosa

## 1. Introducción y Alcance
Este documento presenta la auditoría formal de los métodos numéricos y matemáticos implementados en el núcleo del sistema (`core/`), abarcando:
- `modelo_optico.py` (Ley de Beer-Lambert modificada con corrección por desplazamiento volumétrico de agua y Regresión PLS-R).
- `modelo_microfluido.py` (Ecuaciones de Hagen-Poiseuille y régimen hidrodinámico laminar).
- `simulacion_parametrica.py` (Barridos paramétricos y evaluación de sensibilidad).

---

## 2. Verificación de Ecuaciones Diferenciales e Integración Temporal
Tras la inspección exhaustiva del código fuente en `core/`:
- **Ausencia de métodos de integración temporal (Euler, Runge-Kutta, etc.):** Se confirma categóricamente que **no** existen sistemas de ecuaciones diferenciales ordinarias (EDO) ni métodos numéricos de integración en el tiempo (como el método de Euler).
- **Naturaleza del modelo:** El sistema opera mediante **evaluación analítica directa** de ecuaciones físicas estacionarias (estado estacionario), complementada con interpolación lineal unidimensional (`np.interp`) para espectros de absortividad molar y regresión multivariada por mínimos cuadrados parciales (PLS-R) a través de `scikit-learn`.

---

## 3. Fundamentos Matemáticos y Numéricos Implementados

### A. Modelo Óptico (`modelo_optico.py`)
1. **Ley de Beer-Lambert Modificada (Amerov et al., 2004):**
   $$A_{\text{neta}}(\lambda, C) = \left( \varepsilon_{\text{glucosa}}(\lambda) - \varepsilon_{\text{agua}}(\lambda) \cdot \delta_w \right) \cdot C \cdot L$$
   Donde $\delta_w = 6.15$ es el coeficiente de desplazamiento volumétrico de agua, $C$ es la concentración molar (mM), y $L$ es la longitud óptica (mm).
2. **Interpolación Espectral:**
   Los coeficientes de absortividad tabulares discretos se evalúan mediante interpolación lineal unidimensional:
   $$\varepsilon(\lambda) = \text{np.interp}(\lambda, \Lambda_{\text{tabla}}, \varepsilon_{\text{tabla}})$$
3. **Regresión Quimiométrica Multivariante (PLS-R):**
   Modelado predictivo basado en reducción de dimensionalidad por variables latentes (LVs), utilizando $N_{\text{comp}} = 6$ componentes principales óptimos sobre espectros normalizados (SNV).

### B. Modelo Microfluídico (`modelo_microfluido.py`)
1. **Hidrodinámica Laminar (Hagen-Poiseuille / Navier-Stokes simplificado):**
   - Velocidad media: $v = \frac{Q}{A_c}$ donde $A_c = w \cdot h$.
   - Diámetro hidráulico: $D_h = \frac{4 A_c}{P}$.
   - Número de Reynolds: $\text{Re} = \frac{\rho \cdot v \cdot D_h}{\mu} < 1.0$ (garantizando flujo laminar estricto).
   - Tiempo de residencia: $t_r = \frac{L}{v}$.

---

## 4. Conclusión
El núcleo de simulación y procesamiento del biosensor NIR de glucosa opera de manera determinista y analítica en régimen estacionario, sin requerir esquemas numéricos iterativos ni métodos de discretización temporal (Euler).
