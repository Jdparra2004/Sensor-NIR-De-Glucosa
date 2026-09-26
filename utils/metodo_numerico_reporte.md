# Reporte de Auditoría del Método Numérico — Simulador de Biosensor NIR de Glucosa

## 1. Resumen Ejecutivo
Este documento presenta la auditoría técnica y formal de los modelos numéricos implementados en el núcleo del simulador (`core/modelo_optico.py` y `core/modelo_microfluido.py`). El objetivo es documentar rigurosamente los fundamentos físicos, ecuaciones gobernantes, esquemas de discretización, parámetros iniciales y algoritmos utilizados para la simulación espectroscópica infrarroja (NIR) y el transporte microfluídico laminar de glucosa en sudor ecrino.

---

## 2. Modelo Óptico: Ley de Beer-Lambert Modificada y PLS-R
El modelo óptico evalúa la absorción de radiación en el infrarrojo cercano (NIR) dentro de la ventana analítica óptima ($1600\text{--}1700\text{ nm}$), incorporando corrección por exclusión volumétrica de agua (Amerov et al., 2004) y regresión multivariante por mínimos cuadrados parciales (PLS-R).

### 2.1. Ecuación Gobernante
La absorbancia neta $A_{\text{neta}}$ se modela mediante la Ley de Beer-Lambert modificada para disoluciones acuosas ultra-diluidas:
$$A_{\text{neta}}(\lambda, C) = \left[ \varepsilon_{\text{g}}(\lambda) - \varepsilon_{\text{w}}(\lambda) \cdot \delta_{\text{w}} \right] \cdot C \cdot L$$

Donde:
- $\varepsilon_{\text{g}}(\lambda)$: Coeficiente de absortividad molar de la glucosa $[\text{mM}^{-1} \cdot \text{mm}^{-1}]$ en la longitud de onda $\lambda$.
- $\varepsilon_{\text{w}}(\lambda)$: Coeficiente de absortividad del agua $[\text{mm}^{-1}]$ en $\lambda$.
- $\delta_{\text{w}}$: Coeficiente adimensional de desplazamiento volumétrico de agua ($\delta_{\text{w}} \approx 6.15$).
- $C$: Concentración molar de glucosa $[\text{mM}]$ (rango fisiológico analítico: $0.01$ a $1.0\text{ mM}$).
- $L$: Longitud del camino óptico $[\text{mm}]$ (configurable: $0.1$ a $2.0\text{ mm}$).

### 2.2. Discretización y Parámetros Espectrales
- **Ventana espectral:** $1000\text{ nm}$ a $1700\text{ nm}$.
- **Longitud de onda de referencia principal:** $\lambda_{\text{ref}} = 1650\text{ nm}$.
- **Puntos de discretización espectral (Tabla tabulada de absortividades):**
  - $1000\text{ nm}$, $1100\text{ nm}$, $1200\text{ nm}$, $1300\text{ nm}$, $1400\text{ nm}$, $1450\text{ nm}$, $1550\text{ nm}$, $1600\text{ nm}$, $1650\text{ nm}$, $1700\text{ nm}$.
- **Método de interpolación:** Interpolación lineal unidimensional (`np.interp`) para evaluar longitudes de onda continuas entre los nodos tabulados.

### 2.3. Inferencia Inversa y Calibración Empírica
A partir de la absorbancia medida $A$, la concentración estimada $C_{\text{final}}$ se recupera mediante inversión analítica con corrección lineal empírica:
$$C_{\text{teorica}} = \frac{|A|}{|\varepsilon_{\text{net}} \cdot L|}$$
$$C_{\text{final}} = \max\left(0.0, (C_{\text{teorica}} \cdot \alpha) + β\right)$$
Donde $\alpha$ es el factor de escala y $\beta$ es el sesgo instrumental.

### 2.4. Modelo Quimiométrico PLS-R
Para validación con conjuntos de datos experimentales masivos (ej. Dataset NTNU):
- **Algoritmo:** Regresión por Mínimos Cuadrados Parciales (`sklearn.cross_decomposition.PLSRegression`).
- **Filtrado de canales (Masking):** Exclusión automática de bandas con ruido extremo o absorción saturada de agua ($<500\text{ nm}$, cruce de detector $1090\text{--}1110\text{ nm}$, bandas fuertes de agua $1800\text{--}2100\text{ nm}$ y $>2300\text{ nm}$).
- **Optimización de componentes:** Selección automática de variables latentes óptimas (LVs) mediante validación cruzada de 5 particiones ($k=5$ CV) maximizando el coeficiente $R^2$.

---

## 3. Modelo Microfluídico: Hidrodinámica Laminar de Hagen-Poiseuille
El modelo microfluídico simula el transporte pasivo del sudor ecrino en microcanales rectangulares para garantizar un régimen estacionario sin turbulencias.

### 3.1. Ecuaciones Gobernantes
- **Área transversal ($A_c$):** $A_c = w \cdot h$ [m$^2$] (donde $w$ es ancho y $h$ es alto).
- **Perímetro humedecido ($P$):** $P = 2 \cdot (w + h)$ [m].
- **Diámetro hidráulico ($D_h$):** $D_h = \frac{4 A_c}{P}$ [m].
- **Velocidad media ($v$):** $v = \frac{Q}{A_c}$ [m/s] (donde $Q$ es el caudal volumétrico).
- **Número de Reynolds ($Re$):** 
  $$Re = \frac{\rho \cdot v \cdot D_h}{\mu}$$
  Donde $\rho = 1005.0\text{ kg/m}^3$ (densidad del sudor) y $\mu = 1.0 \times 10^{-3}\text{ Pa}\cdot\text{s}$ (viscosidad dinámica).
- **Criterio de flujo laminar estricto:** $Re < 1.0$.
- **Tiempo de residencia ($t_r$):** $t_r = \frac{L_{\text{canal}}}{v} = \frac{V_{\text{canal}}}{Q}$ [s].

### 3.2. Parámetros Iniciales del Microcanal
- **Ancho ($w$):** $50$ a $500\ \mu\text{m}$ (valor por defecto: $200\ \mu\text{m}$).
- **Alto ($h$):** $10$ a $200\ \mu\text{m}$ (valor por defecto: $50\ \mu\text{m}$).
- **Largo ($L_{\text{canal}}$):** $0.5$ a $5.0\text{ mm}$ (valor por defecto: $1.0\text{ mm}$).
- **Caudal ($Q$):** $1.0$ a $10.0\text{ nL/min}$ (valor por defecto: $5.0\text{ nL/min}$).

---

## 4. Conclusión de la Auditoría
Los métodos numéricos implementados son analíticamente deterministas y computacionalmente estables. El cumplimiento estricto de $Re < 1$ valida el régimen laminar laminar en microcanales, mientras que la ley de Beer-Lambert modificada con corrección por desplazamiento de agua y PLS-R proporciona una base sólida para la estimación cuantitativa de glucosa en sudor in silico.
