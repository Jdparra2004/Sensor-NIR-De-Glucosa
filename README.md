# Simulador NIR-Glucosa — Bioingeniería UPB

**Evaluación paramétrica de un modelo matemático para el monitoreo no invasivo
de glucosa en el sudor, mediante simulación numérica computacional basada en
detección óptica de infrarrojo cercano (NIR)**

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Interfaz gráfica interactiva
python main.py

# Modo consola
python main.py --consola

# Ejecutar y exportar CSV
python main.py --exportar
```

## Estructura del proyecto

```
glucosa_nir/
├── main.py                        # Punto de entrada
├── requirements.txt
├── core/
│   ├── modelo_optico.py           # Ley de Beer-Lambert NIR
│   ├── modelo_microfluido.py      # Transporte laminar conceptual
│   └── simulacion_parametrica.py  # Motor de simulación
├── gui/
│   └── interfaz_grafica.py        # GUI con Tkinter + Matplotlib
├── utils/
│   └── visualizacion.py           # Gráficas de publicación
└── outputs/                       # CSV y figuras exportadas
```

## Fundamento físico

**Modelo óptico:**  `A = ε_g(λ)·C·L - ε_w(λ)·δ_w·C·L`

- `ε_g(λ)`: absorptividad molar de glucosa [mM⁻¹·mm⁻¹]
- `ε_w(λ)`: absorptividad del agua [mm⁻¹]
- `δ_w`: coeficiente de desplazamiento de agua
- `C`: concentración de glucosa en sudor [mM]
- `L`: longitud del camino óptico [mm]

**Rango fisiológico simulado:** 0.01 – 1.0 mM (Gao et al., 2016)

**Ventana espectral principal:** 1600 – 1700 nm (Yang et al., 2025)

## Referencias clave

- Amerov et al. (2004). *Applied Spectroscopy*, 58(10), 1195–1204.
- Heise et al. (2021). *Biosensors*, 11(3).
- Yang et al. (2025). *Advanced Sensor Research*, 4(3).
- Gao et al. (2016). *Nature*, 529, 509–514.
- Squires & Quake (2005). *Reviews of Modern Physics*, 77(3).
