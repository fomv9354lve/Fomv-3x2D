# FOMV: Field Operator for Measured Viability

**Autor:** Osvaldo Morales  
**Licencia:** AGPL-3.0 (código), CC BY-NC-ND 4.0 (documentación)

Este repositorio contiene el código de simulación del modelo **HARD‑nonlinear** utilizado en el paper *"Spectral Diagnostics of Endogenous Memory Systems"*.

## Descripción

El código implementa una simulación de alta resolución sobre un grid de variables lentas (Backlog `B` y Memory `M`). Para cada punto del grid se generan muestras de las variables rápidas (Effort `E`, Governance `G`, Trust `T`, Coherence `C`) y se ejecutan múltiples trayectorias estocásticas para estimar el **tiempo medio al colapso (MFPT)** y la **probabilidad de recuperación (`Q`)**.

Los resultados se almacenan en un DataFrame de pandas y se visualizan mediante:
- Gráfico de contorno 2D (B vs M) con MFPT.
- Gráfico 3D interactivo con **Plotly** (cubo de datos: selecciona cualquier tripleta de variables).
- Widgets deslizantes para explorar cortes 2D.

## Requisitos

- Python 3.7+
- Las dependencias se listan en `requirements.txt`.

## Uso

### En local

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/FOMV.git
   cd FOMV-3x2D
