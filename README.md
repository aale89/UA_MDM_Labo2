# Lab II — Grupo 8 (UA MDM)

Trabajo del Grupo 8 para el Laboratorio II de la Maestría en Data Mining (Universidad Austral) sobre el dataset **PetFinder Adoption Prediction** (Kaggle).

## Estructura del repo

```
.
├── src/ani_lab/         # Código importable (helpers, viz, futuras features/models)
├── app/                 # Dashboard Streamlit
│   └── EDAdashboard.py
├── notebooks/           # Notebooks numerados, outputs limpios
│   └── 01_eda_tabulares.ipynb
├── reference/           # Material de clase del docente (referencia, no se edita)
│   ├── tutoriales/      # Notebooks 01..06 de la cursada
│   ├── augment/         # Helpers de data augmentation (clase ResNet)
│   ├── ppts/            # Presentaciones de teoría
│   └── material_extra/
├── input/               # ⚠️ gitignored — datos locales (Kaggle)
├── work/                # ⚠️ gitignored — artifacts de Optuna, modelos entrenados
├── pyproject.toml
└── README.md
```

## Datos

Los datos no se versionan. Bajarlos manualmente desde Kaggle (`petfinder-adoption-prediction`) y dejarlos en la siguiente disposición:

```
input/
└── train/
    ├── train.csv                      # 14993 filas × 24 columnas
    ├── PetFinder-BreedLabels.csv      # 307 × 3
    ├── PetFinder-ColorLabels.csv      # 7 × 2
    └── PetFinder-StateLabels.csv      # 15 × 2
```

El dashboard y el notebook resuelven rutas relativas a la raíz del repo (`os.path.join("input", "train")`), por lo que **siempre se corren desde la raíz**.

## Cómo correr el proyecto

Requiere **Python 3.14** (verificado con CPython 3.14.4) y [`uv`](https://docs.astral.sh/uv/).

### 1. Crear el entorno e instalar el paquete

```powershell
# desde la raíz del repo
uv venv --python 3.14
uv pip install -e .[dev]
```

Esto crea `.venv/` con Python 3.14, instala las dependencias core (`pandas`, `numpy`, `scikit-learn`, `lightgbm`, `optuna`, `streamlit`, `plotly`, `matplotlib`, `shap`, `joblib`, `tqdm`, `Pillow`) y las herramientas de desarrollo (`pre-commit`, `nbstripout`, `ruff`). El paquete `ani_lab` queda en modo editable, así que `from ani_lab.viz import …` funciona desde notebooks y scripts.

Para sumar la rama deep (PyTorch / ResNet / explicabilidad de imágenes):

```powershell
uv pip install -e .[dev,deep]
```

### 2. Activar el entorno

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

### 3. Arrancar el dashboard Streamlit

```powershell
streamlit run app/EDAdashboard.py
```

Por defecto queda en `http://localhost:8501`. El endpoint `/_stcore/health` devuelve `ok` cuando está listo.

### 4. Abrir el notebook de EDA

```powershell
jupyter lab notebooks/01_eda_tabulares.ipynb
```

El notebook importa helpers vía `from ani_lab.viz import plot_confusion_matrix` — requiere haber hecho el paso 1.

### 5. (Opcional) Activar los pre-commit hooks

```powershell
pre-commit install
```

A partir de ahí, cada commit corre `nbstripout` (limpia outputs de notebooks), `ruff --fix` + `ruff-format` (lint y formato Python) y verificadores básicos (whitespace, EOF, YAML, archivos > 1 MB, conflict markers).

## Artefactos de build (no versionados)

Después del `uv pip install -e .[dev]` van a aparecer en disco directorios autogenerados por setuptools, en particular `src/ani_lab.egg-info/` (metadata del paquete: `PKG-INFO`, `SOURCES.txt`, `requires.txt`, `top_level.txt`, `dependency_links.txt`). Si se hiciera un build no editable, también aparecerían `build/` y `dist/`.

Estos directorios **no se versionan** porque:

- Se regeneran solos en cada instalación; lo "fuente de verdad" vive en `pyproject.toml`.
- Su contenido depende del entorno local (paths absolutos, versiones de setuptools), así que commitearlos genera diffs ruidosos y conflicts entre máquinas.
- Convención estándar de la comunidad Python (PyPA): `*.egg-info/`, `build/`, `dist/`, `.eggs/` siempre van al `.gitignore`.

El `.gitignore` del repo ya bloquea estos patrones en la sección _Build / packaging artifacts_, así que no hace falta intervenir manualmente.

## Agradecimientos

Este repositorio nace como fork de [`AraneoA/UA_MDM_Labo2`](https://github.com/AraneoA/UA_MDM_Labo2), basado a su vez en [`Argentan/DMA_LAB2`](https://github.com/Argentan/DMA_LAB2) de Rafael Crescenzi y Pablo Albini. El material de cursada se conserva intacto bajo `reference/`.
