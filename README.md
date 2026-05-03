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

## Setup

Requiere **Python 3.14**.

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -e .[dev]
```

Si usás `uv`:

```bash
uv venv --python 3.14
uv pip install -e .[dev]
```

Para sumar la rama deep (PyTorch, ResNet, etc.):

```bash
pip install -e .[dev,deep]
```

## Datos

Los datos no se versionan. Bajarlos manualmente desde Kaggle (`petfinder-adoption-prediction`) y dejarlos en la siguiente disposición:

```
input/
└── train/
    ├── train.csv
    ├── PetFinder-BreedLabels.csv
    ├── PetFinder-ColorLabels.csv
    └── PetFinder-StateLabels.csv
```

El dashboard y el notebook resuelven rutas relativas a la raíz del repo (`os.path.join("input", "train")`).

## Correr el dashboard

```bash
streamlit run app/EDAdashboard.py
```

## Notebook EDA

```bash
jupyter lab notebooks/01_eda_tabulares.ipynb
```

El notebook importa helpers vía `from ani_lab.viz import plot_confusion_matrix`. Para que funcione, hay que tener el paquete instalado en modo editable (`pip install -e .`).

## Agradecimientos

Este repositorio nace como fork de [`AraneoA/UA_MDM_Labo2`](https://github.com/AraneoA/UA_MDM_Labo2), basado a su vez en [`Argentan/DMA_LAB2`](https://github.com/Argentan/DMA_LAB2) de Rafael Crescenzi y Pablo Albini. El material de cursada se conserva intacto bajo `reference/`.
