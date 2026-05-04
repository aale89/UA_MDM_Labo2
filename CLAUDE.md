# Guía del proyecto — UA MDM Lab II (Grupo 8)

> **Propósito de este archivo.** Servir de referencia única tanto para colaboradores humanos que se suman al repo como para agentes de IA (Claude Code y similares) que necesiten extraer contexto antes de intervenir. El [`README.md`](README.md) cubre el "qué hace y cómo correrlo"; este documento cubre el "por qué está organizado así, qué decisiones se tomaron y qué reglas mantener".

---

## 1. Resumen ejecutivo

- **Qué es.** Trabajo del **Grupo 8** del Laboratorio II de la **Maestría en Data Mining (Universidad Austral)**, enfocado en el dataset **PetFinder Adoption Prediction** (Kaggle).
- **Objetivo del lab.** Construir un pipeline reproducible de EDA + modelado tabular (LightGBM + Optuna) sobre PetFinder, con un dashboard Streamlit que comunique los hallazgos. La rama deep (ResNet/SHAP/Grad-CAM sobre imágenes) está prevista como extensión opcional.
- **Estado actual.** Layout plano consolidado. Entorno reproducible con `pyproject.toml` + `uv` y Python 3.14. Pre-commit hooks instalables. Notebook EDA funcional importando helpers desde `ani_lab.viz`. Dashboard Streamlit operativo.
- **Origen.** Fork independiente de [`AraneoA/UA_MDM_Labo2`](https://github.com/AraneoA/UA_MDM_Labo2), a su vez basado en [`Argentan/DMA_LAB2`](https://github.com/Argentan/DMA_LAB2) (Rafael Crescenzi y Pablo Albini). El material original del docente se preservó intacto bajo `reference/` y **no se versiona como código activo del Grupo 8**.

---

## 2. Contexto del problema

### 2.1 Dataset

`petfinder-adoption-prediction` (competencia Kaggle). Los datos **no se versionan**: cada colaborador los baja manualmente y los coloca en `input/train/`. El gitignore garantiza que ni los CSVs ni las imágenes ni los `.zip` originales contaminen el repo.

Disposición esperada en disco:

```
input/
└── train/
    ├── train.csv                      # 14993 filas × 24 columnas (ground truth)
    ├── PetFinder-BreedLabels.csv      # 307 × 3 — diccionario de razas
    ├── PetFinder-ColorLabels.csv      # 7 × 2   — diccionario de colores
    └── PetFinder-StateLabels.csv      # 15 × 2  — diccionario de provincias (Malasia)
```

**Campo objetivo (`AdoptionSpeed`)**: variable categórica ordinal con 5 valores (0 = adoptado el mismo día, 4 = sin adopción tras 100+ días).

### 2.2 Decisiones de scope

- **Tabular primero, deep después.** El esfuerzo principal está en `train.csv` (atributos del animal, descripciones, metadatos). Las imágenes, perfiles de sentimiento y rama ResNet quedan como capa opcional habilitable con `pip install -e .[deep]`.
- **Reproducibilidad por sobre todo.** Todo el código asume que se corre **desde la raíz del repo** (rutas relativas tipo `os.path.join("input", "train")`). No reubicar archivos sin actualizar también las rutas.
- **Reuso explícito.** La lógica reutilizable vive en `src/ani_lab/`. El notebook y el dashboard la consumen vía import (`from ani_lab.viz import …`). Evitar redefinir helpers inline.

---

## 3. Estructura del repositorio

```
UA_MDM_Labo2/                     # raíz — siempre se corre desde acá
├── src/ani_lab/                  # paquete importable (instalado en modo -e)
│   ├── __init__.py               # vacío, marca el paquete
│   └── viz.py                    # plot_confusion_matrix, get_artifact_filename
│
├── app/
│   └── EDAdashboard.py           # Streamlit (≈40 KB)
│
├── notebooks/
│   └── 01_eda_tabulares.ipynb    # EDA + baseline tabular (outputs limpios via nbstripout)
│
├── reference/                    # ⚠️ material del docente — NO se edita
│   ├── tutoriales/               # 01..06 (clases dictadas)
│   ├── augment/                  # data augmentation (clase ResNet)
│   ├── ppts/                     # presentaciones de teoría
│   └── material_extra/           # links/docs sueltos
│
├── input/                        # 🚫 gitignored — datos crudos de Kaggle
├── work/                         # 🚫 gitignored — artifacts Optuna, predicciones
├── models/                       # 🚫 gitignored — .joblib entrenados (futuro)
│
├── pyproject.toml                # Python 3.14, deps, ruff
├── .pre-commit-config.yaml       # nbstripout + ruff + hooks básicos
├── .gitignore                    # endurecido — ver §6
├── README.md                     # quickstart de setup y ejecución
├── CLAUDE.md                     # ← este archivo
└── .claude/                      # 🚫 gitignored — PLAN.md y artefactos de agentes
```

### 3.1 Qué archivo es responsable de qué

| Archivo / dir                       | Responsabilidad                                                       |
| ----------------------------------- | --------------------------------------------------------------------- |
| `src/ani_lab/viz.py`                | Funciones de visualización y utilidades Optuna reutilizables          |
| `app/EDAdashboard.py`               | UI Streamlit. Carga CSVs desde `input/train/` (línea 101)             |
| `notebooks/01_eda_tabulares.ipynb`  | EDA exploratorio + experimentación con LightGBM + Optuna              |
| `reference/`                        | Material académico congelado: solo lectura, no se ejecuta en el pipeline |
| `pyproject.toml`                    | Única fuente de verdad de dependencias y configuración de tooling      |
| `.pre-commit-config.yaml`           | Garantiza outputs limpios en notebooks y formato consistente en Python |

---

## 4. Stack técnico

- **Python 3.14** (CPython 3.14.4 verificado). Decisión del equipo; no degradar sin acordar.
- **Gestor de entornos**: [`uv`](https://docs.astral.sh/uv/) (reemplaza venv + pip).
- **Core**: `pandas`, `numpy`, `scipy`, `scikit-learn`.
- **Modelado tabular**: `lightgbm` + `optuna` (HPO) + `joblib` (persistencia).
- **Visualización**: `plotly`, `matplotlib`, `streamlit` (dashboard), `shap` (explicabilidad).
- **Soporte**: `tqdm`, `Pillow`.
- **Extra `[deep]`** (opt-in): `torch`, `torchvision` para la rama ResNet.
- **Extra `[dev]`**: `pre-commit`, `nbstripout`, `ruff`.

El paquete `ani_lab` se instala en modo editable (`pip install -e .`), por lo que cualquier cambio en `src/ani_lab/` se ve inmediatamente desde notebooks y scripts sin reinstalar.

---

## 5. Cómo ponerse a trabajar

> Resumen aquí; instrucciones canónicas en [`README.md`](README.md). En caso de discrepancia, gana el README.

```powershell
# Desde la raíz del repo (Windows PowerShell)
uv venv --python 3.14
uv pip install -e .[dev]               # core + tooling
# uv pip install -e .[dev,deep]        # opcional: suma torch/torchvision

.venv\Scripts\Activate.ps1

# Datos (manualmente desde Kaggle): dejar bajo input/train/
streamlit run app/EDAdashboard.py      # dashboard en :8501
jupyter lab notebooks/01_eda_tabulares.ipynb

# Pre-commit (opcional pero recomendado)
pre-commit install
```

**Verificación rápida del setup**:

- `streamlit run app/EDAdashboard.py` debe levantar sin errores. `GET /_stcore/health` devuelve `ok`.
- En el notebook, `from ani_lab.viz import plot_confusion_matrix` debe resolverse. Si falla, faltó el `pip install -e .`.

---

## 6. Convenciones de versionado y datos

### 6.1 Qué entra al repo y qué no

`.gitignore` está endurecido (ver archivo). Reglas mentales:

| Tipo | Versionado | Dónde vive |
| --- | --- | --- |
| Código (`.py`, `.ipynb` sin outputs) | ✅ | `src/`, `app/`, `notebooks/` |
| Datos crudos (`.csv`, `.json`, `.jpg`) | ❌ | `input/` (local, gitignored) |
| Modelos entrenados (`.joblib`, `.pkl`) | ❌ | `models/`, `work/` (gitignored) |
| Artifacts Optuna | ❌ | `work/optuna_artifacts/` |
| `.zip` (datasets comprimidos) | ❌ | bajar y descomprimir, luego borrar |
| Configs de IDE / agentes | ❌ | `.idea/`, `.vscode/`, `.claude/` |
| Build artifacts (`*.egg-info/`, `build/`, `dist/`, `.eggs/`) | ❌ | autogenerados; ver §6.2 |

**Regla práctica**: si un archivo pesa más de unos pocos cientos de KB y no es código, casi seguro debe estar gitignored. `check-added-large-files` (pre-commit) bloquea archivos > 1 MB como red de seguridad.

### 6.2 Por qué no versionamos `*.egg-info/` (ni `build/`, `dist/`, `.eggs/`)

Después de `uv pip install -e .[dev]` aparece `src/ani_lab.egg-info/` con `PKG-INFO`, `SOURCES.txt`, `requires.txt`, `top_level.txt` y `dependency_links.txt`. Es **metadata autogenerada por setuptools** durante la instalación editable.

No se versiona por tres razones:

1. **Se regenera sola.** Cada `pip install -e .` la reescribe desde cero. La fuente de verdad de la metadata es `pyproject.toml`; commitear el `egg-info` duplica esa información y la duplicación se desactualiza.
2. **Es local a la máquina.** Algunos campos (paths absolutos, versión de setuptools, hashes) varían entre entornos, así que commitearla genera diffs ruidosos y conflictos entre colaboradores.
3. **Convención PyPA.** Todos los gitignores oficiales de Python (GitHub, PyPA samples) bloquean `*.egg-info/`, `build/`, `dist/`, `.eggs/`. Es estándar de la comunidad, no idiosincrasia del proyecto.

El `.gitignore` del repo cubre estos patrones en la sección _Build / packaging artifacts_. Si en algún momento aparece un directorio con esos nombres trackeado en `git status`, es síntoma de que el `.gitignore` se rompió, no de algo que haya que commitear.

### 6.3 Branching

- `main` = línea estable. No se hace push directo a `main` sin PR (convención de equipo; aún no protegida formalmente — pendiente del paso 12 del plan).
- Trabajo nuevo en `feat/...`, `fix/...`, `docs/...`. La rama actual (`feat/consolidacion-estructura`) es donde se está consolidando la reorganización inicial.
- Commits **atómicos** y con mensaje convencional (`feat:`, `fix:`, `chore:`, `docs:`, `build:`). Ver `git log` reciente como referencia de estilo.

### 6.4 Pre-commit (qué hace cada hook)

| Hook | Qué corrige / verifica |
| --- | --- |
| `trailing-whitespace` | Espacios al final de línea |
| `end-of-file-fixer` | Newline final faltante |
| `check-yaml` | Sintaxis de YAML |
| `check-added-large-files` | Bloquea commits con archivos > 1 MB |
| `check-merge-conflict` | Marcadores `<<<<<<<` olvidados |
| `nbstripout` | Limpia outputs de `.ipynb` (excluye `reference/`, que se conserva intacto) |
| `ruff --fix` + `ruff-format` | Lint y formato Python (line-length 100, target py314) |

`reference/` está excluido de `nbstripout` y de `ruff` (`extend-exclude` en `pyproject.toml`) — es archivo histórico.

---

## 7. Reglas de oro para intervenir el repo

Estas reglas existen para no romper supuestos del proyecto. Si algo se siente forzado, preguntar antes de violarlas.

1. **Siempre correr desde la raíz del repo.** Las rutas relativas (`os.path.join("input", "train")`) lo asumen. Mover el cwd rompe carga de datos.
2. **No tocar `reference/`.** Es material del docente, congelado. Si hace falta reusar algo de ahí, copiar a `src/ani_lab/` y adaptar.
3. **No commitear datos.** Ni CSV, ni imágenes, ni `.zip`, ni modelos. El `.gitignore` los bloquea pero conviene revisar `git status` antes de cada commit.
4. **Lógica reutilizable → `src/ani_lab/`.** No copiar funciones entre notebook y dashboard. Si una función aparece en dos lugares, mudala al paquete y refactorizar los dos call sites.
5. **Outputs de notebooks limpios.** `nbstripout` lo hace automáticamente si los hooks están instalados. Notebook commiteado con outputs = revisión rechazada.
6. **No introducir nuevas deps sin pasar por `pyproject.toml`.** Nada de `pip install` ad hoc en el venv. Siempre `uv add` o editar `pyproject.toml` y `uv pip install -e .[dev]`.
7. **Python 3.14 es target fijo.** Cambiarlo afecta wheels disponibles (sobre todo en la rama deep). Coordinar con el equipo antes.
8. **Claude/agentes no commitean por iniciativa.** Solo si el usuario lo pide explícitamente. Lo mismo para `git push`, force-push, branch deletion, etc.

---

## 8. Mapa de componentes principales (referencias rápidas)

- **Carga de datos del dashboard** — [`app/EDAdashboard.py:101`](app/EDAdashboard.py) define `BASE_PATH = os.path.join("input", "train")` y a partir de ahí lee `train.csv` y los tres archivos de labels.
- **Helpers de visualización** — [`src/ani_lab/viz.py:14`](src/ani_lab/viz.py) `plot_confusion_matrix(y_test, y_pred, labels=None, ...)` devuelve un `plotly.graph_objects.Figure` con matriz de confusión normalizada + conteos.
- **Helpers Optuna** — [`src/ani_lab/viz.py:7`](src/ani_lab/viz.py) `get_artifact_filename(study, prefix)` recupera artifact_ids del best_trial filtrando por prefijo.
- **Notebook principal** — `notebooks/01_eda_tabulares.ipynb` (importa `from ani_lab.viz import ...`).

---

## 9. Roadmap pendiente

Pasos heredados de [`.claude/PLAN.md`](.claude/PLAN.md) que aún no se cerraron (el plan no está versionado, vive solo en disco):

- **Limpieza de disco local** (no impacta el repo, solo el filesystem):
  - Borrar `.zip` grandes ya descomprimidos: `input/participant.zip` (~7.96 GB) y el `petfinder-adoption-prediction.zip` interno de `petfinder_deep_COMPARTIR/` (~1.94 GB).
  - Borrar `input/petfinder_deep_COMPARTIR/.venv/` (entorno virtual extraviado).
  - Mover datos vivos del Grupo 8 desde el árbol legacy `Labo2_grupo_8/` al canónico `<root>/input/train/` y `<root>/work/`.
- **Workflow del repo**:
  - Activar protección de rama sobre `main` y configurar Issues / Project board en GitHub.
- **Trabajo de modelado** (lo que viene):
  - Partir el notebook EDA en piezas temáticas más chicas a medida que crezca.
  - Mover funciones de feature engineering / entrenamiento desde el notebook a `src/ani_lab/` (futuras submódulos `features.py`, `models.py`).
  - Agregar `tests/` y `scripts/download_data.py`.
- **Largo plazo**: DVC para versionar datos, MLflow u `optuna-dashboard` para tracking de experimentos.

---

## 10. Glosario rápido

- **AdoptionSpeed**: variable objetivo de PetFinder. Ordinal 0–4 (0 = adopción inmediata, 4 = no adoptado tras 100 días).
- **`ani_lab`**: nombre del paquete Python interno del Grupo 8. Vive en `src/ani_lab/`.
- **`reference/`**: material del docente preservado como archivo histórico. No es parte del pipeline activo.
- **Rama deep**: extensión opcional del trabajo, basada en ResNet50 + SHAP + Grad-CAM sobre las imágenes de las mascotas. Se activa con `[deep]`.
- **`uv`**: gestor de paquetes/entornos de Astral. Reemplaza `python -m venv` + `pip` con un único binario más rápido.

---

## 11. Para agentes de IA — checklist mínimo antes de intervenir

1. Leer este archivo y `README.md`.
2. Revisar `git status` y `git log -10` antes de cualquier acción que modifique archivos versionados.
3. Comprobar la rama actual; si es `main`, pedir confirmación antes de commitear.
4. No tocar `reference/` salvo lectura.
5. Si una intervención requiere agregar dependencias, editar `pyproject.toml` (no instalar ad hoc).
6. Si una intervención toca notebooks, asumir que `nbstripout` correrá en commit — no embeber outputs grandes.
7. Pedir confirmación antes de: `git push`, `git reset --hard`, borrar archivos en `input/`/`work/`, modificar `.gitignore`, modificar `pyproject.toml` de forma que cambie el contrato de instalación.
