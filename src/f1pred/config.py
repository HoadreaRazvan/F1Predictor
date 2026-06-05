from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATASET_PATH = OUTPUT_DIR / "results_long.parquet"
FEATURES_PATH = OUTPUT_DIR / "features.parquet"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"

SEASONS = list(range(2018, 2026))
FIRST_SEASON = SEASONS[0]
LAST_SEASON = SEASONS[-1]

PODIUM_CUTOFF = 3

F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

FORM_WINDOW = 5

LOGREG_PARAMS = dict(lr=0.1, n_iters=3000, l2=0.01, class_weight=True)
TREE_PARAMS = dict(max_depth=8, min_samples_split=10, min_samples_leaf=5)
FOREST_PARAMS = dict(
    n_estimators=60,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
)

RANDOM_SEED = 42

MODEL_KEYS = ("logreg", "tree", "forest")

SWEEP_GRID = {
    "logreg": {
        "lr": [0.01, 0.03, 0.1, 0.3, 1.0],
        "n_iters": [500, 1000, 2000, 3000, 5000],
        "l2": [0.0, 0.001, 0.01, 0.1, 1.0],
    },
    "tree": {
        "max_depth": [2, 4, 6, 8, 10, 12, 16],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "min_samples_split": [2, 5, 10, 20, 50],
    },
    "forest": {
        "n_estimators": [10, 20, 40, 60, 100, 150],
        "max_depth": [4, 6, 8, 12, 16],
        "max_features": ["sqrt", "log2", None],
    },
}

SWEEP_GRID_QUICK = {
    "logreg": {"lr": [0.03, 0.1, 0.3], "l2": [0.0, 0.01, 0.1]},
    "tree": {"max_depth": [2, 6, 10, 16], "min_samples_leaf": [1, 5, 20]},
    "forest": {"n_estimators": [10, 40, 80], "max_depth": [4, 8, 16]},
}

GRID_2D = {
    "logreg": ("lr", [0.01, 0.03, 0.1, 0.3, 1.0], "l2", [0.0, 0.001, 0.01, 0.1, 1.0]),
    "tree": ("max_depth", [2, 4, 6, 8, 12, 16], "min_samples_leaf", [1, 2, 5, 10, 20]),
    "forest": ("max_depth", [4, 6, 8, 12, 16], "n_estimators", [10, 20, 40, 60, 100]),
}

GRID_2D_QUICK = {
    "logreg": ("lr", [0.03, 0.1, 0.3], "l2", [0.0, 0.01, 0.1]),
    "tree": ("max_depth", [2, 6, 10, 16], "min_samples_leaf", [1, 5, 20]),
    "forest": ("max_depth", [4, 8, 16], "n_estimators", [10, 40, 80]),
}

OOB_N_ESTIMATORS = [5, 10, 20, 40, 60, 100, 150]

PRIMARY_METRIC = "podium_hit_rate"


def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, PREDICTIONS_DIR, EXPERIMENTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
