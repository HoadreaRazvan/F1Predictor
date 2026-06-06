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

EVAL_METRICS = ("accuracy", "podium_hit_rate", "exact_podium_set", "winner_acc")


def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, PREDICTIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
