from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATASET_PATH = OUTPUT_DIR / "results_long.parquet"
RACE_EXTRAS_PATH = OUTPUT_DIR / "race_extras.parquet"
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

LOGREG_PARAMS = dict(lr=0.05, n_iters=1500, l2=0.01, class_weight=False)
TREE_PARAMS = dict(max_depth=6, min_samples_split=20, min_samples_leaf=10)
FOREST_PARAMS = dict(
    n_estimators=60,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features="log2",
)

RANDOM_PARAMS = dict()

RANDOM_SEED = 42

TUNE_METRIC = "podium_hit_rate"

LOGREG_GRID = dict(
    lr=[0.05, 0.1, 0.3],
    n_iters=[1500, 3000],
    l2=[0.0, 0.01, 0.1],
    class_weight=[True, False],
)
TREE_GRID = dict(
    max_depth=[4, 6, 8, 12],
    min_samples_split=[5, 10, 20],
    min_samples_leaf=[2, 5, 10],
)
FOREST_GRID = dict(
    n_estimators=[40, 60, 100],
    max_depth=[8, 12, 16],
    min_samples_leaf=[2, 4, 8],
    max_features=["sqrt", "log2"],
)

RANDOM_GRID = dict()

GRIDS = {"logreg": LOGREG_GRID, "tree": TREE_GRID, "forest": FOREST_GRID, "random": RANDOM_GRID}

MODEL_KEYS = ("logreg", "tree", "forest", "random")

EVAL_METRICS = ("accuracy", "podium_hit_rate", "exact_podium_set", "winner_acc")

def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, PREDICTIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
