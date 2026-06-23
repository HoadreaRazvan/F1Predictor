from .base import BaseModel, Standardizer
from .logistic_regression import LogisticRegression
from .decision_tree import DecisionTree
from .random_forest import RandomForest
from .random_predictor import RandomPredictor

__all__ = [
    "BaseModel",
    "Standardizer",
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "RandomPredictor",
    "build_model",
]

def build_model(key: str, **overrides):
    from .. import config

    key = key.lower()
    if key == "logreg":
        params = {**config.LOGREG_PARAMS, **overrides}
        return LogisticRegression(**params)
    if key == "tree":
        params = {**config.TREE_PARAMS, **overrides}
        return DecisionTree(**params)
    if key == "forest":
        params = {**config.FOREST_PARAMS, "random_state": config.RANDOM_SEED, **overrides}
        return RandomForest(**params)
    if key == "random":
        params = {**config.RANDOM_PARAMS, "random_state": config.RANDOM_SEED, **overrides}
        return RandomPredictor(**params)
    raise ValueError(f"Model necunoscut: {key!r}. Folosește unul din {config.MODEL_KEYS}.")
