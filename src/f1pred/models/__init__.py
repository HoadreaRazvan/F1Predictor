from .base import BaseModel, Standardizer
from .logistic_regression import LogisticRegression
from .decision_tree import DecisionTree
from .random_forest import RandomForest

__all__ = [
    "BaseModel",
    "Standardizer",
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
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
    raise ValueError(f"Model necunoscut: {key!r}. Folosește unul din {config.MODEL_KEYS}.")
