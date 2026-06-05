import numpy as np

from f1pred.models import DecisionTree, LogisticRegression, RandomForest


def test_logreg_separable():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.2 > 0).astype(int)
    model = LogisticRegression(lr=0.3, n_iters=3000).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95
    p = model.predict_proba(X)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert np.argmax(model.feature_importances_) == 0


def test_decision_tree_learns_and_rule():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 4))
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
    tree = DecisionTree(max_depth=5, min_samples_leaf=5).fit(X, y)
    assert (tree.predict(X) == y).mean() > 0.97
    assert tree.feature_importances_[3] < 0.05


def test_random_forest_beats_or_matches_tree_on_noise():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(700, 6))
    y = ((X[:, 0] + 0.8 * X[:, 1] + rng.normal(scale=0.5, size=700)) > 0).astype(int)
    Xtr, Xte, ytr, yte = X[:500], X[500:], y[:500], y[500:]
    tree = DecisionTree(max_depth=6, min_samples_leaf=5).fit(Xtr, ytr)
    forest = RandomForest(n_estimators=40, max_depth=6, min_samples_leaf=5,
                          oob_score=True, random_state=1).fit(Xtr, ytr)
    acc_tree = (tree.predict(Xte) == yte).mean()
    acc_forest = (forest.predict(Xte) == yte).mean()
    assert acc_forest >= acc_tree - 0.02
    assert forest.oob_score_ is not None and forest.oob_score_ > 0.6


def test_entropy_criterion_pure_split():
    X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]])
    y = np.array([0, 0, 0, 1, 1, 1])
    tree = DecisionTree(max_depth=1, min_samples_split=2, min_samples_leaf=1).fit(X, y)
    assert (tree.predict(X) == y).all()
    assert tree.feature_importances_[0] > 0
