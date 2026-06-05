from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_COLORS = {"logreg": "#1f77b4", "tree": "#2ca02c", "forest": "#d62728"}


def plot_metric_per_season(per_season_by_model: dict, metric: str, path: Path, title: str | None = None) -> Path:
    models = list(per_season_by_model.keys())
    seasons = sorted(per_season_by_model[models[0]]["season"].tolist())
    x = np.arange(len(seasons))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        df = per_season_by_model[m].set_index("season").reindex(seasons)
        ax.bar(x + i * width, df[metric].to_numpy(), width, label=m, color=_COLORS.get(m))
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(seasons)
    ax.set_xlabel("Sezon-test")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} per sezon (validare rolling)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_feature_importance(names, importances, model_key: str, path: Path, top: int = 15) -> Path:
    imp = np.asarray(importances, dtype=float)
    order = np.argsort(imp)[::-1][:top]
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(order))[::-1]
    ax.barh(y, imp[order], color=_COLORS.get(model_key, "#555"))
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("Importanță")
    ax.set_title(f"Importanța feature-urilor — {model_key}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion(cm: dict, model_key: str, path: Path) -> Path:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["non-podium", "podium"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["non-podium", "podium"])
    ax.set_xlabel("Prezis"); ax.set_ylabel("Real")
    ax.set_title(f"Matrice de confuzie — {model_key}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_standings(pred_tbl, act_tbl, season: int, path: Path, top: int = 10) -> Path:
    act = act_tbl.head(top).set_index("driver")["points"]
    pred = pred_tbl.set_index("driver")["pred_points"].reindex(act.index).fillna(0)
    x = np.arange(len(act))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, act.to_numpy(), 0.4, label="Real", color="#444")
    ax.bar(x + 0.2, pred.to_numpy(), 0.4, label="Prezis", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(act.index, rotation=45, ha="right")
    ax.set_ylabel("Puncte")
    ax.set_title(f"Clasament campionat {season}: real vs prezis")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


_METRIC_MARKERS = {
    "accuracy": "o",
    "f1": "s",
    "podium_hit_rate": "^",
    "winner_acc": "D",
    "champ_top3_overlap": "v",
}


def plot_validation_curve(df, param: str, path: Path, model_key: str,
                          metrics_list=None, time_col: str = "fit_seconds") -> Path:
    metrics_list = metrics_list or [m for m in _METRIC_MARKERS if m in df.columns]
    x = np.arange(len(df))
    labels = df["value_str"].tolist() if "value_str" in df.columns else [str(v) for v in df["value"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    for m in metrics_list:
        ax.plot(x, df[m].to_numpy(), marker=_METRIC_MARKERS.get(m, "o"), label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(param)
    ax.set_ylabel("Scor metrică")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    if time_col in df.columns:
        ax2 = ax.twinx()
        ax2.plot(x, df[time_col].to_numpy(), color="#888", linestyle="--", marker="x", label="timp antrenare (s)")
        ax2.set_ylabel("Timp antrenare (s)", color="#888")
        ax2.tick_params(axis="y", labelcolor="#888")

    ax.set_title(f"{model_key}: performanță & timp vs {param}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_train_time(df, path: Path) -> Path:
    models = df["model"].tolist()
    secs = df["fit_seconds"].to_numpy()
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, secs, color=[_COLORS.get(m, "#555") for m in models])
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Timp de antrenare (secunde)")
    ax.set_title("Timp de antrenare per model (set complet de antrenament)")
    ax.grid(axis="y", alpha=0.3)
    for b, s in zip(bars, secs):
        ax.text(b.get_x() + b.get_width() / 2, s, f"{s:.3f}s", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_grid_heatmap(matrix, values_a, values_b, param_a: str, param_b: str,
                      path: Path, model_key: str, metric: str) -> Path:
    mat = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(1.4 * len(values_b) + 2, 1.0 * len(values_a) + 2))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(values_b)))
    ax.set_xticklabels([("all" if v is None else str(v)) for v in values_b])
    ax.set_yticks(np.arange(len(values_a)))
    ax.set_yticklabels([("all" if v is None else str(v)) for v in values_a])
    ax.set_xlabel(param_b)
    ax.set_ylabel(param_a)
    ax.set_title(f"{model_key}: {metric} pe grid {param_a} × {param_b}")
    vmax = np.nanmax(mat) if np.isfinite(mat).any() else 1.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        color="white" if mat[i, j] < vmax * 0.7 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc(curves_by_model: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    for m, (fpr, tpr, auc_val) in curves_by_model.items():
        ax.plot(fpr, tpr, color=_COLORS.get(m, None), label=f"{m} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="aleator (AUC=0.5)")
    ax.set_xlabel("Rată fals pozitive (FPR)")
    ax.set_ylabel("Rată adevărat pozitive (TPR)")
    ax.set_title("Curbe ROC — clasificarea is_podium")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_pr(curves_by_model: dict, path: Path, baseline: float | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    for m, (recall, precision, ap) in curves_by_model.items():
        ax.plot(recall, precision, color=_COLORS.get(m, None), label=f"{m} (AP={ap:.3f})")
    if baseline is not None:
        ax.axhline(baseline, color="gray", linestyle="--", label=f"bază (={baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Curbe Precision-Recall — clasificarea is_podium")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_calibration(curves_by_model: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    for m, (mean_pred, frac_pos) in curves_by_model.items():
        mp = np.asarray(mean_pred, dtype=float)
        fp = np.asarray(frac_pos, dtype=float)
        ok = np.isfinite(mp) & np.isfinite(fp)
        ax.plot(mp[ok], fp[ok], marker="o", color=_COLORS.get(m, None), label=m)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="calibrare perfectă")
    ax.set_xlabel("Probabilitate medie prezisă")
    ax.set_ylabel("Frecvență reală de podium")
    ax.set_title("Curbă de calibrare (reliability diagram)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_learning_curve(df_by_model: dict, path: Path, metric: str = "podium_hit_rate") -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, df in df_by_model.items():
        ax.plot(df["n_seasons"].to_numpy(), df[metric].to_numpy(),
                marker="o", color=_COLORS.get(m, None), label=m)
    ax.set_xlabel("Număr de sezoane de antrenament")
    ax.set_ylabel(metric)
    ax.set_title(f"Learning curve — {metric} vs volum de date")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_convergence(loss_history, path: Path, model_key: str = "logreg") -> Path:
    loss = np.asarray(loss_history, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(len(loss)), loss, color=_COLORS.get(model_key, "#1f77b4"))
    ax.set_xlabel("Iterație")
    ax.set_ylabel("Pierdere (cross-entropy + L2)")
    ax.set_title(f"Convergența antrenării — {model_key}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_oob_vs_estimators(df, path: Path) -> Path:
    n = df["n_estimators"].to_numpy()
    err = df["oob_error"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n, err, marker="o", color=_COLORS.get("forest", "#d62728"))
    ax.set_xlabel("Număr de arbori (n_estimators)")
    ax.set_ylabel("Eroare OOB (1 − scor OOB)")
    ax.set_title("Random Forest: eroarea out-of-bag vs numărul de arbori")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_metric_boxplot(values_by_model: dict, metric: str, path: Path) -> Path:
    models = list(values_by_model.keys())
    data = [np.asarray(values_by_model[m], dtype=float) for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, showmeans=True, patch_artist=True)
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(_COLORS.get(m, "#999"))
        patch.set_alpha(0.55)
    for i, d in enumerate(data, start=1):
        ax.scatter(np.full(len(d), i), d, color="black", alpha=0.6, s=18, zorder=3)
    ax.set_xticks(np.arange(1, len(models) + 1))
    ax.set_xticklabels(models)
    ax.set_ylabel(metric)
    ax.set_title(f"Distribuția {metric} pe sezoane-test (stabilitatea modelelor)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_threshold_sweep(df, model_key: str, path: Path) -> Path:
    x = df["threshold"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ("accuracy", "precision", "recall", "f1"):
        ax.plot(x, df[m].to_numpy(), marker="o", ms=3, label=m)
    ax.axvline(0.5, color="gray", linestyle="--", label="prag implicit 0.5")
    best = df.loc[df["f1"].idxmax()]
    ax.axvline(best["threshold"], color="red", linestyle=":", label=f"F1 max @ {best['threshold']:.2f}")
    ax.set_xlabel("Prag de decizie pe P(podium)")
    ax.set_ylabel("Scor metrică")
    ax.set_ylim(0, 1)
    ax.set_title(f"Sensibilitatea la pragul de decizie — {model_key}")
    ax.legend(loc="lower center", ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
