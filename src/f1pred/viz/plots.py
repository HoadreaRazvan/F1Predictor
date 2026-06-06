from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_COLORS = {"logreg": "#1f77b4", "tree": "#2ca02c", "forest": "#d62728"}

_METRIC_LABELS = {
    "accuracy": "Acuratețe",
    "podium_hit_rate": "Rată nimerire podium",
    "exact_podium_set": "Podium exact",
    "winner_acc": "Câștigător (Top-1)",
}


def metric_label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, metric)


def plot_metric_per_season(per_season_by_model: dict, metric: str, path: Path, title: str | None = None) -> Path:
    models = list(per_season_by_model.keys())
    seasons = sorted(per_season_by_model[models[0]]["season"].tolist())
    x = np.arange(len(seasons))
    width = 0.8 / max(len(models), 1)
    label = metric_label(metric)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        df = per_season_by_model[m].set_index("season").reindex(seasons)
        ax.bar(x + i * width, df[metric].to_numpy(), width, label=m, color=_COLORS.get(m))
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(seasons)
    ax.set_xlabel("Sezon-test")
    ax.set_ylabel(label)
    ax.set_ylim(0, 1)
    ax.set_title(title or f"{label} pe sezon (test)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_metric_summary(summary, metrics_list, path: Path, title: str | None = None) -> Path:
    models = summary["model"].tolist()
    x = np.arange(len(metrics_list))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        row = summary[summary["model"] == m].iloc[0]
        vals = [float(row[mt]) for mt in metrics_list]
        bars = ax.bar(x + i * width, vals, width, label=m, color=_COLORS.get(m))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([metric_label(mt) for mt in metrics_list])
    ax.set_ylabel("Scor (medie pe sezoane-test)")
    ax.set_ylim(0, 1)
    ax.set_title(title or "Performanța modelelor — medii pe sezoanele de test")
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


_MODEL_ORDER = ("logreg", "tree", "forest")
_REGIM_STYLE = {"cu": "-", "fara": "--"}


def plot_insezon_comparison(df, metric: str, season: int, path: Path) -> Path:
    label = metric_label(metric)
    models = [m for m in _MODEL_ORDER if m in set(df["model"])]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in models:
        for regim, style in _REGIM_STYLE.items():
            sub = df[(df["model"] == m) & (df["regim"] == regim)].sort_values("round")
            if sub.empty:
                continue
            cummean = sub[metric].expanding().mean().to_numpy()
            ax.plot(sub["round"].to_numpy(), cummean, linestyle=style,
                    color=_COLORS.get(m), marker="o", ms=3,
                    label=f"{m} — {'cu' if regim == 'cu' else 'fără'} in-sezon")
    ax.set_xlabel("Runda")
    ax.set_ylabel(f"{label} (medie cumulativă)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{label} — efectul curselor in-sezon ({season})")
    ax.legend(loc="lower right", fontsize=8, ncol=len(models))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
