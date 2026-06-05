from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from . import config
from .evaluation import metrics
from .evaluation.rolling import rolling_cross_year
from .features.engineering import FEATURE_COLUMNS, feature_matrix, get_feature_dataset
from .models import build_model

MODEL_KEYS = config.MODEL_KEYS
from .predict.race_podium import predict_proba_for_races, rank_drivers
from .predict.season_standings import actual_standings, predicted_standings

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

MODEL_NAMES = {
    "logreg": "Regresie Logistică",
    "tree": "Arbore de Decizie",
    "forest": "Random Forest",
}


def _resolve_models(model_arg: str) -> list[str]:
    if model_arg == "all":
        return list(MODEL_KEYS)
    if model_arg not in MODEL_KEYS:
        raise SystemExit(f"Model necunoscut: {model_arg}. Alege din {MODEL_KEYS} sau 'all'.")
    return [model_arg]


def _fit_before(feats: pd.DataFrame, key: str, season: int, upto_round: int | None = None):
    mask = feats["season"] < season
    if upto_round is not None:
        mask = mask | ((feats["season"] == season) & (feats["round"] < upto_round))
    train = feats[mask]
    if train.empty:
        raise SystemExit(f"Nu există date de antrenament înainte de sezonul {season}.")
    X, y, _ = feature_matrix(train)
    model = build_model(key)
    model.fit(X, y)
    return model, len(train)


def cmd_build_data(args) -> None:
    config.ensure_dirs()
    feats = get_feature_dataset(force=True)
    print(f"\nDataset gata: {len(feats)} rânduri, {len(FEATURE_COLUMNS)} feature-uri")
    print(feats.groupby("season")["round"].nunique().to_string())
    print(f"Rată podium: {feats['is_podium'].mean():.3f}")


def cmd_train(args) -> None:
    config.ensure_dirs()
    feats = get_feature_dataset()
    train = feats[feats["season"] <= args.train_until]
    if train.empty:
        raise SystemExit(f"Nu există date până în {args.train_until}.")
    X, y, names = feature_matrix(train)

    for key in _resolve_models(args.model):
        print(f"Antrenez {MODEL_NAMES[key]} pe sezoanele <= {args.train_until} "
              f"({len(train)} rânduri)...")
        model = build_model(key)
        model.fit(X, y)
        path = config.MODELS_DIR / f"{key}.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {"model": model, "feature_names": names, "train_until": args.train_until, "key": key},
                f,
            )
        acc = metrics.accuracy(y, model.predict(X))
        print(f"  acc in-sample={acc:.3f}  -> salvat {path}")
        fi = model.feature_importances_
        if fi is not None:
            top = sorted(zip(names, fi), key=lambda t: t[1], reverse=True)[:8]
            print("  top feature-uri:", ", ".join(f"{n}={v:.3f}" for n, v in top))


def cmd_evaluate(args) -> None:
    config.ensure_dirs()
    feats = get_feature_dataset()
    keys = _resolve_models(args.model)

    results = {}
    for key in keys:
        print(f"\n=== Validare rolling: {MODEL_NAMES[key]} ===")
        results[key] = rolling_cross_year(feats, key, verbose=True)

    print("\n================ SUMAR (medii pe sezoane-test) ================")
    summary_rows = []
    for key, res in results.items():
        ov = res.overall()
        summary_rows.append(
            {
                "model": key,
                "accuracy": ov.get("overall_accuracy", float("nan")),
                "f1": ov.get("overall_f1", float("nan")),
                "podium_hit": ov["mean_podium_hit_rate"],
                "exact_podium": ov["mean_exact_podium_set"],
                "winner_acc": ov["mean_winner_acc"],
                "champ_top3": ov["mean_champ_top3_overlap"],
            }
        )
    summary = pd.DataFrame(summary_rows).round(3)
    print(summary.to_string(index=False))

    summary.to_csv(config.OUTPUT_DIR / "evaluation_summary.csv", index=False)
    for key, res in results.items():
        res.per_season.to_csv(config.OUTPUT_DIR / f"eval_{key}_per_season.csv", index=False)
    print(f"\nCSV-uri salvate în {config.OUTPUT_DIR}")

    if not args.no_plots:
        from .viz import plots

        per_season_by_model = {k: r.per_season for k, r in results.items()}
        plots.plot_metric_per_season(
            per_season_by_model, "podium_hit_rate", config.PLOTS_DIR / "podium_hit_rate.png"
        )
        plots.plot_metric_per_season(
            per_season_by_model, "accuracy", config.PLOTS_DIR / "accuracy.png"
        )
        for key, res in results.items():
            if res.feature_importances is not None:
                plots.plot_feature_importance(
                    res.feature_names, res.feature_importances, key,
                    config.PLOTS_DIR / f"feature_importance_{key}.png",
                )
            cm = metrics.confusion_matrix(res.y_true_all, res.y_pred_all)
            plots.plot_confusion(cm, key, config.PLOTS_DIR / f"confusion_{key}.png")
        print(f"Grafice salvate în {config.PLOTS_DIR}")


def cmd_experiments(args) -> None:
    config.ensure_dirs()
    from .evaluation import experiments as exp

    feats = get_feature_dataset()
    keys = _resolve_models(args.model)
    make_plots = not args.no_plots
    if make_plots:
        from .viz import plots

    sweep_grid = config.SWEEP_GRID_QUICK if args.quick else config.SWEEP_GRID
    grid_2d_cfg = config.GRID_2D_QUICK if args.quick else config.GRID_2D
    metric = config.PRIMARY_METRIC
    EXP, PLT = config.EXPERIMENTS_DIR, config.PLOTS_DIR

    print("\n=== Sweep 1D de hiperparametri ===")
    for key in keys:
        for param, values in sweep_grid.get(key, {}).items():
            print(f"  [{MODEL_NAMES[key]}] {param} = {[exp.fmt_value(v) for v in values]}")
            df = exp.sweep_1d(feats, key, param, values)
            df.to_csv(EXP / f"sweep_{key}_{param}.csv", index=False)
            if make_plots:
                plots.plot_validation_curve(df, param, PLT / f"valcurve_{key}_{param}.png", key)

    print(f"\n=== Grid search 2D (heatmap, metrică = {metric}) ===")
    for key in keys:
        if key not in grid_2d_cfg:
            continue
        pa, va, pb, vb = grid_2d_cfg[key]
        print(f"  [{MODEL_NAMES[key]}] {pa} × {pb}")
        mat = exp.grid_2d(feats, key, pa, va, pb, vb, metric=metric)
        pd.DataFrame(mat, index=[exp.fmt_value(v) for v in va],
                     columns=[exp.fmt_value(v) for v in vb]).to_csv(EXP / f"grid_{key}_{pa}_x_{pb}.csv")
        if make_plots:
            plots.plot_grid_heatmap(mat, va, vb, pa, pb, PLT / f"heatmap_{key}_{pa}_x_{pb}.png", key, metric)

    print("\n=== Timp de antrenare per model ===")
    tt = exp.train_time_per_model(feats, model_keys=keys)
    tt.to_csv(EXP / "train_time.csv", index=False)
    print(tt.to_string(index=False))
    if make_plots:
        plots.plot_train_time(tt, PLT / "train_time_models.png")

    print("\n=== ROC / Precision-Recall / Calibrare / Stabilitate / Prag ===")
    roc_curves, pr_curves, cal_curves, per_season_by_model = {}, {}, {}, {}
    baseline = None
    for key in keys:
        res = rolling_cross_year(feats, key, verbose=False)
        y_true, y_proba = res.y_true_all, res.y_proba_all
        per_season_by_model[key] = res.per_season
        if not len(y_true):
            continue
        if baseline is None:
            baseline = float(np.mean(y_true))
        fpr, tpr, _ = metrics.roc_curve(y_true, y_proba)
        roc_curves[key] = (fpr, tpr, metrics.auc(fpr, tpr))
        rec, prec, _ = metrics.pr_curve(y_true, y_proba)
        pr_curves[key] = (rec, prec, metrics.average_precision(y_true, y_proba))
        cal_curves[key] = metrics.calibration_curve(y_true, y_proba, n_bins=10)[:2]

        thr = exp.threshold_sweep(y_true, y_proba)
        thr.to_csv(EXP / f"threshold_{key}.csv", index=False)
        if make_plots:
            plots.plot_threshold_sweep(thr, key, PLT / f"threshold_{key}.png")
        f1_best = thr.loc[thr["f1"].idxmax(), "threshold"]
        print(f"  [{MODEL_NAMES[key]}] AUC={roc_curves[key][2]:.3f}  AP={pr_curves[key][2]:.3f}  "
              f"F1 max @ prag={f1_best:.2f}")

    if make_plots and roc_curves:
        plots.plot_roc(roc_curves, PLT / "roc_all.png")
        plots.plot_pr(pr_curves, PLT / "pr_all.png", baseline=baseline)
        plots.plot_calibration(cal_curves, PLT / "calibration_all.png")
        for metric_name in ("podium_hit_rate", "accuracy"):
            vals = {k: df[metric_name].to_numpy() for k, df in per_season_by_model.items()
                    if metric_name in df.columns and len(df)}
            if vals:
                plots.plot_metric_boxplot(vals, metric_name, PLT / f"stability_{metric_name}.png")

    print("\n=== Learning curve (volum de date) ===")
    lc = {}
    for key in keys:
        lc[key] = exp.learning_curve(feats, key)
        lc[key].to_csv(EXP / f"learning_curve_{key}.csv", index=False)
    if make_plots and lc:
        plots.plot_learning_curve(lc, PLT / f"learning_curve_{metric}.png", metric=metric)

    if "logreg" in keys:
        loss = exp.convergence_logreg(feats)
        pd.DataFrame({"iteration": np.arange(len(loss)), "loss": loss}).to_csv(
            EXP / "convergence_logreg.csv", index=False)
        if make_plots and len(loss):
            plots.plot_convergence(loss, PLT / "convergence_logreg.png")
        if len(loss):
            print(f"\nConvergență logreg: {len(loss)} iterații, loss final = {loss[-1]:.4f}")

    if "forest" in keys:
        oob = exp.oob_vs_estimators(feats)
        oob.to_csv(EXP / "oob_forest.csv", index=False)
        if make_plots:
            plots.plot_oob_vs_estimators(oob, PLT / "oob_forest.png")
        best = oob.loc[oob["oob_error"].idxmin()]
        print(f"OOB forest: eroare minimă = {best['oob_error']:.3f} la n_estimators = {int(best['n_estimators'])}")

    print(f"\nCSV-uri salvate în {EXP}")
    if make_plots:
        print(f"Grafice salvate în {PLT}")


def cmd_predict_race(args) -> None:
    config.ensure_dirs()
    feats = get_feature_dataset()
    race = feats[(feats["season"] == args.season) & (feats["round"] == args.round)]
    if race.empty:
        raise SystemExit(f"Nu găsesc cursa: sezon {args.season}, runda {args.round}.")

    key = _resolve_models(args.model)[0]
    model, n_train = _fit_before(feats, key, args.season, upto_round=args.round)
    proba = predict_proba_for_races(model, race)
    ranked = rank_drivers(race, proba)

    event = race["event_name"].iloc[0]
    print(f"\n{event} {args.season} (runda {args.round}) — {MODEL_NAMES[key]} "
          f"(antrenat pe {n_train} rânduri)")
    podium = ranked.head(3)["driver"].tolist()
    print(f"\nPODIUM PREZIS:  P1={podium[0]}  P2={podium[1]}  P3={podium[2]}")

    show = ranked.head(10).copy()
    show["proba"] = show["proba"].round(3)
    show["podium_real"] = (show["position"] <= config.PODIUM_CUTOFF).map({True: "✓", False: ""})
    print("\nClasament prezis (top 10):")
    print(show[["pred_position", "driver", "team", "grid", "proba", "position", "podium_real"]]
          .to_string(index=False))

    if race["position"].notna().any():
        actual = race.sort_values("position").head(3)["driver"].tolist()
        hit = len(set(podium) & set(actual))
        print(f"\nPodium real:    P1={actual[0]}  P2={actual[1]}  P3={actual[2]}  "
              f"(nimerite: {hit}/3)")

    out = config.PREDICTIONS_DIR / f"race_{args.season}_r{args.round}_{key}.csv"
    ranked.to_csv(out, index=False)
    print(f"\nSalvat: {out}")


def cmd_predict_season(args) -> None:
    config.ensure_dirs()
    feats = get_feature_dataset()
    season_df = feats[feats["season"] == args.season]
    if season_df.empty:
        raise SystemExit(f"Nu există date pentru sezonul {args.season}.")

    key = _resolve_models(args.model)[0]
    model, n_train = _fit_before(feats, key, args.season)
    proba = predict_proba_for_races(model, season_df)

    pred_tbl = predicted_standings(season_df, proba)
    act_tbl = actual_standings(season_df)

    print(f"\n=== Sezonul {args.season} — {MODEL_NAMES[key]} (antrenat pe {n_train} rânduri) ===")

    print("\nPodium prezis per cursă:")
    rows = []
    df = season_df.copy(); df["_p"] = list(proba)
    for rnd, race in df.groupby("round", sort=True):
        ranked = rank_drivers(race, race["_p"].to_numpy())
        pred = ranked.head(3)["driver"].tolist()
        actual = race.sort_values("position").head(3)["driver"].tolist()
        hit = len(set(pred) & set(actual)) if len(actual) == 3 else 0
        rows.append({"round": int(rnd), "event": race["event_name"].iloc[0][:24],
                     "P1": pred[0], "P2": pred[1], "P3": pred[2], "hit/3": hit})
    race_tbl = pd.DataFrame(rows)
    print(race_tbl.to_string(index=False))
    print(f"\nMedie nimeriri podium: {race_tbl['hit/3'].mean():.2f}/3")

    print("\nCLASAMENT CAMPIONAT — TOP 3:")
    pred3 = pred_tbl.head(3)["driver"].tolist()
    act3 = act_tbl.head(3)["driver"].tolist()
    print(f"  Prezis: P1={pred3[0]}  P2={pred3[1]}  P3={pred3[2]}")
    print(f"  Real:   P1={act3[0]}  P2={act3[1]}  P3={act3[2]}")
    overlap = len(set(pred3) & set(act3))
    print(f"  Suprapunere top 3: {overlap}/3 | ordine exactă: {'DA' if pred3 == act3 else 'nu'}")

    print("\nClasament prezis (top 10):")
    print(pred_tbl.head(10)[["pred_rank", "driver", "team", "pred_points", "pred_wins", "pred_podiums"]]
          .to_string(index=False))

    pred_tbl.to_csv(config.PREDICTIONS_DIR / f"standings_{args.season}_{key}_pred.csv", index=False)
    act_tbl.to_csv(config.PREDICTIONS_DIR / f"standings_{args.season}_actual.csv", index=False)
    race_tbl.to_csv(config.PREDICTIONS_DIR / f"podiums_{args.season}_{key}.csv", index=False)
    print(f"\nCSV-uri salvate în {config.PREDICTIONS_DIR}")

    if not args.no_plots:
        from .viz import plots

        plots.plot_standings(pred_tbl, act_tbl, args.season,
                             config.PLOTS_DIR / f"standings_{args.season}_{key}.png")
        print(f"Grafic salvat în {config.PLOTS_DIR}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="f1pred", description="Predicție podium F1 cu 3 modele ML de la zero.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("build-data", help="Construiește datasetul din cache-ul FastF1").set_defaults(func=cmd_build_data)

    pt = sub.add_parser("train", help="Antrenează și salvează un model")
    pt.add_argument("--model", default="all", help="logreg | tree | forest | all")
    pt.add_argument("--train-until", type=int, default=config.LAST_SEASON - 1, help="ultimul sezon de antrenament")
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("evaluate", help="Validare rolling cross-an + grafice")
    pe.add_argument("--model", default="all", help="logreg | tree | forest | all")
    pe.add_argument("--no-plots", action="store_true", help="nu genera grafice PNG")
    pe.set_defaults(func=cmd_evaluate)

    px = sub.add_parser("experiments", help="Sweep hiperparametri, timpi de antrenare și grafice de diagnostic")
    px.add_argument("--model", default="all", help="logreg | tree | forest | all")
    px.add_argument("--quick", action="store_true", help="grile reduse de hiperparametri (rulare rapidă)")
    px.add_argument("--no-plots", action="store_true", help="nu genera grafice PNG")
    px.set_defaults(func=cmd_experiments)

    pr = sub.add_parser("predict-race", help="Podiumul prezis al unei curse")
    pr.add_argument("--season", type=int, required=True)
    pr.add_argument("--round", type=int, required=True)
    pr.add_argument("--model", default="forest", help="logreg | tree | forest")
    pr.set_defaults(func=cmd_predict_race)

    ps = sub.add_parser("predict-season", help="Podium per cursă + clasament campionat prezis")
    ps.add_argument("--season", type=int, required=True)
    ps.add_argument("--model", default="forest", help="logreg | tree | forest")
    ps.add_argument("--no-plots", action="store_true", help="nu genera grafice PNG")
    ps.set_defaults(func=cmd_predict_season)
    return p


def main(argv=None) -> None:
    for stream in ("stdout", "stderr"):
        try:
            getattr(__import__("sys"), stream).reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
