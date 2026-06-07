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


def cmd_fetch_extras(args) -> None:
    from .data import extras

    df = extras.build_race_extras(throttle=args.throttle, force=args.force)
    print(f"\nÎmbogățiri gata: {len(df)} rânduri -> {config.RACE_EXTRAS_PATH}")
    print("Rulează apoi: python main.py build-data")


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


_METRIC_PLOT_FILE = {
    "accuracy": "accuracy.png",
    "podium_hit_rate": "podium_hit_rate.png",
    "exact_podium_set": "exact_podium.png",
    "winner_acc": "winner_acc.png",
}


def cmd_evaluate(args) -> None:
    config.ensure_dirs()
    feats = get_feature_dataset()
    keys = _resolve_models(args.model)
    eval_metrics = list(config.EVAL_METRICS)

    results = {}
    for key in keys:
        print(f"\n=== Validare temporală (train ≤ S-2, validare S-1, test S): {MODEL_NAMES[key]} ===")
        results[key] = rolling_cross_year(feats, key, verbose=True)

    print("\n================ SUMAR (medii pe sezoanele de test) ================")
    summary_rows = []
    for key, res in results.items():
        ov = res.overall()
        summary_rows.append({"model": key, **{m: ov[f"mean_{m}"] for m in eval_metrics}})
    summary = pd.DataFrame(summary_rows).round(3)
    print(summary.to_string(index=False))

    summary.to_csv(config.OUTPUT_DIR / "evaluation_summary.csv", index=False)
    for key, res in results.items():
        res.per_season.to_csv(config.OUTPUT_DIR / f"eval_{key}_per_season.csv", index=False)
    print(f"\nCSV-uri salvate în {config.OUTPUT_DIR}")

    if not args.no_plots:
        from .viz import plots

        per_season_by_model = {k: r.per_season for k, r in results.items()}
        for m in eval_metrics:
            plots.plot_metric_per_season(per_season_by_model, m, config.PLOTS_DIR / _METRIC_PLOT_FILE[m])
        plots.plot_metric_summary(summary, eval_metrics, config.PLOTS_DIR / "metric_summary.png")
        for key, res in results.items():
            if res.feature_importances is not None:
                plots.plot_feature_importance(
                    res.feature_names, res.feature_importances, key,
                    config.PLOTS_DIR / f"feature_importance_{key}.png",
                )
        print(f"Grafice salvate în {config.PLOTS_DIR}")


def cmd_tune(args) -> None:
    config.ensure_dirs()
    feats = get_feature_dataset()
    keys = _resolve_models(args.model)

    from .evaluation import tuning

    best = tuning.run_tuning(feats, keys, metric=args.metric, plots=not args.no_plots)

    print("\n============== CONFIGURAȚII OPTIME (selectate pe validare) ==============")
    for key in keys:
        print(f"  {MODEL_NAMES[key]:20s}: {best[key]}")
    print("\nNotă: hiperparametrii din config.py NU au fost modificați. Tabelele complete "
          "(toate configurațiile) sunt în outputs/tuning_<model>.csv.")


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
    model, n_train = _fit_before(feats, key, args.season - 1)
    proba = predict_proba_for_races(model, season_df)

    pred_tbl = predicted_standings(season_df, proba)
    act_tbl = actual_standings(season_df)

    print(f"\n=== Sezonul {args.season} — {MODEL_NAMES[key]} "
          f"(antrenat pe sezoanele ≤ {args.season - 2}, {n_train} rânduri) ===")

    val_df = feats[feats["season"] == args.season - 1]
    if not val_df.empty:
        val_proba = predict_proba_for_races(model, val_df)
        _, y_val, _ = feature_matrix(val_df)
        y_val_pred = (np.asarray(val_proba, dtype=float) >= 0.5).astype(int)
        vpod = metrics.podium_metrics(val_df, val_proba)
        print(f"\nValidare {args.season - 1}: acc={metrics.accuracy(y_val, y_val_pred):.3f} "
              f"podium_hit={vpod['podium_hit_rate']:.3f} exact_podium={vpod['exact_podium_set']:.3f} "
              f"top1={vpod['winner_acc']:.3f}")

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


def _race_metrics(model, race) -> dict:
    proba = predict_proba_for_races(model, race)
    _, y, _ = feature_matrix(race)
    y_pred = (np.asarray(proba, dtype=float) >= 0.5).astype(int)
    pod = metrics.podium_metrics(race, proba)
    return {
        "accuracy": metrics.accuracy(y, y_pred),
        "podium_hit_rate": pod["podium_hit_rate"],
        "exact_podium_set": pod["exact_podium_set"],
        "winner_acc": pod["winner_acc"],
    }


def cmd_insezon(args) -> None:
    config.ensure_dirs()
    feats = get_feature_dataset()
    season = args.season
    keys = _resolve_models(args.model)
    eval_metrics = list(config.EVAL_METRICS)

    season_df = feats[feats["season"] == season]
    if season_df.empty:
        raise SystemExit(f"Nu există date pentru sezonul {season}.")
    rounds = sorted(int(r) for r in season_df["round"].unique())

    rows = []
    for key in keys:
        print(f"\n=== {MODEL_NAMES[key]}: cu vs fără curse in-sezon ({season}) ===")
        base_model, _ = _fit_before(feats, key, season)
        for r in rounds:
            race = season_df[season_df["round"] == r]
            if race.empty:
                continue
            aug_model, _ = _fit_before(feats, key, season, upto_round=r)
            for regim, model in (("fara", base_model), ("cu", aug_model)):
                rows.append({"round": r, "model": key, "regim": regim, **_race_metrics(model, race)})

    df = pd.DataFrame(rows)
    out_csv = config.OUTPUT_DIR / f"insezon_{season}.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n================ SUMAR insezon {season} (medii pe sezon) ================")
    summary = (df.groupby(["model", "regim"])[eval_metrics].mean().round(3))
    print(summary.to_string())
    print("\nΔ (cu − fără):")
    for key in keys:
        cu = summary.loc[(key, "cu")]
        fara = summary.loc[(key, "fara")]
        delta = (cu - fara).round(3)
        print(f"  [{key:6s}] " + "  ".join(f"{m}={delta[m]:+.3f}" for m in eval_metrics))
    print(f"\nCSV salvat în {out_csv}")

    if not args.no_plots:
        from .viz import plots

        for m in eval_metrics:
            plots.plot_insezon_comparison(
                df, m, season, config.PLOTS_DIR / f"insezon_{season}_{_METRIC_PLOT_FILE[m]}"
            )
        print(f"Grafice salvate în {config.PLOTS_DIR}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="f1pred", description="Predicție podium F1 cu 3 modele ML de la zero.")
    sub = p.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("fetch-extras",
                        help="(o singură dată, cu rețea) descarcă tururi/pit-stops/safety car")
    pf.add_argument("--throttle", type=float, default=0.5, help="pauză (s) între runde")
    pf.add_argument("--force", action="store_true", help="reconstruiește de la zero (altfel reia)")
    pf.set_defaults(func=cmd_fetch_extras)

    sub.add_parser("build-data", help="Construiește datasetul din cache-ul FastF1").set_defaults(func=cmd_build_data)

    pt = sub.add_parser("train", help="Antrenează și salvează un model")
    pt.add_argument("--model", default="all", help="logreg | tree | forest | all")
    pt.add_argument("--train-until", type=int, default=config.LAST_SEASON - 1, help="ultimul sezon de antrenament")
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("evaluate", help="Validare temporală train/validare/test pe ani + grafice")
    pe.add_argument("--model", default="all", help="logreg | tree | forest | all")
    pe.add_argument("--no-plots", action="store_true", help="nu genera grafice PNG")
    pe.set_defaults(func=cmd_evaluate)

    ptn = sub.add_parser("tune", help="Optimizează hiperparametrii (grid search pe validare) + grafice")
    ptn.add_argument("--model", default="all", help="logreg | tree | forest | all")
    ptn.add_argument("--metric", default=config.TUNE_METRIC,
                     help="metrica de selecție pe validare "
                          "(accuracy | podium_hit_rate | exact_podium_set | winner_acc)")
    ptn.add_argument("--no-plots", action="store_true", help="nu genera grafice PNG")
    ptn.set_defaults(func=cmd_tune)

    pr = sub.add_parser("predict-race", help="Podiumul prezis al unei curse")
    pr.add_argument("--season", type=int, required=True)
    pr.add_argument("--round", type=int, required=True)
    pr.add_argument("--model", default="forest", help="logreg | tree | forest")
    pr.set_defaults(func=cmd_predict_race)

    ps = sub.add_parser("predict-season", help="Podium per cursă + clasament campionat prezis")
    ps.add_argument("--season", type=int, required=True)
    ps.add_argument("--model", default="forest", help="logreg | tree | forest")
    ps.set_defaults(func=cmd_predict_season)

    pi = sub.add_parser("insezon", help="Compară performanța cu vs fără cursele in-sezon + grafice")
    pi.add_argument("--season", type=int, default=config.LAST_SEASON, help="sezonul analizat (implicit ultimul)")
    pi.add_argument("--model", default="all", help="logreg | tree | forest | all")
    pi.add_argument("--no-plots", action="store_true", help="nu genera grafice PNG")
    pi.set_defaults(func=cmd_insezon)
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
