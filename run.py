"""End-to-end pipeline: data -> features -> model -> validation -> figures/tables.

    python run.py
"""
import json

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from src import viz
from src.data import PROCESSED, ROOT, get_matches, get_nicknames, load_events
from src.evaluate import half_season_reidentification
from src.features import build_player_table
from src.similarity import FEATURES, ScoutingModel

COMP, SEASON = 2, 27            # Premier League 2015/16
OUT = ROOT / "outputs"
FIG = OUT / "figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("1/5 loading events")
    events = load_events(COMP, SEASON)
    matches = get_matches(COMP, SEASON)

    print("2/5 building features")
    players = build_player_table(events, min_minutes=900)
    players["player"] = players.index.to_series().map(get_nicknames(COMP, SEASON)).fillna(players.player)
    players.round(4).to_csv(PROCESSED / "player_features_epl_2015_16.csv")

    print("3/5 fitting model")
    model = ScoutingModel(n_clusters=8).fit(players)
    roles = model.players[["player", "team", "group", "role", "minutes", "pc1", "pc2"]]
    roles.round(3).to_csv(OUT / "player_roles.csv")
    model.role_summary().to_csv(OUT / "role_summary.csv", index=False)
    diag = model.cluster_diagnostics()
    diag.round(4).to_csv(OUT / "k_diagnostics.csv", index=False)

    print("4/5 validating")
    metrics = {k: round(v, 4) if isinstance(v, float) else v
               for k, v in half_season_reidentification(events, matches).items()}
    metrics["n_players"] = int(len(model.players))
    metrics["pca_components"] = int(model.pca.n_components_)
    metrics["pca_variance"] = round(float(model.pca.explained_variance_ratio_.sum()), 4)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("5/5 figures")
    viz.role_map(model, FIG / "role_map.png")
    viz.k_diagnostics(diag, 8, FIG / "k_diagnostics.png")
    for p, slug in [("N'Golo Kanté", "kante"), ("Mesut Özil", "ozil"), ("Riyad Mahrez", "mahrez")]:
        viz.pizza(model, p, FIG / f"pizza_{slug}.png")
    viz.compare(model, "N'Golo Kanté", "Idrissa Gueye", FIG / "compare_kante_gueye.png")
    viz.similar_bars(model, "N'Golo Kanté", 10, FIG / "similar_kante.png", other_teams=True)

    # full similarity export for the interactive explorer
    sims = {}
    for pid, r in model.players.iterrows():
        s = model.similar(r.player, n=10)
        sims[int(pid)] = [[int(i), float(v)] for i, v in s.similarity.items()]
    export = {
        "labels": [viz.FEATURE_LABEL[f] for f in viz.RADAR_FEATURES],
        "groups": {k: len(v) for k, v in viz.RADAR_GROUPS.items()},
        "roles": model.role_summary().to_dict(orient="records"),
        "features": [f for f in viz.RADAR_FEATURES],
        "players": [
            {"id": int(pid), "name": r.player, "team": r.team, "group": r.group, "role": r.role,
             "minutes": round(float(r.minutes)), "pc": [round(float(r.pc1), 3), round(float(r.pc2), 3)],
             "pct": [int(model.percentiles.loc[pid, f]) for f in viz.RADAR_FEATURES],
             "raw": [round(float(model.players.loc[pid, f]), 2) for f in viz.RADAR_FEATURES],
             "similar": sims[int(pid)],
             "emb": [round(float(x), 4) for x in model._unit[model.players.index.get_loc(pid)]]}
            for pid, r in model.players.iterrows()
        ],
        "metrics": metrics,
    }
    (OUT / "explorer_data.json").write_text(json.dumps(export, ensure_ascii=False))
    import runpy
    runpy.run_path(str(ROOT / "explorer" / "build.py"))
    print(json.dumps(metrics, indent=2))
    print(model.role_summary().to_string(index=False))


if __name__ == "__main__":
    main()
