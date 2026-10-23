"""Does the similarity engine capture a *stable* playing style?

Test: split the season into two halves by date. Fit the scaler + PCA on the
first half only, embed both halves, and for each player ask: among all
second-half profiles, is the player's own second-half profile the nearest
neighbour of their first-half profile?

If the features describe style (not noise), players should re-identify
themselves far more often than chance (1 / number of candidates).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_player_table
from src.similarity import ScoutingModel


def half_season_reidentification(events: pd.DataFrame, matches: pd.DataFrame,
                                 min_minutes: float = 600) -> dict:
    cutoff = matches.match_date.sort_values().iloc[len(matches) // 2]
    first = matches.loc[matches.match_date < cutoff, "match_id"]
    h1 = build_player_table(events[events.match_id.isin(first)], min_minutes)
    h2 = build_player_table(events[~events.match_id.isin(first)], min_minutes)
    h1, h2 = h1[h1.group != "GK"], h2[h2.group != "GK"]
    common = h1.index.intersection(h2.index)

    model = ScoutingModel().fit(h1)
    e1 = model.embed(h1.loc[common])
    e2 = model.embed(h2)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    sims = e1 @ e2.T                                    # (common, all H2 players)
    ranks = (-sims).argsort(axis=1)
    own = np.array([h2.index.get_loc(p) for p in common])
    pos = np.array([np.where(ranks[i] == own[i])[0][0] for i in range(len(common))])

    # baseline: same test using position group alone (random player from same group)
    grp_size = h2.group.value_counts()
    baseline_top5 = np.mean([min(5 / grp_size[h1.loc[p, "group"]], 1) for p in common])

    return {
        "cutoff_date": cutoff,
        "players_tested": int(len(common)),
        "candidates": int(len(h2)),
        "top1": float(np.mean(pos == 0)),
        "top5": float(np.mean(pos < 5)),
        "top10": float(np.mean(pos < 10)),
        "median_rank": float(np.median(pos) + 1),
        "random_top5": 5 / len(h2),
        "position_only_top5": float(baseline_top5),
    }
