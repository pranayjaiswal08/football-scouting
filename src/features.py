"""Turn raw StatsBomb events into a per-player season feature table.

Every counting stat is expressed per 90 minutes. Defensive volume stats are
also possession-adjusted (PAdj): a defender in a side that sees little of the
ball gets more chances to tackle, so raw counts are scaled by
0.5 / opponent_possession_share for each match.

Pitch coordinates are StatsBomb's 120 x 80 yards, attacking left to right.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

GOAL_X, GOAL_Y = 120.0, 40.0
SET_PIECES = {"Throw-in", "Free Kick", "Goal Kick", "Corner", "Kick Off"}

POSITION_GROUP = {
    "Goalkeeper": "GK",
    "Left Center Back": "CB", "Right Center Back": "CB", "Center Back": "CB",
    "Left Back": "FB", "Right Back": "FB", "Left Wing Back": "FB", "Right Wing Back": "FB",
    "Left Defensive Midfield": "CM", "Right Defensive Midfield": "CM", "Center Defensive Midfield": "CM",
    "Left Center Midfield": "CM", "Right Center Midfield": "CM", "Center Midfield": "CM",
    "Center Attacking Midfield": "AM", "Left Attacking Midfield": "AM", "Right Attacking Midfield": "AM",
    "Left Midfield": "W", "Right Midfield": "W", "Left Wing": "W", "Right Wing": "W",
    "Center Forward": "FW", "Left Center Forward": "FW", "Right Center Forward": "FW",
}
GROUP_LABEL = {"GK": "Goalkeeper", "CB": "Centre-back", "FB": "Full-back", "CM": "Central midfield",
               "AM": "Attacking midfield", "W": "Winger / wide mid", "FW": "Forward"}


# --------------------------------------------------------------------------- minutes
def minutes_played(events: pd.DataFrame) -> pd.DataFrame:
    """Minutes per player per match from Starting XI, substitutions and red cards."""
    ev = events.assign(t=events.minute + events.second / 60)
    match_end = ev.groupby("match_id").t.max()
    rows = []
    for mid, g in ev[ev.type.isin(["Starting XI", "Substitution", "Foul Committed", "Bad Behaviour"])].groupby("match_id"):
        end = match_end[mid]
        on, off, team_of = {}, {}, {}
        for r in g.itertuples():
            if r.type == "Starting XI":
                for p in json.loads(r.lineup):
                    on[p["id"]] = 0.0
                    team_of[p["id"]] = r.team
            elif r.type == "Substitution":
                off[r.player_id] = r.t
                on[r.replacement_id] = r.t
                team_of[r.replacement_id] = r.team
            elif r.card in ("Red Card", "Second Yellow"):
                off[r.player_id] = min(off.get(r.player_id, end), r.t)
        for pid, t_on in on.items():
            rows.append((mid, pid, team_of[pid], max(off.get(pid, end) - t_on, 0.0)))
    out = pd.DataFrame(rows, columns=["match_id", "player_id", "team", "minutes"])
    out["player_id"] = out.player_id.astype(int)
    return out


# --------------------------------------------------------------------------- helpers
def _dist_to_goal(x, y):
    return np.hypot(GOAL_X - x, GOAL_Y - y)


def _progressive(df: pd.DataFrame) -> pd.Series:
    """Ball moved >=25% closer to goal and ends outside the defending 40%."""
    d0 = _dist_to_goal(df.x, df.y)
    d1 = _dist_to_goal(df.end_x, df.end_y)
    return ((d0 - d1) >= 0.25 * d0) & (df.end_x >= 48)


def _in_box(x, y):
    return (x >= 102) & (y >= 18) & (y <= 62)


# --------------------------------------------------------------------------- features
def build_player_table(events: pd.DataFrame, min_minutes: float = 900) -> pd.DataFrame:
    ev = events
    mins = minutes_played(ev)

    # possession share per team per match (pass-count proxy)
    passes_tm = ev[ev.type == "Pass"].groupby(["match_id", "team"]).size().rename("n").reset_index()
    passes_tm["poss"] = passes_tm.n / passes_tm.groupby("match_id").n.transform("sum")
    padj = passes_tm.assign(padj=0.5 / (1 - passes_tm.poss))[["match_id", "team", "padj"]]

    pl = ev[ev.player_id.notna()].copy()
    pl["player_id"] = pl.player_id.astype(int)
    pl = pl.merge(padj, on=["match_id", "team"], how="left")

    passes = pl[pl.type == "Pass"]
    open_play = passes[~passes.sub_type.isin(SET_PIECES)]
    complete = passes.outcome == "Complete"
    op_complete = open_play[open_play.outcome == "Complete"]
    carries = pl[pl.type == "Carry"]
    shots = pl[pl.type == "Shot"]
    np_shots = shots[shots.sub_type != "Penalty"]
    duels = pl[pl.type == "Duel"]
    tackles = duels[duels.sub_type == "Tackle"]
    tackles_won = tackles[tackles.outcome.isin(["Won", "Success In Play", "Success Out"])]
    dribbles = pl[pl.type == "Dribble"]
    receipts = pl[(pl.type == "Ball Receipt*") & (pl.outcome.isna())]

    # expected assists: xG of the shot each key pass led to
    xg_by_keypass = shots.dropna(subset=["key_pass_id"]).set_index("key_pass_id").xg
    passes_xa = passes.assign(xa=passes.id.map(xg_by_keypass).fillna(0.0))

    def count(df):
        return df.groupby("player_id").size()

    def total(df, col):
        return df.groupby("player_id")[col].sum()

    raw = pd.DataFrame({
        # shooting
        "np_xg": total(np_shots, "xg"),
        "np_goals": count(np_shots[np_shots.outcome == "Goal"]),
        "shots": count(np_shots),
        "touches_in_box": count(receipts[_in_box(receipts.x, receipts.y)]),
        # creation
        "xa": total(passes_xa, "xa"),
        "key_passes": count(passes[passes.pass_shot_assist | passes.pass_goal_assist]),
        "assists": count(passes[passes.pass_goal_assist]),
        "through_balls": count(passes[passes.pass_through_ball]),
        "passes_into_box": count(op_complete[_in_box(op_complete.end_x, op_complete.end_y) & ~_in_box(op_complete.x, op_complete.y)]),
        "crosses": count(open_play[open_play.pass_cross]),
        # passing / progression
        "passes": count(passes),
        "passes_completed": count(passes[complete]),
        "progressive_passes": count(op_complete[_progressive(op_complete)]),
        "final_third_passes": count(op_complete[(op_complete.end_x >= 80) & (op_complete.x < 80)]),
        "long_passes": count(passes[np.hypot(passes.end_x - passes.x, passes.end_y - passes.y) >= 32]),
        "progressive_carries": count(carries[_progressive(carries)]),
        "dribbles_completed": count(dribbles[dribbles.outcome == "Complete"]),
        "dribbles_attempted": count(dribbles),
        "dispossessed": count(pl[pl.type.isin(["Dispossessed", "Miscontrol"])]),
        # defending (possession adjusted)
        "tackles_won_padj": tackles_won.groupby("player_id").padj.sum(),
        "interceptions_padj": pl[pl.type == "Interception"].groupby("player_id").padj.sum(),
        "pressures_padj": pl[pl.type == "Pressure"].groupby("player_id").padj.sum(),
        "final_third_pressures": count(pl[(pl.type == "Pressure") & (pl.x >= 80)]),
        "recoveries": count(pl[pl.type == "Ball Recovery"]),
        "clearances": count(pl[pl.type == "Clearance"]),
        "blocks": count(pl[pl.type == "Block"]),
        "aerials_lost": count(duels[duels.sub_type == "Aerial Lost"]),
        "dribbled_past": count(pl[pl.type == "Dribbled Past"]),
    }).fillna(0.0)

    # identity: name, primary position (event-weighted), team with most minutes
    ident = (pl.groupby("player_id")
               .agg(player=("player", "first"), position=("position", lambda s: s.mode().iat[0])))
    ident["group"] = ident.position.map(POSITION_GROUP)
    m = mins.groupby("player_id").agg(minutes=("minutes", "sum"), matches=("match_id", "nunique"))
    team = mins.groupby(["player_id", "team"]).minutes.sum().reset_index() \
               .sort_values("minutes").drop_duplicates("player_id", keep="last").set_index("player_id").team

    df = ident.join(m).join(team).join(raw).fillna(0.0)
    df = df[df.minutes >= min_minutes].copy()

    per90 = df.minutes / 90
    rate_cols = [c for c in raw.columns if c not in ("passes_completed", "dribbles_attempted")]
    out = df[["player", "team", "position", "group", "minutes", "matches"]].copy()
    for c in rate_cols:
        out[c] = df[c] / per90
    out["pass_completion"] = df.passes_completed / df.passes.replace(0, np.nan) * 100
    out["dribble_success"] = df.dribbles_completed / df.dribbles_attempted.replace(0, np.nan) * 100
    out["np_xg_per_shot"] = df.np_xg / df.shots.replace(0, np.nan)
    out = out.fillna(0.0)
    out.index.name = "player_id"
    return out.sort_values("minutes", ascending=False)


if __name__ == "__main__":
    from src.data import PROCESSED, load_events

    from src.data import get_nicknames

    t = build_player_table(load_events(2, 27))
    t["player"] = t.index.to_series().map(get_nicknames(2, 27)).fillna(t.player)
    t.to_csv(PROCESSED / "player_features_epl_2015_16.csv")
    print(t.shape)
    print(t.group.value_counts())
