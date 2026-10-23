"""Download and cache StatsBomb open event data.

StatsBomb open data: https://github.com/statsbomb/open-data
Licence: free for non-commercial use with attribution to StatsBomb.

Usage:
    python -m src.data --competition 2 --season 27      # Premier League 2015/16
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)        # both folders are git-ignored, so create them on a fresh clone
PROCESSED.mkdir(parents=True, exist_ok=True)


def _get_json(url: str, retries: int = 4):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    path = RAW / f"matches_{competition_id}_{season_id}.json"
    if not path.exists():
        path.write_text(json.dumps(_get_json(f"{BASE}/matches/{competition_id}/{season_id}.json")))
    m = json.loads(path.read_text())
    return pd.DataFrame(
        {
            "match_id": x["match_id"],
            "match_date": x["match_date"],
            "home_team": x["home_team"]["home_team_name"],
            "away_team": x["away_team"]["away_team_name"],
            "home_score": x["home_score"],
            "away_score": x["away_score"],
        }
        for x in m
    )


def _flatten_event(e: dict, match_id: int) -> dict:
    """Keep only the fields the feature pipeline needs (keeps the cache small)."""
    t = e["type"]["name"]
    row = {
        "match_id": match_id,
        "id": e["id"],
        "index": e["index"],
        "period": e["period"],
        "minute": e["minute"],
        "second": e["second"],
        "type": t,
        "team": e["team"]["name"],
        "possession_team": e["possession_team"]["name"],
        "player_id": e.get("player", {}).get("id"),
        "player": e.get("player", {}).get("name"),
        "position": e.get("position", {}).get("name"),
        "x": (e.get("location") or [None, None])[0],
        "y": (e.get("location") or [None, None])[1],
        "under_pressure": bool(e.get("under_pressure", False)),
        "outcome": None,
        "end_x": None,
        "end_y": None,
        "sub_type": None,
        "xg": None,
        "pass_cross": False,
        "pass_shot_assist": False,
        "pass_goal_assist": False,
        "pass_through_ball": False,
        "lineup": None,
        "replacement_id": None,
        "key_pass_id": None,
        "card": None,
    }
    d = e.get(t.lower().replace(" ", "_").replace("*", ""), {}) or {}
    if t == "Pass":
        row["outcome"] = d.get("outcome", {}).get("name", "Complete")
        row["end_x"], row["end_y"] = d["end_location"][:2]
        row["sub_type"] = d.get("type", {}).get("name")
        row["pass_cross"] = bool(d.get("cross", False))
        row["pass_shot_assist"] = bool(d.get("shot_assist", False))
        row["pass_goal_assist"] = bool(d.get("goal_assist", False))
        row["pass_through_ball"] = bool(d.get("through_ball", False)) or d.get("technique", {}).get("name") == "Through Ball"
    elif t == "Carry":
        row["end_x"], row["end_y"] = d["end_location"][:2]
    elif t == "Shot":
        row["outcome"] = d["outcome"]["name"]
        row["xg"] = d.get("statsbomb_xg")
        row["key_pass_id"] = d.get("key_pass_id")
        row["sub_type"] = d.get("type", {}).get("name")
    elif t in ("Dribble", "Duel", "Interception", "Ball Receipt*", "50/50"):
        row["outcome"] = (d.get("outcome") or {}).get("name")
        row["sub_type"] = (d.get("type") or {}).get("name")
    elif t == "Goal Keeper":
        row["sub_type"] = (d.get("type") or {}).get("name")
        row["outcome"] = (d.get("outcome") or {}).get("name")
    elif t in ("Foul Committed", "Bad Behaviour"):
        row["card"] = (d.get("card") or {}).get("name")
    elif t == "Starting XI":
        row["lineup"] = json.dumps(
            [{"id": p["player"]["id"], "name": p["player"]["name"], "position": p["position"]["name"]}
             for p in e["tactics"]["lineup"]]
        )
    elif t == "Substitution":
        row["replacement_id"] = d["replacement"]["id"]
        row["lineup"] = json.dumps({"name": d["replacement"]["name"]})
    elif t == "Tactical Shift":
        row["lineup"] = json.dumps(
            [{"id": p["player"]["id"], "name": p["player"]["name"], "position": p["position"]["name"]}
             for p in e["tactics"]["lineup"]]
        )
    return row


def _fetch_match(match_id: int) -> pd.DataFrame:
    cache = RAW / "events" / f"{match_id}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    events = _get_json(f"{BASE}/events/{match_id}.json")
    df = pd.DataFrame(_flatten_event(e, match_id) for e in events)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def load_events(competition_id: int, season_id: int, workers: int = 16) -> pd.DataFrame:
    out = PROCESSED / f"events_{competition_id}_{season_id}.parquet"
    if out.exists():
        return pd.read_parquet(out).sort_values(["match_id", "index"], ignore_index=True)
    matches = get_matches(competition_id, season_id)
    frames = []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_fetch_match, m): m for m in matches.match_id}
        for i, f in enumerate(as_completed(futs), 1):
            frames.append(f.result())
            if i % 50 == 0 or i == len(futs):
                print(f"  downloaded {i}/{len(futs)} matches")
    # parallel downloads finish in random order; sort so every run is byte-identical
    events = pd.concat(frames, ignore_index=True).sort_values(["match_id", "index"], ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out, index=False)
    return events


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", type=int, default=2)
    ap.add_argument("--season", type=int, default=27)
    a = ap.parse_args()
    ev = load_events(a.competition, a.season)
    print(f"{len(ev):,} events, {ev.match_id.nunique()} matches, {ev.player_id.nunique()} players")


def get_nicknames(competition_id: int, season_id: int, workers: int = 16) -> dict[int, str]:
    """Map player_id -> common name (e.g. 'Sergio Agüero') from lineup files."""
    path = RAW / f"nicknames_{competition_id}_{season_id}.json"
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    names: dict[int, str] = {}
    matches = get_matches(competition_id, season_id)
    with ThreadPoolExecutor(workers) as ex:
        for lineups in ex.map(lambda m: _get_json(f"{BASE}/lineups/{m}.json"), matches.match_id):
            for team in lineups:
                for p in team["lineup"]:
                    full = p["player_name"].split()
                    short = full[0] if len(full) == 1 else f"{full[0]} {full[-1]}"
                    names[p["player_id"]] = p.get("player_nickname") or short
    path.write_text(json.dumps(names, ensure_ascii=False))
    return names
