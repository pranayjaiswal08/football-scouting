"""Player role clustering and similarity search.

Pipeline
--------
1. Select ~24 per-90 style features (outfield players only).
2. log1p the count-rate features to tame long tails, then z-score.
3. PCA, keeping enough components for 85% of variance (removes redundancy,
   e.g. shots / xG / touches in box all measure the same thing).
4. K-Means in PCA space -> playing-style roles.
5. Cosine similarity in PCA space -> "players like X".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = [
    # shooting
    "np_xg", "shots", "touches_in_box",
    # creation
    "xa", "key_passes", "through_balls", "passes_into_box", "crosses",
    # passing & progression
    "passes", "pass_completion", "progressive_passes", "final_third_passes", "long_passes",
    "progressive_carries", "dribbles_completed", "dispossessed",
    # defending
    "tackles_won_padj", "interceptions_padj", "pressures_padj", "final_third_pressures",
    "recoveries", "clearances", "blocks", "aerials_lost",
]
NOT_LOGGED = {"pass_completion"}
# lower is better: flipped when converting to percentiles for radars
NEGATIVE = {"dispossessed", "aerials_lost"}

FEATURE_LABEL = {
    "np_xg": "Non-pen xG", "shots": "Shots", "touches_in_box": "Box touches",
    "xa": "xA", "key_passes": "Key passes", "through_balls": "Through balls",
    "passes_into_box": "Passes into box", "crosses": "Crosses",
    "passes": "Passes", "pass_completion": "Pass %", "progressive_passes": "Prog. passes",
    "final_third_passes": "Final-third entries", "long_passes": "Long passes",
    "progressive_carries": "Prog. carries", "dribbles_completed": "Dribbles won",
    "dispossessed": "Ball security", "tackles_won_padj": "Tackles won (PAdj)",
    "interceptions_padj": "Interceptions (PAdj)", "pressures_padj": "Pressures (PAdj)",
    "final_third_pressures": "High pressures", "recoveries": "Recoveries",
    "clearances": "Clearances", "blocks": "Blocks", "aerials_lost": "Aerial security",
}

# raw (not inverted) wording for negative stats, used when describing cluster centroids
RAW_LABEL = {"dispossessed": "Times dispossessed", "aerials_lost": "Aerial duels lost"}

# Clusters are named by which "anchor" player they contain, so names survive re-fits.
ROLE_ANCHORS = [
    ("Ball-playing centre-back", ["Wes Morgan", "Toby Alderweireld", "Robert Huth"]),
    ("Attacking full-back", ["Aaron Cresswell", "Héctor Bellerín", "Nacho Monreal"]),
    ("Ball-winning midfielder", ["N'Golo Kanté", "Idrissa Gueye", "Francis Coquelin"]),
    ("Deep-lying playmaker", ["Mark Noble", "Danny Drinkwater", "Darren Fletcher"]),
    ("Creative playmaker", ["Mesut Özil", "Christian Eriksen", "Juan Mata"]),
    ("Direct dribbling winger", ["Wilfried Zaha", "Sadio Mané", "Anthony Martial"]),
    ("Pressing second striker", ["Dele Alli", "Troy Deeney", "André Ayew"]),
    ("Penalty-box striker", ["Harry Kane", "Jamie Vardy", "Romelu Lukaku"]),
]


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].astype(float).copy()
    for c in FEATURES:
        if c not in NOT_LOGGED:
            X[c] = np.log1p(X[c])
    return X


@dataclass
class ScoutingModel:
    n_clusters: int = 8
    variance_kept: float = 0.85
    random_state: int = 42
    scaler: StandardScaler = field(default_factory=StandardScaler)
    pca: PCA | None = None
    kmeans: KMeans | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, players: pd.DataFrame) -> "ScoutingModel":
        self.players = players[players.group != "GK"].copy()
        Z = self.scaler.fit_transform(transform_features(self.players))
        full = PCA().fit(Z)
        n = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), self.variance_kept) + 1)
        self.pca = PCA(n, random_state=self.random_state).fit(Z)
        self.embedding = self.pca.transform(Z)
        self.kmeans = KMeans(self.n_clusters, n_init=50, random_state=self.random_state).fit(self.embedding)
        self.players["cluster"] = self.kmeans.labels_
        self.players["role"] = self.players.cluster.map(self._name_clusters())
        self.players[["pc1", "pc2"]] = self.embedding[:, :2]
        self._unit = self.embedding / np.linalg.norm(self.embedding, axis=1, keepdims=True)
        self.percentiles = self._percentiles()
        return self

    def embed(self, players: pd.DataFrame) -> np.ndarray:
        """Project new players (e.g. another season) into the fitted space."""
        return self.pca.transform(self.scaler.transform(transform_features(players)))

    def _name_clusters(self) -> dict[int, str]:
        names: dict[int, str] = {}
        by_name = self.players.set_index("player").cluster
        for role, anchors in ROLE_ANCHORS:
            votes = by_name.reindex(anchors).dropna().astype(int)
            if len(votes):
                c = votes.mode().iat[0]
                names.setdefault(c, role)
        for c in range(self.n_clusters):
            names.setdefault(c, f"Role {c}")
        return names

    def _percentiles(self) -> pd.DataFrame:
        """Percentile rank of each feature within the player's position group."""
        P = self.players.groupby("group")[FEATURES].rank(pct=True) * 100
        for c in NEGATIVE:
            P[c] = self.players.groupby("group")[c].rank(pct=True, ascending=False) * 100
        return P.round(0)

    # ------------------------------------------------------------------ query
    def find(self, name: str) -> int:
        hits = self.players[self.players.player.str.contains(name, case=False, regex=False)]
        if hits.empty:
            raise KeyError(f"No player matching {name!r}")
        return hits.minutes.idxmax()

    def similar(self, name: str, n: int = 10, same_group: bool = False,
                other_teams: bool = False, max_minutes: float | None = None) -> pd.DataFrame:
        """Most similar players by cosine similarity in PCA space (score 0-100)."""
        pid = self.find(name)
        i = self.players.index.get_loc(pid)
        sim = self._unit @ self._unit[i]
        out = self.players.assign(similarity=((sim + 1) / 2 * 100).round(1)).drop(index=pid)
        me = self.players.loc[pid]
        if same_group:
            out = out[out.group == me.group]
        if other_teams:
            out = out[out.team != me.team]
        if max_minutes:
            out = out[out.minutes <= max_minutes]
        cols = ["player", "team", "group", "role", "minutes", "similarity"]
        return out.sort_values("similarity", ascending=False)[cols].head(n)

    def profile(self, name: str) -> pd.Series:
        return self.percentiles.loc[self.find(name)]

    # ------------------------------------------------------------------ diagnostics
    def cluster_diagnostics(self, k_range=range(4, 13)) -> pd.DataFrame:
        rows = []
        for k in k_range:
            km = KMeans(k, n_init=20, random_state=self.random_state).fit(self.embedding)
            rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(self.embedding, km.labels_)})
        return pd.DataFrame(rows)

    def role_summary(self) -> pd.DataFrame:
        Z = pd.DataFrame(self.scaler.transform(transform_features(self.players)),
                         columns=FEATURES, index=self.players.index)
        rows = []
        for role, g in self.players.groupby("role"):
            z = Z.loc[g.index].mean().sort_values(ascending=False)
            rows.append({
                "role": role,
                "players": len(g),
                "defining_traits": ", ".join(RAW_LABEL.get(c, FEATURE_LABEL[c]) for c in z.index[:4]),
                "examples": ", ".join(g.sort_values("minutes", ascending=False).player.head(5)),
            })
        return pd.DataFrame(rows).sort_values("players", ascending=False)
