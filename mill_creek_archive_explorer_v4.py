#!/usr/bin/env python3
"""
Mill Creek Archive Explorer v4

A student-facing PyQt6 GUI for exploring the Mill Creek synthetic humanities
archive with qualitative artifacts and quantitative civic data.

Designed to work with:
    qualitative archive zip/folder:
        artifacts/all_artifacts.jsonl
        metrics/artifact_metrics.csv

    quantitative civic data zip/folder:
        mill_creek_population.csv
        mill_creek_school_records.csv
        mill_creek_city_budget.csv
        etc.

The app brings together the analyses prototyped in Levels 3-13:
    - archive browsing and filtering
    - keyword search
    - semantic search with sentence-transformers, with TF-IDF fallback
    - clustering
    - topic modeling
    - supervised classification
    - people/place/institution networks
    - quantitative civic-data plotting
    - audit/anomaly checks

Run:
    python mill_creek_archive_explorer.py

Recommended packages:
    pip install PyQt6 pandas numpy matplotlib scikit-learn networkx sentence-transformers

sentence-transformers is optional. If it is not installed, semantic search uses
TF-IDF fallback.
"""

from __future__ import annotations

import ast
import html
import json
import math
import os
import re
import shutil
import tempfile
import traceback
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    HAS_NETWORKX = False


APP_TITLE = "Mill Creek Archive Explorer v4"


# ---------------------------------------------------------------------
# Canon / known entities
# ---------------------------------------------------------------------

ACTIVE_RULES = [
    ("Riverfront Trail before 2022", "Riverfront Trail", 2022, False),
    ("Prairie River Clinic before 1954", "Prairie River Clinic", 1954, False),
    ("Mill Creek Historical Society before 1988", "Mill Creek Historical Society", 1988, True),
    ("CobberTech Extension Center before 1998", "CobberTech Extension Center", 1998, False),
    ("Olotón Foods before 2001", "Olotón Foods", 2001, False),
    ("Memorial Field before 1948", "Memorial Field", 1948, False),
    ("Mill Creek High School before 1928", "Mill Creek High School", 1928, False),
    ("Maizey Olotón before 1996", "Maizey Olotón", 1996, False),
    ("Nora Reyes before 2022", "Nora Reyes", 2022, False),
    ("Mill Creek Herald before 1988", "Mill Creek Herald", 1988, False),
]

METADATA_LEAKAGE_PATTERNS = [
    r"\bartifact_id\s*:",
    r"\bsource_type\s*:",
    r"\bprimary_location\s*:",
    r"\bcollection\s*:",
    r"\bdate_label\s*:",
    r"\bBODY\s*:",
    r"\bHEADLINE\s*:",
    r"\bBYLINE\s*:",
    r"\bDATELINE\s*:",
]

KNOWN_PEOPLE = [
    "Jonas Millwright", "Clara Hestvik", "Elsie Bratten", "Mayor Nels Hovland",
    "Peter Lunde", "Mayor Ingrid Lunde", "Ruth Ellingson", "Coach Harold Bratten",
    "Dr. Helen Markham", "Mayor Edwin Rask", "Lena Voss", "Janine Roberts",
    "Professor Carrel Englekorn", "Mayor Naomi Reyes", "Rosa Martinez",
    "Maizey Olotón", "Asha Patel", "David Harlan", "Nora Reyes",
]

KNOWN_INSTITUTIONS = [
    "Mill Creek Chronicle", "Mill Creek Herald", "Mill Creek City Council",
    "Mill Creek Public Schools", "Mill Creek High School",
    "St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission",
    "Mill Creek Women’s Aid Society", "Mill Creek Commercial Club",
    "Prairie River Clinic", "Cobberland County Planning Commission",
    "Mill Creek Historical Society", "CobberTech Extension Center",
    "Cobberland Future Business Leaders Association", "Olotón Foods",
]

KNOWN_LOCATIONS = [
    "Old Mill Bend", "Prairie River bridge", "Downtown/Main Street", "Main Street",
    "Depot District", "South Flats", "North Orchard", "West Rows farms",
    "grain elevator", "Old Grange Hall", "Main Street reading room",
    "St. Ansgar Lutheran Church", "Sacred Heart Catholic Mission",
    "Mill Creek Public Schools", "Mill Creek High School", "Memorial Field",
    "City Hall", "Council Chambers, City Hall", "Prairie River Clinic",
    "Mill Creek Historical Society", "restored depot room", "Olotón Foods",
    "CobberTech Extension Center", "Riverfront Trail",
]

DEFAULT_SEMANTIC_QUERIES = [
    "people losing their homes",
    "public memory after disaster",
    "students learning civic responsibility",
    "churches acting as community infrastructure",
    "farm economy and local identity",
    "the river becoming a public place",
    "women's civic labor",
    "the town remembering itself",
    "downtown decline and redevelopment",
]


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def safe_read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_zip_to_temp(path: Path) -> Path:
    temp = Path(tempfile.mkdtemp(prefix="mill_creek_gui_"))
    with zipfile.ZipFile(path) as z:
        z.extractall(temp)
    return temp


def find_first(root: Path, filename: str) -> Path | None:
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def find_civic_csv_dir(root: Path) -> Path | None:
    csvs = list(root.rglob("*.csv"))
    if not csvs:
        return None
    for p in csvs:
        if p.name == "mill_creek_population.csv":
            return p.parent
    for p in csvs:
        if p.name.startswith("mill_creek_"):
            return p.parent
    return csvs[0].parent


def clean_text(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def parse_maybe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, float) and math.isnan(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    if ";" in s:
        return [x.strip() for x in s.split(";") if x.strip()]
    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]
    return [s]


def text_contains(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def make_vectorizer(max_features: int = 8000, min_df: int = 2) -> TfidfVectorizer:
    extra_stop = {
        "mill", "creek", "cobberland", "prairie", "local", "town",
        "said", "mr", "mrs", "ms", "according", "reported"
    }
    return TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=0.86,
        ngram_range=(1, 2),
        stop_words=list(set(ENGLISH_STOP_WORDS).union(extra_stop)),
        lowercase=True,
        strip_accents="unicode",
    )


def set_table(df: pd.DataFrame, table: QTableWidget, max_rows: int = 500) -> None:
    table.clear()
    if df is None or df.empty:
        table.setRowCount(0)
        table.setColumnCount(0)
        return

    view = df.head(max_rows).copy()
    table.setRowCount(len(view))
    table.setColumnCount(len(view.columns))
    table.setHorizontalHeaderLabels([str(c) for c in view.columns])

    for r in range(len(view)):
        for c, col in enumerate(view.columns):
            val = view.iloc[r, c]
            item = QTableWidgetItem("" if pd.isna(val) else str(val))
            item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            table.setItem(r, c, item)

    table.resizeColumnsToContents()
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)


def dataframe_to_plain(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "(no rows)"
    return df.head(max_rows).to_string(index=False)


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def clear(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.draw()

    def plot_bar(self, labels, values, title: str, xlabel: str = "", ylabel: str = ""):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.bar([str(x) for x in labels], values)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        self.fig.tight_layout()
        self.draw()

    def plot_line(self, x, y, title: str, xlabel: str = "", ylabel: str = ""):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(x, y, marker="o")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.fig.tight_layout()
        self.draw()

    def plot_scatter(self, x, y, title: str, xlabel: str = "", ylabel: str = "", labels=None):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if labels is None:
            ax.scatter(x, y)
        else:
            codes, uniques = pd.factorize(labels)
            ax.scatter(x, y, c=codes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.fig.tight_layout()
        self.draw()

    def plot_heatmap(self, values, xticks, yticks, title: str):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        im = ax.imshow(values, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels([str(x) for x in xticks], rotation=30, ha="right")
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels([str(y) for y in yticks])
        self.fig.colorbar(im, ax=ax)
        self.fig.tight_layout()
        self.draw()

    def plot_points(self, x, y, title: str, xlabel: str = "", ylabel: str = "", connect: bool = False):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.scatter(x, y)
        if connect:
            ax.plot(x, y, alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.fig.tight_layout()
        self.draw()

    def plot_dual_storyline(self, df: pd.DataFrame, x_col: str, y1_col: str, y2_col: str, title: str, connect: bool = True):
        self.fig.clear()
        ax1 = self.fig.add_subplot(111)
        if df is None or df.empty or x_col not in df.columns or y1_col not in df.columns:
            ax1.text(0.5, 0.5, "No data available for this storyline", ha="center", va="center")
            self.draw()
            return

        sub = df[[x_col, y1_col] + ([y2_col] if y2_col in df.columns and y2_col != y1_col else [])].dropna().sort_values(x_col)
        x = sub[x_col]
        ax1.scatter(x, sub[y1_col], label=y1_col)
        if connect:
            ax1.plot(x, sub[y1_col], alpha=0.75)
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y1_col)

        if y2_col in sub.columns and y2_col != y1_col:
            ax2 = ax1.twinx()
            ax2.scatter(x, sub[y2_col], marker="x", label=y2_col)
            if connect:
                ax2.plot(x, sub[y2_col], alpha=0.55)
            ax2.set_ylabel(y2_col)
        ax1.set_title(title)
        self.fig.tight_layout()
        self.draw()

    def plot_network_graph(self, edges_df: pd.DataFrame, center_phrase: str = "", max_edges: int = 35):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if edges_df is None or edges_df.empty:
            ax.text(0.5, 0.5, "No network edges to display", ha="center", va="center")
            ax.axis("off")
            self.draw()
            return

        if not HAS_NETWORKX:
            ax.text(0.5, 0.5, "networkx is not installed", ha="center", va="center")
            ax.axis("off")
            self.draw()
            return

        df = edges_df.sort_values("weight", ascending=False).head(max_edges).copy()
        G = nx.Graph()
        center_nodes = set()
        for _, e in df.iterrows():
            a = str(e["source_label"])
            b = str(e["target_label"])
            w = float(e["weight"])
            G.add_edge(a, b, weight=w)
            if center_phrase and center_phrase.lower() in a.lower():
                center_nodes.add(a)
            if center_phrase and center_phrase.lower() in b.lower():
                center_nodes.add(b)

        if G.number_of_edges() == 0:
            ax.text(0.5, 0.5, "No network edges to display", ha="center", va="center")
            ax.axis("off")
            self.draw()
            return

        pos = nx.spring_layout(G, seed=1776, k=0.9)
        weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
        max_w = max(weights) if weights else 1.0
        widths = [0.7 + 4.0 * (w / max_w) for w in weights]

        node_sizes = []
        for n in G.nodes():
            deg_w = sum(G[n][nbr].get("weight", 1.0) for nbr in G.neighbors(n))
            size = 350 + 30 * deg_w
            if n in center_nodes:
                size *= 1.5
            node_sizes.append(size)

        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.45, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        title = f"Direct Ego Network: {center_phrase}" if center_phrase else "Entity Network"
        ax.set_title(title)
        ax.axis("off")
        self.fig.tight_layout()
        self.draw()


# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------

@dataclass
class ArchiveData:
    artifacts: pd.DataFrame | None = None
    civic: dict[str, pd.DataFrame] | None = None
    archive_root: Path | None = None
    civic_root: Path | None = None
    temp_dirs: list[Path] | None = None

    def __post_init__(self):
        if self.civic is None:
            self.civic = {}
        if self.temp_dirs is None:
            self.temp_dirs = []

    def has_archive(self) -> bool:
        return self.artifacts is not None and not self.artifacts.empty

    def has_civic(self) -> bool:
        return bool(self.civic)


class DataManager:
    def __init__(self):
        self.data = ArchiveData()

    def load_archive(self, path: Path) -> pd.DataFrame:
        root = path
        if path.is_file() and path.suffix.lower() == ".zip":
            root = extract_zip_to_temp(path)
            self.data.temp_dirs.append(root)

        artifacts_path = find_first(root, "all_artifacts.jsonl")
        if artifacts_path is None:
            raise FileNotFoundError("Could not find artifacts/all_artifacts.jsonl in that archive.")

        metrics_path = find_first(root, "artifact_metrics.csv")
        art = pd.DataFrame(safe_read_jsonl(artifacts_path))

        if metrics_path is not None:
            metrics = pd.read_csv(metrics_path)
            useful_cols = [
                "artifact_id", "status", "warning_count", "body_word_count",
                "wall_clock_s", "eval_tokens_per_s", "error"
            ]
            useful_cols = [c for c in useful_cols if c in metrics.columns]
            art = art.merge(metrics[useful_cols], on="artifact_id", how="left")

        for col in [
            "artifact_id", "year", "title", "headline", "title_seed", "body",
            "category", "source_type", "collection", "location", "topic",
            "date_label", "people", "institutions", "locations", "status",
            "warning_count", "body_word_count"
        ]:
            if col not in art.columns:
                art[col] = ""

        art["display_title"] = art["title"].fillna("").astype(str)
        mask = art["display_title"].str.len() == 0
        art.loc[mask, "display_title"] = art.loc[mask, "headline"].fillna("").astype(str)
        mask = art["display_title"].str.len() == 0
        art.loc[mask, "display_title"] = art.loc[mask, "title_seed"].fillna("").astype(str)

        art["year"] = pd.to_numeric(art["year"], errors="coerce").fillna(0).astype(int)
        art["decade"] = (art["year"] // 10) * 10
        art["body"] = art["body"].fillna("").astype(str)
        art["body_preview"] = art["body"].apply(lambda s: clean_text(s)[:650])
        art["computed_word_count"] = art["body"].str.findall(r"\b[\w’'-]+\b").str.len()
        art["body_word_count"] = pd.to_numeric(art["body_word_count"], errors="coerce").fillna(art["computed_word_count"]).astype(int)
        art["status"] = art["status"].replace("", "UNKNOWN").fillna("UNKNOWN")
        art["warning_count"] = pd.to_numeric(art["warning_count"], errors="coerce").fillna(0).astype(int)

        art["analysis_text"] = (
            art["display_title"].fillna("").astype(str) + "\n" +
            art["category"].fillna("").astype(str) + " " +
            art["source_type"].fillna("").astype(str) + " " +
            art["location"].fillna("").astype(str) + " " +
            art["topic"].fillna("").astype(str) + "\n" +
            art["body"].fillna("").astype(str)
        )

        self.data.artifacts = art
        self.data.archive_root = root
        return art

    def load_civic(self, path: Path) -> dict[str, pd.DataFrame]:
        root = path
        if path.is_file() and path.suffix.lower() == ".zip":
            root = extract_zip_to_temp(path)
            self.data.temp_dirs.append(root)

        csv_dir = find_civic_csv_dir(root)
        if csv_dir is None:
            raise FileNotFoundError("Could not find civic CSV files in that folder or zip.")

        civic = {}
        for csv_path in sorted(csv_dir.glob("*.csv")):
            try:
                civic[csv_path.stem] = pd.read_csv(csv_path)
            except Exception:
                pass

        if not civic:
            raise FileNotFoundError("No readable CSV files were found.")

        self.data.civic = civic
        self.data.civic_root = root
        return civic

    def cleanup(self):
        for d in self.data.temp_dirs:
            try:
                shutil.rmtree(d)
            except Exception:
                pass


# ---------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------

def keyword_search(art: pd.DataFrame, query: str, top_k: int = 50) -> pd.DataFrame:
    if not query.strip():
        return pd.DataFrame()
    qterms = [t.lower() for t in re.findall(r"\w+", query)]
    rows = []
    for _, row in art.iterrows():
        text = str(row["analysis_text"]).lower()
        score = sum(text.count(t) for t in qterms)
        if score > 0:
            rows.append({
                "score": score,
                "artifact_id": row["artifact_id"],
                "year": row["year"],
                "category": row["category"],
                "source_type": row["source_type"],
                "display_title": row["display_title"],
                "body_preview": row["body_preview"],
            })
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k) if rows else pd.DataFrame()


def build_semantic_index(art: pd.DataFrame, use_embeddings: bool = True):
    docs = art["analysis_text"].fillna("").astype(str).tolist()

    if use_embeddings:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model.encode(docs, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            return {"mode": "sentence_transformers", "model": model, "embeddings": emb}
        except Exception:
            pass

    vectorizer = make_vectorizer(max_features=9000, min_df=2)
    X = vectorizer.fit_transform(docs)
    return {"mode": "tfidf_fallback", "vectorizer": vectorizer, "matrix": X}


def semantic_search(index: dict[str, Any], art: pd.DataFrame, query: str, top_k: int = 12) -> pd.DataFrame:
    if not query.strip():
        return pd.DataFrame()

    if index["mode"] == "sentence_transformers":
        model = index["model"]
        emb = index["embeddings"]
        q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims = emb @ q[0]
    else:
        vectorizer = index["vectorizer"]
        X = index["matrix"]
        q = vectorizer.transform([query])
        sims = cosine_similarity(X, q).ravel()

    order = np.argsort(sims)[::-1][:top_k]
    rows = []
    for rank, idx in enumerate(order, start=1):
        row = art.iloc[int(idx)]
        rows.append({
            "rank": rank,
            "score": float(sims[idx]),
            "artifact_id": row["artifact_id"],
            "year": row["year"],
            "category": row["category"],
            "source_type": row["source_type"],
            "display_title": row["display_title"],
            "body_preview": row["body_preview"],
        })
    return pd.DataFrame(rows)


def run_clustering(art: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    vectorizer = make_vectorizer(max_features=9000, min_df=2)
    X = vectorizer.fit_transform(art["analysis_text"].fillna("").astype(str))
    model = KMeans(n_clusters=k, random_state=1776, n_init=10)
    labels = model.fit_predict(X)

    terms = np.array(vectorizer.get_feature_names_out())
    top_rows = []
    for cid, center in enumerate(model.cluster_centers_):
        top_idx = np.argsort(center)[::-1][:18]
        top_rows.append({
            "cluster": cid,
            "n_artifacts": int((labels == cid).sum()),
            "top_terms": ", ".join(terms[top_idx]),
        })
    top_terms = pd.DataFrame(top_rows)

    svd = TruncatedSVD(n_components=2, random_state=1776)
    coords = svd.fit_transform(X)
    out = art[["artifact_id", "year", "decade", "category", "source_type", "display_title", "body_preview"]].copy()
    out["cluster"] = labels
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    return out, top_terms, coords


def run_topics(art: pd.DataFrame, n_topics: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vectorizer = make_vectorizer(max_features=9000, min_df=2)
    X = vectorizer.fit_transform(art["analysis_text"].fillna("").astype(str))
    model = NMF(n_components=n_topics, init="nndsvda", random_state=1776, max_iter=800)
    W = model.fit_transform(X)
    H = model.components_
    terms = np.array(vectorizer.get_feature_names_out())

    term_rows = []
    for topic, weights in enumerate(H):
        idx = np.argsort(weights)[::-1][:20]
        term_rows.append({
            "topic": topic,
            "n_dominant": int((W.argmax(axis=1) == topic).sum()),
            "top_terms": ", ".join(terms[idx]),
        })
    top_terms = pd.DataFrame(term_rows)

    assign = art[["artifact_id", "year", "decade", "category", "source_type", "display_title", "body_preview"]].copy()
    assign["topic"] = W.argmax(axis=1)
    assign["topic_weight"] = W.max(axis=1)

    reps = []
    for topic in range(n_topics):
        order = np.argsort(W[:, topic])[::-1][:8]
        for rank, idx in enumerate(order, start=1):
            row = art.iloc[int(idx)]
            reps.append({
                "topic": topic,
                "rank": rank,
                "weight": float(W[idx, topic]),
                "artifact_id": row["artifact_id"],
                "year": row["year"],
                "category": row["category"],
                "display_title": row["display_title"],
                "body_preview": row["body_preview"],
            })
    reps = pd.DataFrame(reps)
    return assign, top_terms, reps



def source_group_from_source_type(source_type: Any, category: Any) -> str:
    """Broader source grouping for more stable student-facing classification tasks."""
    st = str(source_type or "").lower()
    cat = str(category or "").lower()

    if cat == "oral_history" or "oral" in st or "transcript" in st:
        return "oral_history"
    if cat == "photo_caption" or "photo" in st or "caption" in st:
        return "photo_archive"
    if cat == "council_minutes" or "minutes" in st:
        return "minutes_government"
    if cat == "religion" or "sermon" in st or "church" in st or "service" in st or "religious" in st:
        return "religious_community"
    if cat == "school" or "school" in st or "club" in st or "sports" in st:
        return "school_record"
    if "notice" in st or "classified" in st or "advertisement" in st or "public_notice" in st:
        return "notice_ad"
    if "market" in st or "business" in st or "price" in st:
        return "business_record"
    if cat == "newspaper" or "feature" in st or "report" in st or "article" in st:
        return "newspaper_article"
    return "other"


def decade_group_from_year(year: Any) -> str:
    y = int(year)
    if y < 1940:
        return "1920-1939"
    if y < 1960:
        return "1940-1959"
    if y < 1980:
        return "1960-1979"
    if y < 2000:
        return "1980-1999"
    return "2000-2026"


def theme_label(text: str, theme: str) -> str:
    t = str(text or "").lower()
    if theme == "flood_binary":
        words = [
            "flood", "river", "water", "south flats", "sandbag", "buyout",
            "crest", "bridge closure", "relocation", "levee", "washed"
        ]
        return "flood_related" if any(w in t for w in words) else "not_flood_related"
    if theme == "memory_binary":
        words = [
            "memory", "remember", "historical society", "archive", "depot display",
            "oral history", "photograph", "donated", "preservation", "exhibit",
            "commemorate", "dedication"
        ]
        return "memory_related" if any(w in t for w in words) else "not_memory_related"
    if theme == "downtown_binary":
        words = [
            "downtown", "main street", "depot district", "storefront", "commercial",
            "redevelopment", "bypass", "public comment", "business district"
        ]
        return "downtown_related" if any(w in t for w in words) else "not_downtown_related"
    return "unknown"


def make_classification_labels(art: pd.DataFrame, task: str) -> pd.Series:
    if task == "Category":
        return art["category"].astype(str)
    if task == "Source group":
        return art.apply(lambda r: source_group_from_source_type(r.get("source_type", ""), r.get("category", "")), axis=1)
    if task == "Historical period":
        return art["year"].apply(decade_group_from_year)
    if task == "Flood-related":
        return art["analysis_text"].apply(lambda s: theme_label(s, "flood_binary"))
    if task == "Memory/archive-related":
        return art["analysis_text"].apply(lambda s: theme_label(s, "memory_binary"))
    if task == "Downtown/redevelopment-related":
        return art["analysis_text"].apply(lambda s: theme_label(s, "downtown_binary"))
    return art["category"].astype(str)


def run_classification_task(art: pd.DataFrame, task: str, body_only: bool = False) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    df = art.copy()
    text_col = "body" if body_only else "analysis_text"
    df["_label"] = make_classification_labels(df, task).astype(str)

    counts = df["_label"].value_counts()
    valid_classes = counts[counts >= 2].index
    df = df[df["_label"].isin(valid_classes)].copy()
    y = df["_label"].astype(str)

    if len(valid_classes) < 2:
        raise ValueError("Need at least two classes with at least two examples each.")

    # Avoid stratified split failure for very small classes.
    stratify = y if y.value_counts().min() >= 2 else None

    vectorizer = make_vectorizer(max_features=9000, min_df=2)
    X = vectorizer.fit_transform(df[text_col].fillna("").astype(str))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.25, random_state=1776, stratify=stratify
    )

    clf = LogisticRegression(max_iter=1500, solver="lbfgs")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    report = classification_report(y_test, pred, zero_division=0)
    labels = sorted(y.unique())
    cm = pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels)

    test_df = df.loc[idx_test, ["artifact_id", "year", "category", "source_type", "display_title", "body_preview"]].copy()
    test_df["true_label"] = y_test.values
    test_df["predicted_label"] = pred
    test_df["correct"] = test_df["true_label"].astype(str) == test_df["predicted_label"].astype(str)

    return report, cm, test_df.sort_values("correct")



def run_category_classification(art: pd.DataFrame, body_only: bool = False) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    return run_classification_task(art, "Category", body_only=body_only)


def extract_entities(row: pd.Series, civic: dict[str, pd.DataFrame] | None = None) -> dict[str, list[str]]:
    civic = civic or {}
    text = " ".join([
        str(row.get("display_title", "")),
        str(row.get("topic", "")),
        str(row.get("location", "")),
        str(row.get("body", "")),
    ])

    people = set(parse_maybe_list(row.get("people", "")))
    institutions = set(parse_maybe_list(row.get("institutions", "")))
    locations = set(parse_maybe_list(row.get("locations", "")))

    for person in KNOWN_PEOPLE:
        if text_contains(text, person):
            people.add(person)

    all_institutions = set(KNOWN_INSTITUTIONS)
    if "mill_creek_institution_timeline" in civic:
        all_institutions.update(civic["mill_creek_institution_timeline"]["institution"].dropna().astype(str).tolist())
    for inst in all_institutions:
        if text_contains(text, inst):
            institutions.add(inst)

    all_locations = set(KNOWN_LOCATIONS)
    if "mill_creek_place_gazetteer" in civic:
        all_locations.update(civic["mill_creek_place_gazetteer"]["name"].dropna().astype(str).tolist())
    for loc in all_locations:
        if text_contains(text, loc):
            locations.add(loc)

    return {
        "person": sorted(x for x in people if x),
        "institution": sorted(x for x in institutions if x),
        "location": sorted(x for x in locations if x),
    }


def run_network(art: pd.DataFrame, civic: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edge_counts = Counter()
    node_types = {}
    entity_artifacts = defaultdict(set)

    for _, row in art.iterrows():
        entities = extract_entities(row, civic)
        all_entities = []
        for typ, names in entities.items():
            for name in names:
                key = f"{typ}:{name}"
                node_types[key] = typ
                entity_artifacts[key].add(row["artifact_id"])
                all_entities.append(key)
        for a, b in combinations(sorted(set(all_entities)), 2):
            edge_counts[(a, b)] += 1

    node_rows = []
    for key, artifacts in entity_artifacts.items():
        typ, label = key.split(":", 1)
        node_rows.append({
            "node_id": key,
            "label": label,
            "node_type": typ,
            "artifact_count": len(artifacts),
        })
    nodes = pd.DataFrame(node_rows)

    edge_rows = []
    for (a, b), w in edge_counts.items():
        edge_rows.append({
            "source": a,
            "target": b,
            "source_label": a.split(":", 1)[1],
            "target_label": b.split(":", 1)[1],
            "weight": w,
        })
    edges = pd.DataFrame(edge_rows).sort_values("weight", ascending=False) if edge_rows else pd.DataFrame()

    if not nodes.empty and not edges.empty:
        degree = Counter()
        weighted = Counter()
        for _, e in edges.iterrows():
            degree[e["source"]] += 1
            degree[e["target"]] += 1
            weighted[e["source"]] += e["weight"]
            weighted[e["target"]] += e["weight"]
        nodes["degree"] = nodes["node_id"].map(degree).fillna(0).astype(int)
        nodes["weighted_degree"] = nodes["node_id"].map(weighted).fillna(0).astype(int)
        top_nodes = nodes.sort_values(["weighted_degree", "degree"], ascending=False)
    else:
        top_nodes = nodes

    return nodes, edges, top_nodes


def direct_ego_edges(edges: pd.DataFrame, center: str) -> pd.DataFrame:
    if edges is None or edges.empty or not center.strip():
        return pd.DataFrame()
    mask = (
        edges["source_label"].astype(str).str.contains(center, case=False, regex=False) |
        edges["target_label"].astype(str).str.contains(center, case=False, regex=False)
    )
    return edges[mask].sort_values("weight", ascending=False)


def run_audit(art: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flags = []
    for _, row in art.iterrows():
        text = " ".join([
            str(row.get("display_title", "")),
            str(row.get("topic", "")),
            str(row.get("location", "")),
            str(row.get("body", "")),
        ])
        year = int(row["year"])

        for rule, phrase, start_year, photo_soft in ACTIVE_RULES:
            if year < start_year and phrase.lower() in text.lower():
                severity = "soft" if (photo_soft and row.get("category") == "photo_caption") else "hard"
                flags.append({
                    "artifact_id": row["artifact_id"],
                    "year": year,
                    "category": row["category"],
                    "flag_type": "timeline",
                    "severity": severity,
                    "rule": rule,
                    "detail": f"Found {phrase!r} before {start_year}.",
                    "display_title": row["display_title"],
                    "body_preview": row["body_preview"],
                })

        for pattern in METADATA_LEAKAGE_PATTERNS:
            if re.search(pattern, str(row.get("body", "")), flags=re.I):
                flags.append({
                    "artifact_id": row["artifact_id"],
                    "year": year,
                    "category": row["category"],
                    "flag_type": "metadata_leakage",
                    "severity": "hard",
                    "rule": pattern,
                    "detail": "Metadata-like label appears inside body.",
                    "display_title": row["display_title"],
                    "body_preview": row["body_preview"],
                })

        if str(row.get("status", "")) == "WARN" or int(row.get("warning_count", 0) or 0) > 0:
            flags.append({
                "artifact_id": row["artifact_id"],
                "year": year,
                "category": row["category"],
                "flag_type": "generation_status",
                "severity": "soft",
                "rule": "generation warning",
                "detail": f"status={row.get('status')}; warning_count={row.get('warning_count')}",
                "display_title": row["display_title"],
                "body_preview": row["body_preview"],
            })

    flags_df = pd.DataFrame(flags)

    repeated = (
        art.groupby(["display_title", "category"])
        .agg(
            count=("artifact_id", "count"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            sample_artifacts=("artifact_id", lambda s: "; ".join(list(s.astype(str))[:10])),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    repeated = repeated[repeated["count"] >= 3].copy()

    category_summary = art.groupby("category").agg(
        count=("artifact_id", "count"),
        unique_titles=("display_title", "nunique"),
        mean_words=("body_word_count", "mean"),
        warn_count=("status", lambda s: int((s == "WARN").sum())),
    ).reset_index()
    category_summary["title_repetition_ratio"] = 1 - category_summary["unique_titles"] / category_summary["count"]

    summary = pd.DataFrame([
        {"metric": "artifacts", "value": len(art)},
        {"metric": "flags", "value": len(flags_df)},
        {"metric": "hard_flags", "value": int((flags_df["severity"] == "hard").sum()) if not flags_df.empty else 0},
        {"metric": "soft_flags", "value": int((flags_df["severity"] == "soft").sum()) if not flags_df.empty else 0},
        {"metric": "repeated_titles", "value": len(repeated)},
        {"metric": "metadata_leakage", "value": int((flags_df["flag_type"] == "metadata_leakage").sum()) if not flags_df.empty else 0},
    ])

    return flags_df, repeated, category_summary, summary



# ---------------------------------------------------------------------
# Civic storyline helpers
# ---------------------------------------------------------------------

THEME_KEYWORDS = {
    "flood": [
        "flood", "river", "south flats", "sandbag", "buyout", "crest",
        "water", "relocation", "levee", "bridge closure"
    ],
    "memory": [
        "memory", "remember", "historical society", "archive", "depot display",
        "oral history", "photograph", "donated", "preservation", "exhibit",
        "commemorate", "dedication"
    ],
    "downtown": [
        "downtown", "main street", "depot district", "storefront", "commercial",
        "redevelopment", "bypass", "business district", "public comment"
    ],
    "school": [
        "school", "student", "pupil", "class", "teacher", "program",
        "graduation", "yearbook", "team", "club"
    ],
    "business": [
        "business", "store", "market", "grain", "cream", "egg", "prices",
        "olotón", "main street", "commercial", "elevator"
    ],
}


def find_civic_table(civic: dict[str, pd.DataFrame], contains: str) -> pd.DataFrame | None:
    contains = contains.lower()
    for name, df in civic.items():
        if contains in name.lower():
            return df
    return None


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in df.columns:
        lc = str(c).lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c
    return None


def yearly_theme_counts(art: pd.DataFrame, theme: str) -> pd.DataFrame:
    words = THEME_KEYWORDS.get(theme, [])
    if not words:
        return pd.DataFrame(columns=["year", f"{theme}_artifact_mentions"])
    texts = art["analysis_text"].fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=art.index)
    for w in words:
        mask = mask | texts.str.contains(w.lower(), regex=False, na=False)
    out = (
        art.loc[mask]
        .groupby("year")
        .agg(**{f"{theme}_artifact_mentions": ("artifact_id", "count")})
        .reset_index()
    )
    return out


def artifact_counts_by_year(art: pd.DataFrame) -> pd.DataFrame:
    return art.groupby("year").agg(total_artifacts=("artifact_id", "count")).reset_index()


def merge_yearly(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left is None or left.empty:
        return right.copy()
    if right is None or right.empty:
        return left.copy()
    return left.merge(right, on="year", how="outer").sort_values("year").fillna(0)


def build_storyline_dataframe(art: pd.DataFrame, civic: dict[str, pd.DataFrame], storyline: str) -> tuple[pd.DataFrame, str, str, str]:
    """Return df, x_col, y1_col, y2_col for a guided storyline."""
    base = artifact_counts_by_year(art)

    if storyline == "South Flats: flood damage, buyouts, and text mentions":
        flood = find_civic_table(civic, "flood")
        theme = yearly_theme_counts(art, "flood")
        df = merge_yearly(base, theme)
        if flood is not None and "year" in flood.columns:
            cols = ["year"]
            for cand in [
                "spring_crest_ft", "homes_affected", "sandbags_filled",
                "buyout_acres", "bridge_closures"
            ]:
                c = first_existing_col(flood, [cand])
                if c and c not in cols:
                    cols.append(c)
            df = df.merge(flood[cols], on="year", how="outer").sort_values("year").fillna(0)
        y1 = first_existing_col(df, ["homes_affected", "buyout_acres", "spring_crest_ft"]) or "total_artifacts"
        y2 = "flood_artifact_mentions" if "flood_artifact_mentions" in df.columns else "total_artifacts"
        return df, "year", y1, y2

    if storyline == "Public memory: archive work and memory language":
        budget = find_civic_table(civic, "budget")
        theme = yearly_theme_counts(art, "memory")
        df = merge_yearly(base, theme)
        if budget is not None and "year" in budget.columns:
            cols = ["year"]
            for cand in [
                "historical_society_budget", "archive_budget", "parks_and_culture",
                "library_historical_budget", "civic_projects_budget"
            ]:
                c = first_existing_col(budget, [cand])
                if c and c not in cols:
                    cols.append(c)
            if len(cols) > 1:
                df = df.merge(budget[cols], on="year", how="outer").sort_values("year").fillna(0)
        y1 = first_existing_col(df, ["historical_society_budget", "archive_budget", "parks_and_culture"]) or "memory_artifact_mentions"
        y2 = "memory_artifact_mentions"
        return df, "year", y1, y2

    if storyline == "Downtown and redevelopment: business counts and text":
        business = find_civic_table(civic, "business")
        theme = yearly_theme_counts(art, "downtown")
        df = merge_yearly(base, theme)
        if business is not None and "year" in business.columns:
            cols = ["year"]
            for cand in [
                "downtown_businesses", "main_street_businesses", "total_businesses",
                "retail_businesses", "vacant_storefronts"
            ]:
                c = first_existing_col(business, [cand])
                if c and c not in cols:
                    cols.append(c)
            if len(cols) > 1:
                df = df.merge(business[cols], on="year", how="outer").sort_values("year").fillna(0)
        y1 = first_existing_col(df, ["downtown_businesses", "main_street_businesses", "total_businesses"]) or "downtown_artifact_mentions"
        y2 = "downtown_artifact_mentions"
        return df, "year", y1, y2

    if storyline == "Schools: enrollment, graduation, and school artifacts":
        school = find_civic_table(civic, "school")
        theme = yearly_theme_counts(art, "school")
        df = merge_yearly(base, theme)
        if school is not None and "year" in school.columns:
            cols = ["year"]
            for cand in [
                "enrollment", "graduation_rate", "graduates", "attendance_rate",
                "student_club_participation", "business_club_members"
            ]:
                c = first_existing_col(school, [cand])
                if c and c not in cols:
                    cols.append(c)
            if len(cols) > 1:
                df = df.merge(school[cols], on="year", how="outer").sort_values("year").fillna(0)
        y1 = first_existing_col(df, ["enrollment", "graduation_rate", "graduates"]) or "school_artifact_mentions"
        y2 = "school_artifact_mentions"
        return df, "year", y1, y2

    if storyline == "Land use: South Flats to Riverfront Trail":
        land = find_civic_table(civic, "land_use")
        theme = yearly_theme_counts(art, "flood")
        df = merge_yearly(base, theme)
        if land is not None and "year" in land.columns:
            cols = ["year"]
            for cand in [
                "south_flats_households", "park_acres", "trail_acres",
                "buyout_acres", "residential_acres", "floodplain_open_space_acres"
            ]:
                c = first_existing_col(land, [cand])
                if c and c not in cols:
                    cols.append(c)
            if len(cols) > 1:
                df = df.merge(land[cols], on="year", how="outer").sort_values("year").fillna(0)
        y1 = first_existing_col(df, ["south_flats_households", "residential_acres", "park_acres"]) or "flood_artifact_mentions"
        y2 = first_existing_col(df, ["park_acres", "trail_acres", "flood_artifact_mentions"]) or "flood_artifact_mentions"
        return df, "year", y1, y2

    if storyline == "Population and civic change":
        pop = find_civic_table(civic, "population")
        theme = yearly_theme_counts(art, "downtown")
        df = merge_yearly(base, theme)
        if pop is not None and "year" in pop.columns:
            cols = ["year"]
            for cand in ["population", "households", "median_age", "labor_force"]:
                c = first_existing_col(pop, [cand])
                if c and c not in cols:
                    cols.append(c)
            if len(cols) > 1:
                df = df.merge(pop[cols], on="year", how="outer").sort_values("year").fillna(0)
        y1 = first_existing_col(df, ["population", "households"]) or "total_artifacts"
        y2 = "downtown_artifact_mentions" if "downtown_artifact_mentions" in df.columns else "total_artifacts"
        return df, "year", y1, y2

    # fallback
    theme = yearly_theme_counts(art, "flood")
    df = merge_yearly(base, theme)
    return df, "year", "total_artifacts", "flood_artifact_mentions" if "flood_artifact_mentions" in df.columns else "total_artifacts"


def storyline_interpretation(storyline: str) -> str:
    notes = {
        "South Flats: flood damage, buyouts, and text mentions":
            "This view connects the civic flood record with the language of flood, water, South Flats, sandbagging, and buyouts in the archive. Ask whether the textual archive becomes louder during years of material damage, or whether memory appears later.",
        "Public memory: archive work and memory language":
            "This view asks when Mill Creek begins to remember itself institutionally. Compare memory/archive language in the artifacts with public spending or civic support for historical work.",
        "Downtown and redevelopment: business counts and text":
            "This view connects economic change with documentary language about Main Street, the Depot District, the bypass, and redevelopment. Ask whether economic decline and redevelopment are visible in both numbers and narratives.",
        "Schools: enrollment, graduation, and school artifacts":
            "This view treats schools as both quantitative institutions and memory-making institutions. Ask whether school life appears only as enrollment/graduation data, or also as public civic performance.",
        "Land use: South Flats to Riverfront Trail":
            "This view follows the transformation of land from residential floodplain to public space. It is useful for discussing how risk, loss, and recreation can occupy the same landscape over time.",
        "Population and civic change":
            "This view uses population or household data as a background against which artifact themes can be interpreted. Ask whether growth, stability, or decline changes what the town records about itself.",
    }
    return notes.get(storyline, "Use this guided plot to compare civic data with artifact-level textual patterns.")


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

class LoadOverviewTab(QWidget):
    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        layout = QVBoxLayout(self)

        button_row = QHBoxLayout()
        self.load_archive_btn = QPushButton("Open qualitative archive zip/folder")
        self.load_civic_btn = QPushButton("Open quantitative civic data zip/folder")
        self.refresh_btn = QPushButton("Refresh overview")
        button_row.addWidget(self.load_archive_btn)
        button_row.addWidget(self.load_civic_btn)
        button_row.addWidget(self.refresh_btn)
        layout.addLayout(button_row)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        layout.addWidget(self.summary)

        self.canvas = MplCanvas()
        layout.addWidget(self.canvas, stretch=1)

        self.load_archive_btn.clicked.connect(self.load_archive)
        self.load_civic_btn.clicked.connect(self.load_civic)
        self.refresh_btn.clicked.connect(self.refresh)

    def load_archive(self):
        path = QFileDialog.getOpenFileName(self, "Open qualitative archive zip", "", "Zip files (*.zip);;All files (*)")[0]
        if not path:
            folder = QFileDialog.getExistingDirectory(self, "Or choose qualitative archive folder")
            path = folder
        if not path:
            return
        try:
            self.app.manager.load_archive(Path(path))
            self.app.after_data_loaded()
            self.refresh()
        except Exception as e:
            self.app.show_error("Could not load archive", e)

    def load_civic(self):
        path = QFileDialog.getOpenFileName(self, "Open quantitative civic data zip", "", "Zip files (*.zip);;All files (*)")[0]
        if not path:
            folder = QFileDialog.getExistingDirectory(self, "Or choose civic data folder")
            path = folder
        if not path:
            return
        try:
            self.app.manager.load_civic(Path(path))
            self.app.after_data_loaded()
            self.refresh()
        except Exception as e:
            self.app.show_error("Could not load civic data", e)

    def refresh(self):
        art = self.app.artifacts()
        civic = self.app.civic()
        lines = [APP_TITLE, "=" * len(APP_TITLE), ""]
        if art is None:
            lines.append("No qualitative archive loaded.")
        else:
            lines.append(f"Artifacts loaded: {len(art):,}")
            lines.append(f"Years: {art['year'].min()}–{art['year'].max()}")
            lines.append("")
            lines.append("Categories:")
            for k, v in art["category"].value_counts().items():
                lines.append(f"  {k}: {v}")
            lines.append("")
            lines.append("Generation status:")
            for k, v in art["status"].value_counts().items():
                lines.append(f"  {k}: {v}")
            lines.append("")
            lines.append("Mean word count by category:")
            for _, r in art.groupby("category")["body_word_count"].mean().reset_index().iterrows():
                lines.append(f"  {r['category']}: {r['body_word_count']:.1f}")

            counts = art["category"].value_counts()
            self.canvas.plot_bar(counts.index, counts.values, "Artifacts by Category", "Category", "Artifacts")

        lines.append("")
        if civic:
            lines.append(f"Quantitative civic tables loaded: {len(civic)}")
            for name, df in civic.items():
                lines.append(f"  {name}: {len(df)} rows, {len(df.columns)} columns")
        else:
            lines.append("No quantitative civic data loaded.")

        self.summary.setPlainText("\n".join(lines))


class BrowseTab(QWidget):
    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.category = QComboBox()
        self.decade = QComboBox()
        self.search = QLineEdit()
        self.search.setPlaceholderText("filter text")
        self.apply_btn = QPushButton("Apply filters")
        controls.addWidget(QLabel("Category:"))
        controls.addWidget(self.category)
        controls.addWidget(QLabel("Decade:"))
        controls.addWidget(self.decade)
        controls.addWidget(QLabel("Text:"))
        controls.addWidget(self.search)
        controls.addWidget(self.apply_btn)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget()
        self.reader = QTextEdit()
        self.reader.setReadOnly(True)
        splitter.addWidget(self.table)
        splitter.addWidget(self.reader)
        splitter.setSizes([300, 400])
        layout.addWidget(splitter)

        self.apply_btn.clicked.connect(self.refresh_table)
        self.table.itemSelectionChanged.connect(self.show_selected)

    def populate_filters(self):
        art = self.app.artifacts()
        self.category.clear()
        self.decade.clear()
        self.category.addItem("All")
        self.decade.addItem("All")
        if art is not None:
            for c in sorted(art["category"].dropna().astype(str).unique()):
                self.category.addItem(c)
            for d in sorted(art["decade"].dropna().astype(int).unique()):
                self.decade.addItem(str(d))
        self.refresh_table()

    def filtered(self) -> pd.DataFrame:
        art = self.app.artifacts()
        if art is None:
            return pd.DataFrame()
        df = art.copy()
        cat = self.category.currentText()
        dec = self.decade.currentText()
        q = self.search.text().strip()
        if cat and cat != "All":
            df = df[df["category"] == cat]
        if dec and dec != "All":
            df = df[df["decade"] == int(dec)]
        if q:
            mask = df["analysis_text"].str.contains(q, case=False, regex=False, na=False)
            df = df[mask]
        return df

    def refresh_table(self):
        df = self.filtered()
        cols = ["artifact_id", "year", "category", "source_type", "display_title", "body_preview"]
        cols = [c for c in cols if c in df.columns]
        set_table(df[cols], self.table, max_rows=800)

    def show_selected(self):
        art = self.app.artifacts()
        if art is None:
            return
        items = self.table.selectedItems()
        if not items:
            return
        row = items[0].row()
        aid_item = self.table.item(row, 0)
        if aid_item is None:
            return
        aid = aid_item.text()
        sub = art[art["artifact_id"] == aid]
        if sub.empty:
            return
        r = sub.iloc[0]
        text = (
            f"<h2>{html.escape(str(r['display_title']))}</h2>"
            f"<p><b>Artifact:</b> {html.escape(str(r['artifact_id']))}<br>"
            f"<b>Year:</b> {r['year']}<br>"
            f"<b>Category:</b> {html.escape(str(r['category']))}<br>"
            f"<b>Source type:</b> {html.escape(str(r['source_type']))}<br>"
            f"<b>Location:</b> {html.escape(str(r.get('location','')))}<br>"
            f"<b>Topic:</b> {html.escape(str(r.get('topic','')))}</p>"
            f"<hr><p>{html.escape(str(r['body'])).replace(chr(10), '<br>')}</p>"
        )
        self.reader.setHtml(text)


class SearchTab(QWidget):
    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        self.semantic_index = None

        layout = QVBoxLayout(self)

        row1 = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText("Try: the town remembering itself")
        self.keyword_btn = QPushButton("Keyword search")
        self.build_semantic_btn = QPushButton("Build semantic index")
        self.semantic_btn = QPushButton("Semantic search")
        self.embedding_checkbox = QCheckBox("Try sentence-transformers")
        self.embedding_checkbox.setChecked(True)
        row1.addWidget(QLabel("Query:"))
        row1.addWidget(self.query, stretch=1)
        row1.addWidget(self.keyword_btn)
        row1.addWidget(self.build_semantic_btn)
        row1.addWidget(self.semantic_btn)
        row1.addWidget(self.embedding_checkbox)
        layout.addLayout(row1)

        self.status = QLabel("Semantic index not built.")
        layout.addWidget(self.status)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget()
        self.reader = QTextEdit()
        self.reader.setReadOnly(True)
        splitter.addWidget(self.table)
        splitter.addWidget(self.reader)
        splitter.setSizes([300, 350])
        layout.addWidget(splitter)

        self.keyword_btn.clicked.connect(self.do_keyword)
        self.build_semantic_btn.clicked.connect(self.build_semantic)
        self.semantic_btn.clicked.connect(self.do_semantic)
        self.table.itemSelectionChanged.connect(self.show_selected)

    def do_keyword(self):
        art = self.app.require_artifacts()
        df = keyword_search(art, self.query.text(), top_k=80)
        set_table(df, self.table, max_rows=80)
        self.status.setText(f"Keyword results: {len(df)}")

    def build_semantic(self):
        try:
            art = self.app.require_artifacts()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.semantic_index = build_semantic_index(art, use_embeddings=self.embedding_checkbox.isChecked())
            self.status.setText(f"Semantic index built: {self.semantic_index['mode']}")
        except Exception as e:
            self.app.show_error("Semantic index failed", e)
        finally:
            QApplication.restoreOverrideCursor()

    def do_semantic(self):
        try:
            art = self.app.require_artifacts()
            if self.semantic_index is None:
                self.build_semantic()
            df = semantic_search(self.semantic_index, art, self.query.text(), top_k=20)
            set_table(df, self.table, max_rows=40)
            self.status.setText(f"Semantic results: {len(df)} | backend: {self.semantic_index['mode']}")
        except Exception as e:
            self.app.show_error("Semantic search failed", e)

    def show_selected(self):
        art = self.app.artifacts()
        if art is None:
            return
        items = self.table.selectedItems()
        if not items:
            return
        # Locate artifact_id column.
        aid = None
        for c in range(self.table.columnCount()):
            if self.table.horizontalHeaderItem(c).text() == "artifact_id":
                item = self.table.item(items[0].row(), c)
                aid = item.text() if item else None
                break
        if not aid:
            return
        sub = art[art["artifact_id"] == aid]
        if sub.empty:
            return
        r = sub.iloc[0]
        self.reader.setHtml(
            f"<h2>{html.escape(str(r['display_title']))}</h2>"
            f"<p><b>{r['artifact_id']}</b> | {r['year']} | {html.escape(str(r['category']))}</p>"
            f"<p>{html.escape(str(r['body'])).replace(chr(10), '<br>')}</p>"
        )


class PatternsTab(QWidget):
    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.k_clusters = QSpinBox()
        self.k_clusters.setRange(3, 20)
        self.k_clusters.setValue(8)
        self.n_topics = QSpinBox()
        self.n_topics.setRange(3, 20)
        self.n_topics.setValue(10)
        self.cluster_btn = QPushButton("Run clustering")
        self.topic_btn = QPushButton("Run topic modeling")
        controls.addWidget(QLabel("Clusters:"))
        controls.addWidget(self.k_clusters)
        controls.addWidget(self.cluster_btn)
        controls.addWidget(QLabel("Topics:"))
        controls.addWidget(self.n_topics)
        controls.addWidget(self.topic_btn)
        controls.addStretch()
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_l = QVBoxLayout(left)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.table = QTableWidget()
        left_l.addWidget(self.output)
        left_l.addWidget(self.table)
        self.canvas = MplCanvas()
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.setSizes([600, 500])
        layout.addWidget(splitter)

        self.cluster_btn.clicked.connect(self.do_cluster)
        self.topic_btn.clicked.connect(self.do_topics)

    def do_cluster(self):
        try:
            art = self.app.require_artifacts()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            assignments, top_terms, coords = run_clustering(art, self.k_clusters.value())
            self.app.analysis_cache["clusters"] = assignments
            self.output.setPlainText("Cluster top terms\n" + "=" * 30 + "\n\n" + dataframe_to_plain(top_terms, 30))
            set_table(assignments[["artifact_id", "year", "category", "cluster", "display_title", "body_preview"]], self.table)
            self.canvas.plot_scatter(assignments["x"], assignments["y"], "Document Map by Cluster", "SVD 1", "SVD 2", labels=assignments["cluster"])
        except Exception as e:
            self.app.show_error("Clustering failed", e)
        finally:
            QApplication.restoreOverrideCursor()

    def do_topics(self):
        try:
            art = self.app.require_artifacts()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            assignments, top_terms, reps = run_topics(art, self.n_topics.value())
            self.app.analysis_cache["topics"] = assignments
            self.output.setPlainText(
                "Topic top terms\n" + "=" * 30 + "\n\n" + dataframe_to_plain(top_terms, 30) +
                "\n\nRepresentative artifacts\n" + "=" * 30 + "\n\n" +
                dataframe_to_plain(reps[["topic", "rank", "weight", "artifact_id", "year", "category", "display_title"]], 80)
            )
            set_table(reps, self.table, max_rows=120)
            ct = pd.crosstab(assignments["topic"], assignments["category"])
            self.canvas.plot_heatmap(ct.values, ct.columns, ct.index, "Topic by Category")
        except Exception as e:
            self.app.show_error("Topic modeling failed", e)
        finally:
            QApplication.restoreOverrideCursor()


class ClassificationTab(QWidget):
    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.task_select = QComboBox()
        self.task_select.addItems([
            "Category",
            "Source group",
            "Historical period",
            "Flood-related",
            "Memory/archive-related",
            "Downtown/redevelopment-related",
        ])
        self.body_only = QCheckBox("Body only")
        self.run_btn = QPushButton("Run classification")
        controls.addWidget(QLabel("Classification task:"))
        controls.addWidget(self.task_select)
        controls.addWidget(self.body_only)
        controls.addWidget(self.run_btn)
        controls.addStretch()
        layout.addLayout(controls)

        outer = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_l = QVBoxLayout(left)
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.table = QTableWidget()
        left_l.addWidget(QLabel("Report and misclassified examples"))
        left_l.addWidget(self.report)
        left_l.addWidget(self.table)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.addWidget(QLabel("Confusion matrix graph"))
        self.canvas = MplCanvas()
        self.matrix_table = QTableWidget()
        right_l.addWidget(self.canvas, stretch=2)
        right_l.addWidget(self.matrix_table, stretch=1)

        outer.addWidget(left)
        outer.addWidget(right)
        outer.setSizes([700, 600])
        layout.addWidget(outer)

        self.run_btn.clicked.connect(self.run)

    def run(self):
        try:
            art = self.app.require_artifacts()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            task = self.task_select.currentText()
            report, cm, test_df = run_classification_task(art, task, body_only=self.body_only.isChecked())

            accuracy = float((test_df["correct"] == True).mean()) if len(test_df) else 0.0
            self.report.setPlainText(
                f"Classification task: {task}\n"
                f"Classification accuracy on held-out test set: {accuracy:.3f}\n"
                f"Text mode: {'body only' if self.body_only.isChecked() else 'metadata + body'}\n\n"
                "Classification report\n" + "=" * 40 + "\n" + report +
                "\n\nMisclassified examples appear first in the table below. "
                "Each task asks a different historical question, so the confusion matrix should be interpreted differently."
            )

            # Show incorrect predictions first.
            display = test_df.sort_values(["correct", "year"], ascending=[True, True])
            set_table(display, self.table, max_rows=250)
            set_table(cm.reset_index().rename(columns={"index": "true_label"}), self.matrix_table, max_rows=50)

            self.canvas.plot_heatmap(
                cm.values,
                cm.columns,
                cm.index,
                "Confusion Matrix: True Label vs Predicted Label",
            )
        except Exception as e:
            self.app.show_error("Classification failed", e)
        finally:
            QApplication.restoreOverrideCursor()


class NetworkTab(QWidget):
    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        self.nodes = pd.DataFrame()
        self.edges = pd.DataFrame()

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.run_btn = QPushButton("Build network")
        self.ego_query = QLineEdit()
        self.ego_query.setPlaceholderText("South Flats")
        self.ego_btn = QPushButton("Show direct ego graph")
        controls.addWidget(self.run_btn)
        controls.addWidget(QLabel("Ego center:"))
        controls.addWidget(self.ego_query)
        controls.addWidget(self.ego_btn)
        controls.addStretch()
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_l = QVBoxLayout(left)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.table = QTableWidget()
        left_l.addWidget(self.output)
        left_l.addWidget(self.table)
        self.canvas = MplCanvas()
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.setSizes([650, 450])
        layout.addWidget(splitter)

        self.run_btn.clicked.connect(self.run_network)
        self.ego_btn.clicked.connect(self.show_ego)

    def run_network(self):
        try:
            art = self.app.require_artifacts()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            nodes, edges, top_nodes = run_network(art, self.app.civic())
            self.nodes = nodes
            self.edges = edges
            self.output.setPlainText(
                f"NetworkX available: {HAS_NETWORKX}\n"
                f"Entity nodes: {len(nodes)}\n"
                f"Entity co-occurrence edges: {len(edges)}\n\n"
                "Top bridge / connector nodes\n" + "=" * 40 + "\n" +
                dataframe_to_plain(top_nodes.head(40), 40)
            )
            set_table(top_nodes, self.table, max_rows=100)
            if not top_nodes.empty:
                top = top_nodes.head(15)
                self.canvas.plot_bar(top["label"], top["weighted_degree"], "Top Entity Nodes by Weighted Degree", "Entity", "Weighted degree")
        except Exception as e:
            self.app.show_error("Network analysis failed", e)
        finally:
            QApplication.restoreOverrideCursor()

    def show_ego(self):
        if self.edges is None or self.edges.empty:
            self.run_network()
        center = self.ego_query.text().strip()
        df = direct_ego_edges(self.edges, center)
        self.output.setPlainText(
            f"Direct ego graph for: {center}\n"
            f"Edges touching the center phrase: {len(df)}\n\n"
            "The graph view shows the center node and its strongest entity connections. "
            "The table gives the underlying weighted edges.\n\n" +
            dataframe_to_plain(df.head(60), 60)
        )
        set_table(df, self.table, max_rows=100)
        self.canvas.plot_network_graph(df, center_phrase=center, max_edges=35)



class CivicStorylinesTab(QWidget):
    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Guided quantitative/qualitative views. These plots connect civic records "
            "with artifact-language patterns so students can move between numbers, texts, and interpretation."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        controls = QGridLayout()
        self.storyline = QComboBox()
        self.storyline.addItems([
            "South Flats: flood damage, buyouts, and text mentions",
            "Public memory: archive work and memory language",
            "Downtown and redevelopment: business counts and text",
            "Schools: enrollment, graduation, and school artifacts",
            "Land use: South Flats to Riverfront Trail",
            "Population and civic change",
        ])
        self.refresh_btn = QPushButton("Build storyline")
        self.y1_col = QComboBox()
        self.y2_col = QComboBox()
        self.replot_btn = QPushButton("Replot selected variables")
        self.connect_points = QCheckBox("Connect data points")
        self.connect_points.setChecked(True)

        controls.addWidget(QLabel("Storyline:"), 0, 0)
        controls.addWidget(self.storyline, 0, 1, 1, 4)
        controls.addWidget(self.refresh_btn, 0, 5)

        controls.addWidget(QLabel("Left axis:"), 1, 0)
        controls.addWidget(self.y1_col, 1, 1)
        controls.addWidget(QLabel("Right axis / text theme:"), 1, 2)
        controls.addWidget(self.y2_col, 1, 3)
        controls.addWidget(self.connect_points, 1, 4)
        controls.addWidget(self.replot_btn, 1, 5)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_l = QVBoxLayout(left)
        self.notes = QTextEdit()
        self.notes.setReadOnly(True)
        self.table = QTableWidget()
        left_l.addWidget(QLabel("Interpretive prompt"))
        left_l.addWidget(self.notes, stretch=1)
        left_l.addWidget(QLabel("Merged storyline data"))
        left_l.addWidget(self.table, stretch=2)

        self.canvas = MplCanvas()
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.setSizes([650, 650])
        layout.addWidget(splitter)

        self.current_df = pd.DataFrame()
        self.current_x = "year"

        self.refresh_btn.clicked.connect(self.build_storyline)
        self.replot_btn.clicked.connect(self.replot)
        self.storyline.currentTextChanged.connect(self.build_storyline)

    def build_storyline(self):
        try:
            art = self.app.require_artifacts()
            civic = self.app.civic()
            if not civic:
                raise ValueError("Please load the quantitative civic data first.")

            story = self.storyline.currentText()
            df, x_col, y1, y2 = build_storyline_dataframe(art, civic, story)
            self.current_df = df
            self.current_x = x_col

            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != x_col]
            self.y1_col.clear()
            self.y2_col.clear()
            for c in numeric_cols:
                self.y1_col.addItem(str(c))
                self.y2_col.addItem(str(c))
            if y1 in numeric_cols:
                self.y1_col.setCurrentText(str(y1))
            if y2 in numeric_cols:
                self.y2_col.setCurrentText(str(y2))

            self.notes.setPlainText(
                f"{story}\n\n"
                f"{storyline_interpretation(story)}\n\n"
                "Reading question: Do the quantitative records and the artifact language move together, "
                "or do they tell different parts of the story?"
            )
            set_table(df, self.table, max_rows=500)
            self.replot()
        except Exception as e:
            self.app.show_error("Could not build civic storyline", e)

    def replot(self):
        if self.current_df is None or self.current_df.empty:
            return
        y1 = self.y1_col.currentText()
        y2 = self.y2_col.currentText()
        story = self.storyline.currentText()
        self.canvas.plot_dual_storyline(
            self.current_df,
            self.current_x,
            y1,
            y2,
            story,
            connect=self.connect_points.isChecked(),
        )




class CivicTab(QWidget):
    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        layout = QVBoxLayout(self)

        instructions = QLabel(
            "Choose a civic data table, then choose the horizontal and vertical variables to plot. "
            "For a time series, use year on the horizontal axis. For a correlation-style plot, choose any two numeric columns."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        controls = QGridLayout()
        self.table_select = QComboBox()
        self.x_col = QComboBox()
        self.y_col = QComboBox()
        self.refresh_btn = QPushButton("Refresh civic tables")
        self.plot_btn = QPushButton("Plot")
        self.connect_points = QCheckBox("Connect data points")
        self.connect_points.setChecked(False)

        controls.addWidget(QLabel("Civic table:"), 0, 0)
        controls.addWidget(self.table_select, 0, 1, 1, 3)
        controls.addWidget(self.refresh_btn, 0, 4)

        controls.addWidget(QLabel("Horizontal axis (x):"), 1, 0)
        controls.addWidget(self.x_col, 1, 1)
        controls.addWidget(QLabel("Vertical axis (y):"), 1, 2)
        controls.addWidget(self.y_col, 1, 3)
        controls.addWidget(self.plot_btn, 1, 4)

        controls.addWidget(self.connect_points, 2, 1)
        layout.addLayout(controls)

        self.hint = QLabel("Tip: leave 'Connect data points' off for correlation plots; turn it on for year-based trend lines.")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget()
        self.canvas = MplCanvas()
        splitter.addWidget(self.table)
        splitter.addWidget(self.canvas)
        splitter.setSizes([650, 450])
        layout.addWidget(splitter)

        self.refresh_btn.clicked.connect(self.populate)
        self.table_select.currentTextChanged.connect(self.show_table)
        self.plot_btn.clicked.connect(self.plot)

    def populate(self):
        civic = self.app.civic()
        self.table_select.clear()
        for name in sorted(civic.keys()):
            self.table_select.addItem(name)
        self.show_table()

    def show_table(self):
        civic = self.app.civic()
        name = self.table_select.currentText()
        if not civic or name not in civic:
            return
        df = civic[name]
        set_table(df, self.table, max_rows=1000)

        self.x_col.clear()
        self.y_col.clear()

        numeric_cols = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        all_cols = [str(c) for c in df.columns]

        # x-axis can be any column, but plotting works best with numeric columns.
        for col in all_cols:
            self.x_col.addItem(col)
        for col in numeric_cols:
            self.y_col.addItem(col)

        if "year" in df.columns:
            self.x_col.setCurrentText("year")
            # For year-based data, connecting points is often useful, but leave default OFF as requested.
        elif numeric_cols:
            self.x_col.setCurrentText(numeric_cols[0])

        # Pick a reasonable first y-column that is not year.
        for col in numeric_cols:
            if col != "year":
                self.y_col.setCurrentText(col)
                break

    def plot(self):
        civic = self.app.civic()
        name = self.table_select.currentText()
        if not civic or name not in civic:
            return
        df = civic[name]
        x = self.x_col.currentText()
        y = self.y_col.currentText()
        if x not in df.columns or y not in df.columns:
            return

        try:
            sub = df[[x, y]].dropna().copy()

            # Matplotlib scatter requires numeric x. If x is not numeric, use category codes but label clearly.
            if not pd.api.types.is_numeric_dtype(sub[x]):
                sub["_x_codes"] = pd.factorize(sub[x])[0]
                x_values = sub["_x_codes"]
                xlabel = f"{x} (category code)"
            else:
                x_values = sub[x]
                xlabel = x

            # Sort by x only when connecting points.
            if self.connect_points.isChecked():
                sub = sub.assign(_x_plot=x_values).sort_values("_x_plot")
                x_values = sub["_x_plot"] if "_x_plot" in sub else sub[x]

            self.canvas.plot_points(
                x_values,
                sub[y],
                f"{name}: {y} vs {x}",
                xlabel,
                y,
                connect=self.connect_points.isChecked(),
            )
        except Exception as e:
            self.app.show_error("Could not plot civic data", e)


class AuditTab(QWidget):
    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.run_btn = QPushButton("Run archive audit")
        controls.addWidget(self.run_btn)
        controls.addStretch()
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_l = QVBoxLayout(left)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.table = QTableWidget()
        left_l.addWidget(self.output)
        left_l.addWidget(self.table)
        self.canvas = MplCanvas()
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.setSizes([650, 450])
        layout.addWidget(splitter)

        self.run_btn.clicked.connect(self.run_audit)

    def run_audit(self):
        try:
            art = self.app.require_artifacts()
            flags, repeated, category_summary, summary = run_audit(art)
            self.app.analysis_cache["audit_flags"] = flags
            hard = int((flags["severity"] == "hard").sum()) if not flags.empty else 0
            soft = int((flags["severity"] == "soft").sum()) if not flags.empty else 0
            text = (
                "Archive audit\n" + "=" * 40 + "\n\n" +
                dataframe_to_plain(summary, 20) +
                f"\n\nHard flags: {hard}\nSoft flags: {soft}\n\n" +
                "Top repeated titles\n" + "=" * 40 + "\n" +
                dataframe_to_plain(repeated.head(20), 20) +
                "\n\nCategory summary\n" + "=" * 40 + "\n" +
                dataframe_to_plain(category_summary, 20)
            )
            self.output.setPlainText(text)
            set_table(flags, self.table, max_rows=500)
            if not category_summary.empty:
                self.canvas.plot_bar(category_summary["category"], category_summary["title_repetition_ratio"], "Title Repetition Ratio by Category", "Category", "Repetition ratio")
        except Exception as e:
            self.app.show_error("Audit failed", e)


# ---------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 900)

        self.manager = DataManager()
        self.analysis_cache: dict[str, Any] = {}

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.load_tab = LoadOverviewTab(self)
        self.browse_tab = BrowseTab(self)
        self.search_tab = SearchTab(self)
        self.patterns_tab = PatternsTab(self)
        self.classification_tab = ClassificationTab(self)
        self.network_tab = NetworkTab(self)
        self.storylines_tab = CivicStorylinesTab(self)
        self.civic_tab = CivicTab(self)
        self.audit_tab = AuditTab(self)

        self.tabs.addTab(self.load_tab, "Load & Overview")
        self.tabs.addTab(self.browse_tab, "Browse Archive")
        self.tabs.addTab(self.search_tab, "Search by Meaning")
        self.tabs.addTab(self.patterns_tab, "Patterns")
        self.tabs.addTab(self.classification_tab, "Classify")
        self.tabs.addTab(self.network_tab, "Networks")
        self.tabs.addTab(self.storylines_tab, "Civic Storylines")
        self.tabs.addTab(self.civic_tab, "Civic Data")
        self.tabs.addTab(self.audit_tab, "Audit")

        self.statusBar().showMessage("Load a qualitative archive zip/folder to begin.")

    def artifacts(self) -> pd.DataFrame | None:
        return self.manager.data.artifacts

    def civic(self) -> dict[str, pd.DataFrame]:
        return self.manager.data.civic or {}

    def require_artifacts(self) -> pd.DataFrame:
        art = self.artifacts()
        if art is None or art.empty:
            raise ValueError("Please load a qualitative archive first.")
        return art

    def after_data_loaded(self):
        try:
            self.browse_tab.populate_filters()
            self.civic_tab.populate()
            art = self.artifacts()
            civic = self.civic()
            msg = []
            if art is not None:
                msg.append(f"Archive: {len(art)} artifacts")
            if civic:
                msg.append(f"Civic tables: {len(civic)}")
            self.statusBar().showMessage(" | ".join(msg) if msg else "Data loaded.")
        except Exception:
            pass

    def show_error(self, title: str, error: Exception):
        tb = traceback.format_exc()
        QMessageBox.critical(self, title, f"{error}\n\nDetails:\n{tb}")

    def closeEvent(self, event):
        self.manager.cleanup()
        event.accept()


def main():
    import sys
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
