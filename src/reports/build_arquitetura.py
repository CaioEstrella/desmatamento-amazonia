"""Gera reports/arquitetura.png — diagrama de arquitetura do projeto via Graphviz."""

import os
import graphviz

# Aponta para o binário instalado pelo winget (não está no PATH da sessão ainda)
os.environ["PATH"] += r";C:\Program Files\Graphviz\bin"

OUTPUT = "reports/arquitetura"

dot = graphviz.Digraph(
    name="Arquitetura",
    comment="Preditor de Risco de Desmatamento — Amazônia Legal",
    format="png",
    graph_attr={
        "rankdir": "LR",
        "splines": "ortho",
        "bgcolor": "#f8fafc",
        "fontname": "Helvetica",
        "fontsize": "13",
        "pad": "0.6",
        "nodesep": "0.55",
        "ranksep": "1.1",
    },
    node_attr={
        "fontname": "Helvetica",
        "fontsize": "11",
        "style": "filled",
        "penwidth": "1.5",
    },
    edge_attr={
        "fontname": "Helvetica",
        "fontsize": "9",
        "color": "#64748b",
        "arrowsize": "0.8",
    },
)

# ── Estilos por camada ─────────────────────────────────────────────────────────
SOURCE  = {"shape": "cylinder",   "fillcolor": "#dbeafe", "color": "#3b82f6"}
INGEST  = {"shape": "box",        "fillcolor": "#ede9fe", "color": "#7c3aed"}
STORE   = {"shape": "folder",     "fillcolor": "#fef9c3", "color": "#ca8a04"}
FEAT    = {"shape": "box",        "fillcolor": "#dcfce7", "color": "#16a34a"}
MODEL   = {"shape": "box",        "fillcolor": "#fee2e2", "color": "#dc2626"}
OUTPUT_node = {"shape": "box3d", "fillcolor": "#fce7f3", "color": "#db2777"}

# ── Fontes externas ────────────────────────────────────────────────────────────
with dot.subgraph(name="cluster_sources") as s:
    s.attr(label="Fontes Públicas", style="rounded,filled", fillcolor="#eff6ff",
           color="#93c5fd", fontsize="12", fontcolor="#1e3a8a")
    s.node("ibge",   "IBGE\n(municípios, PIB,\npopulação, shapes)", **SOURCE)
    s.node("inpe",   "INPE / PRODES\n(desmatamento km²)",           **SOURCE)
    s.node("icmbio", "ICMBio / FUNAI\n(UC, TI %)",                  **SOURCE)
    s.node("ibama",  "IBAMA\n(autos de infração)",                   **SOURCE)

# ── MCP server ────────────────────────────────────────────────────────────────
dot.node("mcp", "mcp-brasil\n(MCP Server / uvx)", shape="component",
         fillcolor="#e0f2fe", color="#0284c7", style="filled", fontsize="11")

# ── Ingestão ──────────────────────────────────────────────────────────────────
with dot.subgraph(name="cluster_ingest") as s:
    s.attr(label="Ingestão  (src/ingestion/)", style="rounded,filled",
           fillcolor="#f5f3ff", color="#a78bfa", fontsize="12", fontcolor="#4c1d95")
    s.node("ing_ibge",   "ibge_geo.py",    **INGEST)
    s.node("ing_prodes", "prodes.py",      **INGEST)
    s.node("ing_icmbio", "icmbio.py",      **INGEST)
    s.node("ing_ibama",  "ibama.py",       **INGEST)

# ── Armazenamento ─────────────────────────────────────────────────────────────
with dot.subgraph(name="cluster_store") as s:
    s.attr(label="Armazenamento", style="rounded,filled",
           fillcolor="#fefce8", color="#fde047", fontsize="12", fontcolor="#713f12")
    s.node("raw",       "data/raw/\n(.parquet brutos)",      **STORE)
    s.node("processed", "data/processed/\ndataset.parquet",  **STORE)
    s.node("outputs",   "data/outputs/\n(modelos, estudos)", **STORE)

# ── Feature engineering ───────────────────────────────────────────────────────
with dot.subgraph(name="cluster_feat") as s:
    s.attr(label="Features  (src/features/)", style="rounded,filled",
           fillcolor="#f0fdf4", color="#86efac", fontsize="12", fontcolor="#14532d")
    s.node("etl",      "consolidate.py\n(ETL + merge)",         **FEAT)
    s.node("eng",      "engineer.py\n(lags, rolling, Moran I)", **FEAT)

# ── Clustering ────────────────────────────────────────────────────────────────
dot.node("hdbscan", "HDBSCAN\n(src/clustering/\nhdbscan_model.py)",
         shape="box", fillcolor="#fef3c7", color="#d97706", style="filled", fontsize="11")

# ── Modelagem ─────────────────────────────────────────────────────────────────
with dot.subgraph(name="cluster_model") as s:
    s.attr(label="Modelagem  (src/model/)", style="rounded,filled",
           fillcolor="#fef2f2", color="#fca5a5", fontsize="12", fontcolor="#7f1d1d")
    s.node("train",    "train.py\n(LightGBM + Optuna\n50 trials, spatial CV)", **MODEL)
    s.node("evaluate", "evaluate.py\n(RMSE, R², SHAP)",                        **MODEL)
    s.node("predict",  "predict.py\n(score 0–100)",                             **MODEL)

# ── Outputs ───────────────────────────────────────────────────────────────────
with dot.subgraph(name="cluster_out") as s:
    s.attr(label="Outputs  (reports/)", style="rounded,filled",
           fillcolor="#fdf2f8", color="#f0abfc", fontsize="12", fontcolor="#701a75")
    s.node("dash",   "Dashboard HTML\n(GitHub Pages)",        **OUTPUT_node)
    s.node("shap",   "Relatório SHAP\n(beeswarm, waterfall)", **OUTPUT_node)
    s.node("slides", "Slide Deck\n(PowerPoint executivo)",    **OUTPUT_node)

# ── Arestas ───────────────────────────────────────────────────────────────────
# Fontes → MCP
for src in ["ibge", "inpe", "icmbio", "ibama"]:
    dot.edge(src, "mcp")

# MCP → ingestão
dot.edge("mcp", "ing_ibge")
dot.edge("mcp", "ing_prodes")
dot.edge("mcp", "ing_icmbio")
dot.edge("mcp", "ing_ibama")

# Ingestão → raw
for ing in ["ing_ibge", "ing_prodes", "ing_icmbio", "ing_ibama"]:
    dot.edge(ing, "raw")

# raw → ETL → eng → processed
dot.edge("raw",  "etl")
dot.edge("etl",  "eng")
dot.edge("eng",  "processed")

# processed → HDBSCAN → processed (cluster_id)
dot.edge("processed", "hdbscan", xlabel="features")
dot.edge("hdbscan",   "processed", xlabel="cluster_id", style="dashed")

# processed → treino → avaliação → predição → outputs
dot.edge("processed", "train",    xlabel="dataset")
dot.edge("train",     "outputs",  xlabel="best_params\noptuna_study")
dot.edge("train",     "evaluate")
dot.edge("evaluate",  "shap")
dot.edge("evaluate",  "predict")
dot.edge("predict",   "outputs",  xlabel="scores.parquet")
dot.edge("predict",   "dash")
dot.edge("predict",   "slides")

# Renderiza
dot.render(OUTPUT, cleanup=True)
print(f"Diagrama salvo em {OUTPUT}.png")
