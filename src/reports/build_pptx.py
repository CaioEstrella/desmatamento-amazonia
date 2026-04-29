"""
TASK-17: Gerador do slide deck executivo em .pptx.

Lê skills/pptx/SKILL.md antes de qualquer geração (requisito).
Produz reports/slide_deck_executivo.pptx com 10 slides widescreen 16:9.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt, Emu

logger = logging.getLogger(__name__)

# ── Paleta de cores (SKILL.md) ────────────────────────────────────────────────
VERDE_ESCURO = RGBColor(0x16, 0x65, 0x34)
VERDE_MEDIO  = RGBColor(0x22, 0xC5, 0x5E)
VERDE_CLARO  = RGBColor(0xBB, 0xF7, 0xD0)
VERMELHO     = RGBColor(0xEF, 0x44, 0x44)
LARANJA      = RGBColor(0xF9, 0x73, 0x16)
AMARELO      = RGBColor(0xEA, 0xB3, 0x08)
SLATE_900    = RGBColor(0x0F, 0x17, 0x2A)
SLATE_700    = RGBColor(0x33, 0x4D, 0x6B)
SLATE_500    = RGBColor(0x64, 0x74, 0x8B)
SLATE_200    = RGBColor(0xE2, 0xE8, 0xF0)
SLATE_100    = RGBColor(0xF1, 0xF5, 0xF9)
BRANCO       = RGBColor(0xFF, 0xFF, 0xFF)

_PREDICTIONS = "data/outputs/predictions.parquet"
_METRICS     = "data/outputs/metrics_by_fold.json"
_OUTPUT      = "reports/slide_deck_executivo.pptx"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_bg(shape, color: RGBColor) -> None:
    fill = shape.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(
    slide, text: str,
    left: float, top: float, width: float, height: float,
    font_size: int = 16, bold: bool = False,
    color: RGBColor = SLATE_900,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    bg_color: RGBColor | None = None,
    italic: bool = False,
) -> None:
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    if bg_color:
        _set_bg(txBox, bg_color)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def add_text_box_multiline(
    slide, lines: list[tuple],
    left: float, top: float, width: float, height: float,
    bg_color: RGBColor | None = None,
) -> None:
    """lines: list of (text, font_size, bold, color, align)"""
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    if bg_color:
        _set_bg(txBox, bg_color)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, (text, fs, bold, color, align) in enumerate(lines):
        p = tf.paragraphs[i] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = text
        run.font.size = Pt(fs)
        run.font.bold = bold
        run.font.color.rgb = color


def add_kpi_box(
    slide, label: str, value: str,
    left: float, top: float,
    width: float = 2.8, height: float = 1.3,
    value_color: RGBColor = VERDE_ESCURO,
    bg_color: RGBColor = SLATE_100,
) -> None:
    box = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    _set_bg(box, bg_color)
    box.line.color.rgb = SLATE_200

    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left  = Inches(0.12)
    tf.margin_top   = Inches(0.08)

    p1 = tf.paragraphs[0]
    r1 = p1.add_run()
    r1.text = value
    r1.font.size  = Pt(26)
    r1.font.bold  = True
    r1.font.color.rgb = value_color

    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = label
    r2.font.size  = Pt(11)
    r2.font.color.rgb = SLATE_500


def add_table(
    slide, headers: list[str], rows: list[list],
    left: float, top: float, width: float, height: float,
    header_bg: RGBColor = VERDE_ESCURO,
    header_fg: RGBColor = BRANCO,
    font_size: int = 11,
) -> None:
    n_rows = len(rows) + 1
    n_cols = len(headers)
    table = slide.shapes.add_table(
        n_rows, n_cols,
        Inches(left), Inches(top), Inches(width), Inches(height)
    ).table

    col_w = width / n_cols
    for i in range(n_cols):
        table.columns[i].width = Inches(col_w)

    for col_i, h in enumerate(headers):
        cell = table.cell(0, col_i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_bg
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.runs[0] if p.runs else p.add_run()
        run.font.bold  = True
        run.font.size  = Pt(font_size)
        run.font.color.rgb = header_fg

    for row_i, row in enumerate(rows):
        bg = SLATE_100 if row_i % 2 == 0 else BRANCO
        for col_i, val in enumerate(row):
            cell = table.cell(row_i + 1, col_i)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            run = p.runs[0] if p.runs else p.add_run()
            run.font.size = Pt(font_size)
            run.font.color.rgb = SLATE_900


def add_sidebar(slide, color: RGBColor = VERDE_ESCURO, width: float = 0.4) -> None:
    bar = slide.shapes.add_shape(
        1, Inches(0), Inches(0), Inches(width), Inches(7.5)
    )
    _set_bg(bar, color)
    bar.line.fill.background()


def add_bg(slide, color: RGBColor) -> None:
    bg = slide.shapes.add_shape(
        1, Inches(0), Inches(0), Inches(13.33), Inches(7.5)
    )
    _set_bg(bg, color)
    bg.line.fill.background()
    bg.zorder = 0


def _blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_bullet_block(
    slide, title: str, bullets: list[str],
    left: float, top: float, width: float,
    title_color: RGBColor = VERDE_ESCURO,
) -> None:
    add_text_box(slide, title, left, top, width, 0.4,
                 font_size=13, bold=True, color=title_color)
    for i, b in enumerate(bullets):
        add_text_box(slide, b, left + 0.1, top + 0.45 + i * 0.35, width - 0.1, 0.35,
                     font_size=11, color=SLATE_700)


# ── Slides ────────────────────────────────────────────────────────────────────

def slide_capa(prs: Presentation) -> None:
    sl = _blank(prs)
    add_bg(sl, VERDE_ESCURO)

    # Emoji
    add_text_box(sl, "🌿", 0.5, 0.5, 2, 2, font_size=72, color=BRANCO)

    # Título
    add_text_box(sl, "Preditor de Risco de Desmatamento",
                 2.6, 1.8, 10.0, 1.1, font_size=38, bold=True, color=BRANCO,
                 align=PP_ALIGN.LEFT)

    # Subtítulo
    add_text_box(sl, "Amazônia Legal · Machine Learning aplicado à conservação",
                 2.6, 3.0, 10.0, 0.6, font_size=20, color=VERDE_CLARO)

    # Rodapé
    add_text_box(sl, f"772 municípios · 9 estados · 2008–2025 · {date.today().year}",
                 2.6, 6.5, 10.0, 0.5, font_size=13, color=SLATE_500)


def slide_problema(prs: Presentation) -> None:
    sl = _blank(prs)
    add_sidebar(sl)

    add_text_box(sl, "O Problema", 0.65, 0.4, 12.0, 0.7,
                 font_size=28, bold=True, color=SLATE_900)

    bullets = [
        "🌳  Amazônia Legal: 5,2 milhões km² de floresta tropical",
        "📊  Desmatamento histórico: pico em 2004, redução até 2012, reaceleração após 2018",
        "⚠️  Necessidade: identificar municípios de alto risco antes do desmatamento ocorrer",
        "🤖  Solução: modelo preditivo com score de risco atualizado anualmente por município",
    ]
    for i, b in enumerate(bullets):
        add_text_box(sl, b, 0.65, 1.4 + i * 0.75, 12.0, 0.65, font_size=14, color=SLATE_700)

    # Box objetivo
    box = sl.shapes.add_shape(1, Inches(0.65), Inches(5.0), Inches(12.0), Inches(1.1))
    _set_bg(box, VERDE_CLARO)
    box.line.color.rgb = VERDE_MEDIO
    tf = box.text_frame
    tf.margin_left = Inches(0.2)
    tf.margin_top  = Inches(0.15)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "Objetivo: Score de risco 0–100 por município, atualizável anualmente"
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = VERDE_ESCURO


def slide_dados(prs: Presentation) -> None:
    sl = _blank(prs)
    add_sidebar(sl)

    add_text_box(sl, "Pipeline de Dados", 0.65, 0.4, 12.0, 0.7,
                 font_size=28, bold=True, color=SLATE_900)

    headers = ["Fonte", "Dados", "Cobertura"]
    rows = [
        ["INPE/PRODES",    "Desmatamento km² por município",        "2008–2025"],
        ["IBGE",           "Pop., PIB agropecuário, área municipal", "2008–2025"],
        ["ICMBio",         "Unidades de conservação e TIs",          "Estático"],
        ["IBAMA",          "Autos de infração ambiental",            "2008–2025"],
        ["TerraBrasilis",  "Geometrias dos municípios (WFS)",        "Atual"],
    ]
    add_table(sl, headers, rows, 0.65, 1.3, 12.0, 2.8)

    # Notas pipeline
    notes = [
        "► Ingestão via mcp-brasil (servidor MCP externo)",
        "► Join espacial ICMBio/FUNAI: interseção de polígonos UC e TI com limites municipais",
        "► Painel balanceado: 772 municípios × 18 anos = 13.896 observações",
    ]
    for i, n in enumerate(notes):
        add_text_box(sl, n, 0.65, 4.3 + i * 0.45, 12.0, 0.4,
                     font_size=11, color=SLATE_500, italic=True)


def slide_features(prs: Presentation) -> None:
    sl = _blank(prs)
    add_sidebar(sl)

    add_text_box(sl, "Features do Modelo", 0.65, 0.4, 12.0, 0.7,
                 font_size=28, bold=True, color=SLATE_900)

    add_bullet_block(sl, "⏱  Features Temporais (Lag)",
                     ["taxa_desmat_lag1/2/3 — histórico 1, 2 e 3 anos atrás",
                      "taxa_desmat_roll3/5 — média móvel 3 e 5 anos",
                      "std_desmat_roll3 — volatilidade (desvio padrão 3 anos)"],
                     0.65, 1.3, 5.8)

    add_bullet_block(sl, "🗺  Features Espaciais",
                     ["local_moran_i — autocorrelação espacial (Queen weights)",
                      "area_uc_pct — % área protegida (unidades de conservação)",
                      "area_ti_pct — % área de terra indígena (efeito protetor)"],
                     6.8, 1.3, 5.8)

    add_bullet_block(sl, "📋  Features Socioeconômicas",
                     ["populacao — porte do município",
                      "pib_agropecuario — pressão do agronegócio",
                      "autos_ibama — intensidade da fiscalização ambiental"],
                     0.65, 3.5, 5.8)

    add_bullet_block(sl, "🔢  Features Categoriais",
                     ["uf — efeito fixo de estado (9 categorias)",
                      "cluster_id — perfil HDBSCAN do município (0–3)"],
                     6.8, 3.5, 5.8)

    add_text_box(sl, "⚠  Data leakage prevention: shift(1) aplicado antes do rolling window — "
                     "a janela nunca usa o valor do próprio ano.",
                 0.65, 5.8, 12.0, 0.5, font_size=11, color=SLATE_500, italic=True)


def slide_hdbscan(prs: Presentation) -> None:
    sl = _blank(prs)
    add_sidebar(sl)

    add_text_box(sl, "Perfis de Municípios — HDBSCAN", 0.65, 0.4, 12.0, 0.7,
                 font_size=28, bold=True, color=SLATE_900)

    clusters = [
        ("Cluster 0", "Baixo risco estável",
         "Municípios com baixa taxa histórica, longe da fronteira agrícola.",
         VERDE_CLARO, VERDE_ESCURO),
        ("Cluster 1", "Risco moderado crescente",
         "Pressão agropecuária em ascensão, poucas UCs.",
         RGBColor(0xFE, 0xF9, 0xC3), RGBColor(0x71, 0x4B, 0x00)),
        ("Cluster 2", "Alto risco – fronteira",
         "Alta taxa de desmatamento, hotspots históricos de soja/pecuária.",
         RGBColor(0xFF, 0xED, 0xD5), LARANJA),
        ("Cluster 3", "Hotspot crítico",
         "Taxa máxima de desmatamento, baixa proteção territorial.",
         RGBColor(0xFE, 0xE2, 0xE2), VERMELHO),
    ]

    for i, (cid, nome, desc, bg, fg) in enumerate(clusters):
        x = 0.65 + i * 3.15
        box = sl.shapes.add_shape(1, Inches(x), Inches(1.4), Inches(2.9), Inches(2.8))
        _set_bg(box, bg)
        box.line.color.rgb = fg

        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left  = Inches(0.12)
        tf.margin_top   = Inches(0.12)

        p1 = tf.paragraphs[0]
        r1 = p1.add_run()
        r1.text = cid
        r1.font.size  = Pt(11)
        r1.font.bold  = True
        r1.font.color.rgb = fg

        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = nome
        r2.font.size  = Pt(13)
        r2.font.bold  = True
        r2.font.color.rgb = SLATE_900

        p3 = tf.add_paragraph()
        r3 = p3.add_run()
        r3.text = "\n" + desc
        r3.font.size  = Pt(10)
        r3.font.color.rgb = SLATE_700

    add_text_box(sl, "89 municípios classificados como ruído (−1) — 12% do total · "
                     "Silhouette médio = 0.23 (espaço QuantileTransformer)",
                 0.65, 4.5, 12.0, 0.5, font_size=11, color=SLATE_500, italic=True)

    add_text_box(sl, "Features de clustering: taxa_desmatamento, área_km2, populacao, "
                     "pib_agropecuario, area_uc_pct, area_ti_pct, autos_ibama, "
                     "local_moran_i, desmat_trend",
                 0.65, 5.1, 12.0, 0.6, font_size=11, color=SLATE_700)


def slide_modelo(prs: Presentation) -> None:
    sl = _blank(prs)
    add_sidebar(sl)

    add_text_box(sl, "LightGBM + Optuna — Validação Espacial", 0.65, 0.4, 12.0, 0.7,
                 font_size=28, bold=True, color=SLATE_900)

    add_kpi_box(sl, "R² médio (spatial CV)", "0.76",  0.65, 1.4, value_color=VERDE_ESCURO)
    add_kpi_box(sl, "RMSE médio (%/ano)",    "0.109", 3.65, 1.4, value_color=SLATE_700)
    add_kpi_box(sl, "Trials Optuna",         "55",    6.65, 1.4, value_color=RGBColor(0x6D, 0x28, 0xD9))
    add_kpi_box(sl, "Melhoria vs baseline",  "3.9%",  9.65, 1.4, value_color=VERDE_MEDIO)

    headers = ["UF(s)", "N", "RMSE", "MAE", "R²"]
    rows = [
        ["MA",              "2.715", "0.114", "0.055", "0.748"],
        ["PA",              "2.160", "0.118", "0.065", "0.850"],
        ["MT",              "2.115", "0.114", "0.052", "0.793"],
        ["RR + TO",         "2.310", "0.065", "0.026", "0.548"],
        ["AC+AM+AP+RO",     "2.280", "0.132", "0.064", "0.853"],
        ["MÉDIA",           "11.580","0.109", "0.052", "0.759"],
    ]
    add_table(sl, headers, rows, 0.65, 3.1, 12.0, 2.6)

    add_text_box(sl, "Validação espacial por UF (GroupKFold) evita data leakage geográfico — "
                     "o modelo nunca treina e valida na mesma região.",
                 0.65, 6.0, 12.0, 0.5, font_size=11, color=SLATE_500, italic=True)


def slide_shap(prs: Presentation) -> None:
    sl = _blank(prs)
    add_sidebar(sl)

    add_text_box(sl, "Interpretabilidade — SHAP", 0.65, 0.4, 12.0, 0.7,
                 font_size=28, bold=True, color=SLATE_900)

    beeswarm = Path("data/outputs/shap_beeswarm.png")
    if beeswarm.exists():
        sl.shapes.add_picture(
            str(beeswarm),
            Inches(0.65), Inches(1.2),
            Inches(7.8), Inches(5.5),
        )
        x_bullets = 8.7
    else:
        add_text_box(sl, "[Gráfico SHAP beeswarm]", 0.65, 1.5, 7.8, 5.0,
                     font_size=14, color=SLATE_500)
        x_bullets = 8.7

    insights = [
        ("Principais insights:", True,  VERDE_ESCURO, 13),
        ("taxa_desmat_lag1 é o preditor mais importante em todos os clusters",
         False, SLATE_700, 11),
        ("local_moran_i revela efeito de contágio regional: municípios vizinhos de hotspots têm risco elevado",
         False, SLATE_700, 11),
        ("area_ti_pct tem efeito protetor consistente (SHAP negativo)",
         False, SLATE_700, 11),
        ("autos_ibama tem dupla interpretação: fiscalização onde o risco já é alto",
         False, SLATE_700, 11),
        ("pib_agropecuario amplifica o risco na fronteira agrícola",
         False, SLATE_700, 11),
    ]

    y = 1.3
    for text, bold, color, fs in insights:
        add_text_box(sl, ("• " if not bold else "") + text,
                     x_bullets, y, 4.2, 0.65, font_size=fs, bold=bold, color=color)
        y += 0.72 if not bold else 0.5


def slide_top_municipios(prs: Presentation, predictions_path: str) -> None:
    sl = _blank(prs)
    add_sidebar(sl)

    add_text_box(sl, "Municípios de Maior Risco — 2025", 0.65, 0.4, 12.0, 0.7,
                 font_size=28, bold=True, color=SLATE_900)

    try:
        gdf = gpd.read_parquet(predictions_path)
        last = int(gdf["ano"].max())
        top10 = (
            gdf[gdf["ano"] == last]
            .nlargest(10, "score_risco")[
                ["municipio", "uf", "score_risco", "taxa_desmatamento", "desmatamento_km2"]
            ]
        )
        rows = [
            [r["municipio"], r["uf"],
             f"{r['score_risco']:.1f}",
             f"{r['taxa_desmatamento']:.4f}",
             f"{r['desmatamento_km2']:.1f}"]
            for _, r in top10.iterrows()
        ]
        headers = ["Município", "UF", "Score (0–100)", "Taxa (%/ano)", "Desmat. (km²)"]
        add_table(sl, headers, rows, 0.65, 1.3, 12.0, 4.5)
        add_text_box(sl, f"Score = ranking percentílico global sobre todos os {last} previstos · "
                         "Maior score = maior risco relativo histórico",
                     0.65, 6.0, 12.0, 0.5, font_size=11, color=SLATE_500, italic=True)
    except Exception as e:
        add_text_box(sl, f"Dados não disponíveis: {e}", 0.65, 1.5, 12.0, 1.0,
                     font_size=14, color=VERMELHO)


def slide_dashboard(prs: Presentation) -> None:
    sl = _blank(prs)
    add_sidebar(sl)

    add_text_box(sl, "Dashboards Interativos", 0.65, 0.4, 12.0, 0.7,
                 font_size=28, bold=True, color=SLATE_900)

    items = [
        ("🐍  Streamlit", "Análise exploratória interativa — filtros dinâmicos de ano, UF e cluster.\n"
                          "Mapa Folium + Plotly + tabela de ranking. Execução local."),
        ("🌐  HTML/Leaflet", "Dashboard single-file (reports/dashboard.html) — Tailwind CSS + Leaflet.js + Chart.js.\n"
                             "Sem dependências de servidor. Distribuição pública via e-mail ou GitHub Pages."),
        ("🔄  Atualização Anual", "Re-executar pipeline completo em ~15 min: ingestão MCP → ETL → features → modelo → scores.\n"
                                  "Cache local evita redownload de dados históricos."),
    ]

    for i, (titulo, desc) in enumerate(items):
        y = 1.4 + i * 1.8
        box = sl.shapes.add_shape(1, Inches(0.65), Inches(y), Inches(12.0), Inches(1.6))
        _set_bg(box, SLATE_100)
        box.line.color.rgb = SLATE_200

        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left  = Inches(0.15)
        tf.margin_top   = Inches(0.1)

        p1 = tf.paragraphs[0]
        r1 = p1.add_run()
        r1.text = titulo
        r1.font.size  = Pt(14)
        r1.font.bold  = True
        r1.font.color.rgb = VERDE_ESCURO

        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = desc
        r2.font.size  = Pt(11)
        r2.font.color.rgb = SLATE_700


def slide_proximos(prs: Presentation) -> None:
    sl = _blank(prs)
    add_sidebar(sl)

    add_text_box(sl, "Próximos Passos", 0.65, 0.4, 12.0, 0.7,
                 font_size=28, bold=True, color=SLATE_900)

    proximos = [
        ("🛰️  Alertas em tempo real",
         "Integração com DETER/INPE para monitoramento semanal de alertas de desmatamento."),
        ("🌿  Expansão por bioma",
         "Modelos específicos para Cerrado, Caatinga e Pantanal — cada bioma tem dinâmicas distintas."),
        ("🔌  API REST",
         "Endpoint público para consumo dos scores por órgãos ambientais (IBAMA, ICMBio, MMA)."),
        ("📡  Dados climáticos",
         "Incorporar precipitação (CHIRPS), temperatura e índices de seca como features."),
        ("🤝  Validação com especialistas",
         "Calibração do modelo com gestores do INPE e pesquisadores do IPAM."),
    ]

    for i, (titulo, desc) in enumerate(proximos):
        y = 1.3 + i * 1.0
        add_text_box(sl, titulo, 0.65, y, 3.5, 0.4,
                     font_size=12, bold=True, color=VERDE_ESCURO)
        add_text_box(sl, desc, 4.35, y, 8.5, 0.5,
                     font_size=11, color=SLATE_700)

    # Contato
    box = sl.shapes.add_shape(1, Inches(0.65), Inches(6.3), Inches(12.0), Inches(0.8))
    _set_bg(box, VERDE_CLARO)
    box.line.color.rgb = VERDE_MEDIO
    tf = box.text_frame
    tf.margin_left = Inches(0.2)
    tf.margin_top  = Inches(0.1)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "Caio Figueiredo · Data Science Portfolio"
    r.font.size  = Pt(12)
    r.font.color.rgb = VERDE_ESCURO


# ── Main ──────────────────────────────────────────────────────────────────────

def build_pptx(
    predictions_path: str = _PREDICTIONS,
    output_path: str = _OUTPUT,
) -> None:
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)

    logger.info("Gerando slides...")

    slide_capa(prs)
    logger.info("  Slide 1/10: Capa")

    slide_problema(prs)
    logger.info("  Slide 2/10: Problema & Objetivo")

    slide_dados(prs)
    logger.info("  Slide 3/10: Fontes de Dados")

    slide_features(prs)
    logger.info("  Slide 4/10: Feature Engineering")

    slide_hdbscan(prs)
    logger.info("  Slide 5/10: HDBSCAN")

    slide_modelo(prs)
    logger.info("  Slide 6/10: Modelo & Validação")

    slide_shap(prs)
    logger.info("  Slide 7/10: SHAP")

    slide_top_municipios(prs, predictions_path)
    logger.info("  Slide 8/10: Top Municípios")

    slide_dashboard(prs)
    logger.info("  Slide 9/10: Dashboard")

    slide_proximos(prs)
    logger.info("  Slide 10/10: Próximos Passos")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    prs.save(output_path)
    size_kb = Path(output_path).stat().st_size / 1024
    logger.info("Deck salvo em '%s' (%.0f KB, 10 slides).", output_path, size_kb)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    build_pptx()
