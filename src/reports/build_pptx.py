"""
TASK-17: Gerador do slide deck executivo em .pptx.

Produz reports/slide_deck_executivo.pptx com 10 slides widescreen 16:9.
Design moderno: paleta verde/slate, barras de acento, KPI cards, imagens SHAP.
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

# ── Paleta ────────────────────────────────────────────────────────────────────
VERDE_ESCURO  = RGBColor(0x16, 0x65, 0x34)
VERDE_MEDIO   = RGBColor(0x22, 0xC5, 0x5E)
VERDE_CLARO   = RGBColor(0xBB, 0xF7, 0xD0)
VERDE_BG      = RGBColor(0xF0, 0xFD, 0xF4)
VERMELHO      = RGBColor(0xEF, 0x44, 0x44)
VERMELHO_BG   = RGBColor(0xFE, 0xE2, 0xE2)
LARANJA       = RGBColor(0xF9, 0x73, 0x16)
LARANJA_BG    = RGBColor(0xFF, 0xED, 0xD5)
AMARELO       = RGBColor(0xD9, 0x77, 0x06)
AMARELO_BG    = RGBColor(0xFF, 0xFB, 0xEB)
AZUL_MED      = RGBColor(0x0E, 0xA5, 0xE9)
AZUL_BG       = RGBColor(0xEF, 0xF6, 0xFF)
ROXO_MED      = RGBColor(0x79, 0x35, 0xDB)
ROXO_BG       = RGBColor(0xF5, 0xF3, 0xFF)
SLATE_900     = RGBColor(0x0F, 0x17, 0x2A)
SLATE_700     = RGBColor(0x33, 0x4D, 0x6B)
SLATE_500     = RGBColor(0x64, 0x74, 0x8B)
SLATE_300     = RGBColor(0xCB, 0xD5, 0xE1)
SLATE_200     = RGBColor(0xE2, 0xE8, 0xF0)
SLATE_100     = RGBColor(0xF1, 0xF5, 0xF9)
SLATE_50      = RGBColor(0xF8, 0xFA, 0xFC)
BRANCO        = RGBColor(0xFF, 0xFF, 0xFF)

# ── Layout ────────────────────────────────────────────────────────────────────
SLIDE_W   = 13.33
SLIDE_H   = 7.5
MAR_L     = 0.55          # margem esquerda padrão
MAR_R     = 12.75         # borda direita
CONT_W    = MAR_R - MAR_L  # 12.2"
TITLE_T   = 0.28          # topo do título
CONT_T    = 1.08          # início da área de conteúdo
FOOTER_T  = 6.98

_PREDICTIONS = "data/outputs/predictions.parquet"
_METRICS     = "data/outputs/metrics_by_fold.json"
_OUTPUT      = "reports/slide_deck_executivo.pptx"

_SHAP_BEESWARM  = "data/outputs/shap_beeswarm.png"
_SHAP_CLUSTER   = "data/outputs/shap_por_cluster.png"
_ARQUITETURA    = "reports/arquitetura.png"


# ── Primitivos ────────────────────────────────────────────────────────────────

def _fill(shape, color: RGBColor) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = color


def _no_line(shape) -> None:
    shape.line.fill.background()


def _rect(slide, left, top, width, height, color: RGBColor,
          line_color: RGBColor | None = None, rounded: bool = False):
    shape_id = 5 if rounded else 1
    s = slide.shapes.add_shape(shape_id, Inches(left), Inches(top), Inches(width), Inches(height))
    _fill(s, color)
    if line_color:
        s.line.color.rgb = line_color
    else:
        _no_line(s)
    return s


def _oval(slide, left, top, width, height, color: RGBColor, alpha_sim: bool = False):
    s = slide.shapes.add_shape(9, Inches(left), Inches(top), Inches(width), Inches(height))
    _fill(s, color)
    _no_line(s)
    return s


def _blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def txt(slide, text: str, left, top, width, height,
        size: int = 14, bold: bool = False, italic: bool = False,
        color: RGBColor = SLATE_900,
        align: PP_ALIGN = PP_ALIGN.LEFT,
        wrap: bool = True) -> None:
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color


def txt_multi(slide, lines: list[tuple], left, top, width, height, wrap: bool = True) -> None:
    """lines: [(text, size, bold, italic, color, align)]"""
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = wrap
    for i, (text, size, bold, italic, color, align) in enumerate(lines):
        p = tf.paragraphs[i] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color


# ── Componentes reutilizáveis ─────────────────────────────────────────────────

def add_slide_bg(slide, color: RGBColor = BRANCO) -> None:
    _rect(slide, 0, 0, SLIDE_W, SLIDE_H, color)


def add_content_header(slide, title: str, accent: RGBColor = VERDE_ESCURO,
                       subtitle: str | None = None) -> None:
    """Faixa superior com título, linha de acento e número do slide."""
    # Faixa de fundo levíssima
    _rect(slide, 0, 0, SLIDE_W, 0.95, SLATE_50)
    # Barra de acento esquerda vertical
    _rect(slide, MAR_L, TITLE_T, 0.06, 0.58, accent)
    # Título
    txt(slide, title, MAR_L + 0.14, TITLE_T, CONT_W - 0.14, 0.62,
        size=26, bold=True, color=SLATE_900)
    if subtitle:
        txt(slide, subtitle, MAR_L + 0.14, 0.80, CONT_W - 0.14, 0.28,
            size=11, color=SLATE_500, italic=True)
    # Linha divisória
    _rect(slide, 0, 0.96, SLIDE_W, 0.025, SLATE_200)


def add_footer(slide, page: int, total: int = 10,
               label: str = "Preditor de Risco · Amazônia Legal") -> None:
    _rect(slide, 0, FOOTER_T, SLIDE_W, 0.525, SLATE_50)
    _rect(slide, 0, FOOTER_T, SLIDE_W, 0.02, SLATE_200)
    txt(slide, label, MAR_L, FOOTER_T + 0.07, 9.0, 0.35, size=9, color=SLATE_500)
    txt(slide, f"{page} / {total}", 12.4, FOOTER_T + 0.07, 0.8, 0.35,
        size=9, color=SLATE_500, align=PP_ALIGN.RIGHT)


def kpi_card(slide, value: str, label: str, note: str,
             left: float, top: float, width: float = 2.85, height: float = 1.35,
             accent: RGBColor = VERDE_ESCURO, bg: RGBColor = SLATE_50) -> None:
    """KPI card com barra de acento esquerda."""
    _rect(slide, left, top, 0.055, height, accent)
    box = _rect(slide, left + 0.055, top, width - 0.055, height, bg,
                line_color=SLATE_200)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left  = Inches(0.15)
    tf.margin_top   = Inches(0.12)
    p1 = tf.paragraphs[0]
    r1 = p1.add_run()
    r1.text = value
    r1.font.size  = Pt(28)
    r1.font.bold  = True
    r1.font.color.rgb = accent
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = label
    r2.font.size  = Pt(11)
    r2.font.bold  = True
    r2.font.color.rgb = SLATE_700
    p3 = tf.add_paragraph()
    r3 = p3.add_run()
    r3.text = note
    r3.font.size  = Pt(9)
    r3.font.color.rgb = SLATE_500


def cluster_card(slide, cluster_id: str, title: str, desc: str,
                 stats: list[tuple[str, str]],
                 left: float, top: float, width: float, height: float,
                 accent: RGBColor, bg: RGBColor) -> None:
    """Card de cluster com badge, título, descrição e stats."""
    # Fundo
    box = _rect(slide, left, top, width, height, bg, line_color=SLATE_200)
    # Faixa superior colorida
    _rect(slide, left, top, width, 0.32, accent)

    # Badge + título na faixa
    txt(slide, cluster_id, left + 0.15, top + 0.04, 0.8, 0.26,
        size=11, bold=True, color=BRANCO)
    txt(slide, title, left + 0.7, top + 0.03, width - 0.85, 0.29,
        size=11, bold=True, color=BRANCO)

    # Descrição
    txt(slide, desc, left + 0.12, top + 0.40, width - 0.24, 0.65,
        size=9.5, color=SLATE_700)

    # Stats
    y_s = top + 1.10
    for i, (label, val) in enumerate(stats):
        x_s = left + 0.12 + i * (width / len(stats))
        txt(slide, val, x_s, y_s, width / len(stats) - 0.05, 0.30,
            size=11, bold=True, color=accent)
        txt(slide, label, x_s, y_s + 0.28, width / len(stats) - 0.05, 0.25,
            size=8, color=SLATE_500)


def bullet_row(slide, icon: str, title: str, desc: str,
               left: float, top: float, width: float = CONT_W) -> None:
    """Linha de bullet com ícone, título em negrito e descrição."""
    txt(slide, icon, left, top, 0.45, 0.42, size=16)
    txt(slide, title, left + 0.45, top, width * 0.28, 0.38, size=11, bold=True, color=VERDE_ESCURO)
    txt(slide, desc, left + 0.45 + width * 0.28, top, width * 0.68, 0.40, size=10, color=SLATE_700)


def add_table(slide, headers: list[str], rows: list[list],
              left: float, top: float, width: float, height: float,
              font_size: int = 11, header_bg: RGBColor = VERDE_ESCURO) -> None:
    tbl = slide.shapes.add_table(
        len(rows) + 1, len(headers),
        Inches(left), Inches(top), Inches(width), Inches(height)
    ).table
    col_w = width / len(headers)
    for i in range(len(headers)):
        tbl.columns[i].width = Inches(col_w)
    for ci, h in enumerate(headers):
        cell = tbl.cell(0, ci)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_bg
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.runs[0] if p.runs else p.add_run()
        r.font.bold = True
        r.font.size = Pt(font_size)
        r.font.color.rgb = BRANCO
    for ri, row in enumerate(rows):
        bg = SLATE_50 if ri % 2 == 0 else BRANCO
        for ci, val in enumerate(row):
            cell = tbl.cell(ri + 1, ci)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.runs[0] if p.runs else p.add_run()
            r.font.size = Pt(font_size)
            r.font.color.rgb = SLATE_900


def mini_bar_chart(slide, values: list[tuple[str, float]],
                   left: float, top: float, width: float, height: float,
                   bar_color: RGBColor = VERDE_ESCURO,
                   alert_color: RGBColor = VERMELHO,
                   alert_threshold: float | None = None) -> None:
    """Gráfico de barras horizontais simples desenhado com shapes."""
    max_val = max(v for _, v in values)
    bar_area_w = width - 1.6   # reserva espaço para rótulos
    bar_h = height / len(values) * 0.65
    gap = height / len(values) * 0.35

    for i, (label, val) in enumerate(values):
        y = top + i * (bar_h + gap)
        bw = bar_area_w * (val / max_val)
        color = alert_color if (alert_threshold and val >= alert_threshold) else bar_color
        # Barra
        _rect(slide, left + 1.5, y, bw, bar_h, color)
        # Rótulo do eixo
        txt(slide, label, left, y, 1.45, bar_h, size=8, color=SLATE_700,
            align=PP_ALIGN.RIGHT)
        # Valor
        txt(slide, f"{val:,.0f}", left + 1.5 + bw + 0.05, y, 0.9, bar_h,
            size=8, bold=True, color=color)


# ── Slides ────────────────────────────────────────────────────────────────────

def slide_capa(prs: Presentation) -> None:
    sl = _blank(prs)
    # Fundo verde escuro
    _rect(sl, 0, 0, SLIDE_W, SLIDE_H, VERDE_ESCURO)
    # Círculos decorativos (simulação de depth)
    _oval(sl, 8.8, -1.2, 5.5, 5.5, RGBColor(0x10, 0x4D, 0x27))
    _oval(sl, 10.5, 1.2, 3.8, 3.8, RGBColor(0x0D, 0x3E, 0x1F))
    _oval(sl, 7.5, 4.5, 2.8, 2.8, RGBColor(0x1A, 0x7A, 0x3C))
    # Faixa de acento verde médio (bottom)
    _rect(sl, 0, 6.7, SLIDE_W, 0.8, RGBColor(0x0D, 0x3E, 0x1F))

    # Ícone
    txt(sl, "🌿", 0.55, 0.55, 1.8, 1.8, size=68, color=BRANCO)

    # Título
    txt(sl, "Preditor de Risco de Desmatamento",
        0.55, 1.85, 11.0, 1.3, size=40, bold=True, color=BRANCO)
    # Linha de acento
    _rect(sl, 0.55, 3.08, 5.2, 0.055, VERDE_MEDIO)
    # Subtítulo
    txt(sl, "Amazônia Legal  ·  Machine Learning para Conservação",
        0.55, 3.22, 11.0, 0.65, size=18, color=VERDE_CLARO)

    # Stats no rodapé
    stats = [("808", "municípios"), ("9", "estados"),
             ("18 anos", "2008–2025"), ("2026", "previsão")]
    for i, (val, lab) in enumerate(stats):
        x = 0.55 + i * 3.0
        txt(sl, val, x, 6.72, 2.8, 0.36, size=16, bold=True, color=BRANCO,
            align=PP_ALIGN.CENTER)
        txt(sl, lab, x, 6.98, 2.8, 0.3, size=10, color=VERDE_CLARO,
            align=PP_ALIGN.CENTER)


def slide_problema(prs: Presentation) -> None:
    sl = _blank(prs)
    add_slide_bg(sl)
    add_content_header(sl, "O Problema & Objetivo", VERMELHO)
    add_footer(sl, 2)

    items = [
        ("🌳", "5,2 milhões km²",
         "A Amazônia Legal cobre 59% do território brasileiro — o maior bioma tropical do planeta."),
        ("📈", "Reaceleração pós-2019",
         "Após a mínima histórica em 2012, o desmatamento acelerou: 12.415 km² em 2021 (pico recente)."),
        ("⏰", "Resposta tardia",
         "Alertas INPE chegam com 12–18 meses de defasagem. Fiscalização precisa de antecipação."),
        ("🤖", "Nossa solução",
         "Score 0–100 por município, recalculável anualmente. Identifica hotspots antes da perda florestal."),
    ]

    for i, (icon, title, desc) in enumerate(items):
        y = CONT_T + 0.1 + i * 1.2
        _rect(sl, MAR_L, y, CONT_W, 1.05, SLATE_50, line_color=SLATE_200)
        txt(sl, icon, MAR_L + 0.15, y + 0.18, 0.55, 0.7, size=24)
        txt(sl, title, MAR_L + 0.82, y + 0.08, 2.8, 0.38, size=13, bold=True, color=SLATE_900)
        txt(sl, desc, MAR_L + 0.82, y + 0.50, CONT_W - 1.05, 0.48, size=11, color=SLATE_700)

    # Box objetivo destaque
    _rect(sl, MAR_L, 6.08, CONT_W, 0.72, VERDE_BG, line_color=VERDE_MEDIO)
    txt(sl, "🎯  Objetivo: Score de risco 0–100 por município · Rank percentílico anual · "
           "Atualizável a cada publicação INPE/PRODES",
        MAR_L + 0.2, 6.13, CONT_W - 0.4, 0.60,
        size=11, bold=True, color=VERDE_ESCURO)


def slide_dados(prs: Presentation) -> None:
    sl = _blank(prs)
    add_slide_bg(sl)
    add_content_header(sl, "Pipeline de Dados",
                       subtitle="Todas as fontes são APIs públicas acessadas via mcp-brasil")
    add_footer(sl, 3)

    # Tabela de fontes
    headers = ["Fonte", "Dados", "Cobertura", "Frequência"]
    rows = [
        ["INPE / PRODES",   "Desmatamento km² por município",          "2008–2025", "Anual"],
        ["IBGE",            "População, PIB agropecuário, área",        "2008–2025", "Anual"],
        ["ICMBio / FUNAI",  "Unidades de conservação + Terras Indígenas","Estático",  "—"],
        ["IBAMA",           "Autos de infração ambiental",              "2008–2025", "Anual"],
        ["IBGE FTP",        "Shapefile de municípios (geometrias)",     "2022",       "—"],
    ]
    add_table(sl, headers, rows, MAR_L, CONT_T + 0.05, CONT_W, 2.55, font_size=11)

    # KPIs de escala
    kpi_card(sl, "808",   "municípios", "Amazônia Legal", 0.55, 3.88, 2.8, 1.0, VERDE_ESCURO)
    kpi_card(sl, "9",     "estados",    "AM, PA, MT, RO, AC, AP, RR, TO, MA", 3.5, 3.88, 2.8, 1.0, SLATE_700)
    kpi_card(sl, "14.544","observações","históricas (808 × 18 anos)", 6.45, 3.88, 2.8, 1.0, AZUL_MED)
    kpi_card(sl, "808",   "previsões",  "2026 via LightGBM", 9.4, 3.88, 2.8, 1.0, LARANJA)

    # Notas pipeline
    notas = [
        "► Cache local em data/raw/ — API chamada apenas uma vez por dataset",
        "► Join espacial ICMBio/FUNAI: interseção de polígonos UC e TI com limites municipais (GeoPandas)",
        "► Dados IBGE com lacunas preenchidas por forward-fill anual",
    ]
    for i, n in enumerate(notas):
        txt(sl, n, MAR_L, 5.15 + i * 0.38, CONT_W, 0.35, size=10, color=SLATE_500, italic=True)

    # Barra histórica simplificada (desflorestamento por ano)
    years_data = [
        ("2008", 13289), ("2012", 4427), ("2015", 6117),
        ("2019", 10895), ("2021", 12416), ("2023", 8025),
        ("2025", 5258),
    ]
    txt(sl, "Desmatamento total — Amazônia Legal (km²/ano, seleção)",
        MAR_L, 6.3, CONT_W, 0.3, size=9, bold=True, color=SLATE_700)
    max_v = max(v for _, v in years_data)
    bar_w_total = CONT_W / len(years_data)
    for i, (yr, val) in enumerate(years_data):
        x = MAR_L + i * bar_w_total
        bh = 0.5 * (val / max_v)
        color = VERMELHO if val >= 10000 else VERDE_ESCURO
        _rect(sl, x + 0.08, 6.88 - bh, bar_w_total * 0.72, bh, color)
        txt(sl, yr, x, 6.88, bar_w_total, 0.25, size=7, color=SLATE_700, align=PP_ALIGN.CENTER)


def slide_features(prs: Presentation) -> None:
    sl = _blank(prs)
    add_slide_bg(sl)
    add_content_header(sl, "Feature Engineering",
                       subtitle="Lag features, rolling window e análise de autocorrelação espacial")
    add_footer(sl, 4)

    groups = [
        ("⏱  Temporais (Lag & Rolling)", VERDE_ESCURO, VERDE_BG, [
            "taxa_desmat_lag1/2/3 — histórico 1, 2 e 3 anos atrás",
            "taxa_desmat_roll3/5  — média móvel 3 e 5 anos",
            "std_desmat_roll3     — volatilidade (desvio padrão 3 anos)",
        ]),
        ("🗺  Espaciais", AZUL_MED, AZUL_BG, [
            "local_moran_i  — autocorrelação espacial (Queen weights)",
            "area_uc_pct    — % área protegida por UCs",
            "area_ti_pct    — % área de Terra Indígena",
        ]),
        ("📋  Socioeconômicas", LARANJA, LARANJA_BG, [
            "populacao        — porte e urbanização do município",
            "pib_agropecuario — pressão do agronegócio",
            "autos_ibama      — intensidade da fiscalização",
        ]),
        ("🔢  Categoriais", ROXO_MED, ROXO_BG, [
            "uf         — efeito fixo de estado (9 categorias)",
            "cluster_id — perfil HDBSCAN do município (0–3)",
        ]),
    ]

    for i, (title, accent, bg, bullets) in enumerate(groups):
        col = i % 2
        row = i // 2
        x = MAR_L + col * 6.1
        y = CONT_T + 0.1 + row * 2.35
        bh = 2.22
        _rect(sl, x, y, 5.85, bh, bg, line_color=SLATE_200)
        _rect(sl, x, y, 5.85, 0.36, accent)
        txt(sl, title, x + 0.12, y + 0.04, 5.6, 0.30, size=11, bold=True, color=BRANCO)
        for j, b in enumerate(bullets):
            txt(sl, "• " + b, x + 0.14, y + 0.45 + j * 0.46, 5.55, 0.44,
                size=10, color=SLATE_700)

    txt(sl, "⚠  Data leakage prevention: shift(1) antes do rolling — a janela nunca usa o valor do próprio ano-alvo.",
        MAR_L, 6.75, CONT_W, 0.40, size=10, italic=True, color=SLATE_500)


def slide_hdbscan(prs: Presentation) -> None:
    sl = _blank(prs)
    add_slide_bg(sl)
    add_content_header(sl, "Perfis de Municípios — Clusterização HDBSCAN",
                       subtitle="Features estruturais, sem o target — evita raciocínio circular")
    add_footer(sl, 5)

    # 4 clusters em grade 2×2
    clusters = [
        ("C0", "Alta Proteção Ambiental",
         "Alta cobertura de UCs (38%). Tendência de desmatamento decrescente (0.69×). "
         "Municípios com floresta protegida por instrumentos legais.",
         [("212", "municípios"), ("0.116%", "taxa média/ano"), ("38%", "área UC")],
         VERDE_ESCURO, VERDE_BG),
        ("C1", "Periféricos Estáveis",
         "Menor porte e PIB. Sem UCs ou TIs relevantes. "
         "Baixa pressão agropecuária e desmatamento em queda (0.65×).",
         [("312", "municípios"), ("0.117%", "taxa média/ano"), ("0.3%", "área UC")],
         AZUL_MED, AZUL_BG),
        ("C2", "Fronteira com Terras Indígenas",
         "Alta taxa de desmatamento histórica. Presença de TIs (21%) como barreira "
         "natural. Alto PIB agropecuário — fronteira de expansão.",
         [("127", "municípios"), ("0.183%", "taxa média/ano"), ("21%", "área TI")],
         LARANJA, LARANJA_BG),
        ("C3", "Grandes · Pressão Crescente ⚠",
         "Maior área e população. Alta fiscalização (IBAMA). "
         "Tendência preocupante de alta (1.74×) apesar de proteção formal.",
         [("157", "municípios"), ("0.176%", "taxa média/ano"), ("25%", "UC+TI")],
         VERMELHO, VERMELHO_BG),
    ]

    for i, (cid, title, desc, stats, accent, bg) in enumerate(clusters):
        col = i % 2
        row = i // 2
        x = MAR_L + col * 6.1
        y = CONT_T + 0.08 + row * 2.6
        cluster_card(sl, cid, title, desc, stats, x, y, 5.85, 2.45, accent, bg)

    # Rodapé métricas clustering
    txt(sl, f"Silhouette = 0.2881  ·  0 municípios como ruído  ·  "
           f"Features: área, populacao, pib_agropecuario, area_uc_pct, area_ti_pct, "
           f"autos_ibama, local_moran_i",
        MAR_L, 6.76, CONT_W, 0.38, size=9, italic=True, color=SLATE_500)


def slide_modelo(prs: Presentation) -> None:
    sl = _blank(prs)
    add_slide_bg(sl)
    add_content_header(sl, "Modelo — LightGBM + Optuna + Validação Espacial",
                       subtitle="GroupKFold por UF — municípios vizinhos nunca no mesmo fold")
    add_footer(sl, 6)

    # KPIs
    kpi_card(sl, "0.759", "R² médio (spatial CV)", "todos os 5 folds",     0.55, CONT_T + 0.1, 2.85, 1.1, VERDE_ESCURO)
    kpi_card(sl, "0.109", "RMSE médio (%/ano)",    "taxa de desmatamento", 3.55, CONT_T + 0.1, 2.85, 1.1, SLATE_700)
    kpi_card(sl, "55",    "Trials Optuna",          "função objetivo: RMSE espacial", 6.55, CONT_T + 0.1, 2.85, 1.1, ROXO_MED)
    kpi_card(sl, "629",   "Estimadores finais",     "n_estimators (LightGBM)", 9.55, CONT_T + 0.1, 2.85, 1.1, AZUL_MED)

    # Tabela por fold
    headers = ["Fold", "Estado(s)", "N", "RMSE", "MAE", "R²", "WMAPE"]
    rows = [
        ["1", "MA",          "2.715", "0.114", "0.055", "0.748", "30.1%"],
        ["2", "PA",          "2.160", "0.118", "0.065", "0.850", "24.7%"],
        ["3", "MT",          "2.115", "0.114", "0.052", "0.793", "31.9%"],
        ["4", "RR + TO",     "2.310", "0.065", "0.026", "0.548", "56.0%"],
        ["5", "AC+AM+AP+RO", "2.280", "0.132", "0.064", "0.853", "29.2%"],
        ["—", "MÉDIA",       "11.580","0.109", "0.052", "0.759", "34.4%"],
    ]
    add_table(sl, headers, rows, MAR_L, CONT_T + 1.4, CONT_W, 2.8, font_size=10)

    txt(sl, "⚠  Fold 4 (RR+TO): R²=0.548 — regiões esparsas com dados históricos limitados. "
           "Resultado documentado e esperado; média nacional ainda acima de 0.75.",
        MAR_L, 6.50, CONT_W, 0.50, size=10, italic=True, color=SLATE_500)


def slide_shap(prs: Presentation) -> None:
    sl = _blank(prs)
    add_slide_bg(sl)
    add_content_header(sl, "Interpretabilidade — SHAP",
                       subtitle="Feature importance global e explicações por município")
    add_footer(sl, 7)

    # Imagem beeswarm à esquerda
    beeswarm = Path(_SHAP_BEESWARM)
    if beeswarm.exists():
        sl.shapes.add_picture(str(beeswarm), Inches(MAR_L), Inches(CONT_T + 0.05),
                              Inches(7.5), Inches(5.3))
        x_right = 8.3
    else:
        _rect(sl, MAR_L, CONT_T + 0.05, 7.5, 5.3, SLATE_100, line_color=SLATE_200)
        txt(sl, "[shap_beeswarm.png]", MAR_L + 3.0, CONT_T + 2.5, 1.5, 0.4, size=10, color=SLATE_500)
        x_right = 8.3

    # Insights à direita
    insights = [
        (VERDE_ESCURO, True,  "Principais insights"),
        (SLATE_700,    False, "taxa_desmat_lag1 é o preditor mais relevante em todos os clusters"),
        (SLATE_700,    False, "local_moran_i revela contágio regional — vizinhos de hotspots têm risco elevado"),
        (SLATE_700,    False, "area_ti_pct tem efeito protetor consistente (SHAP negativo)"),
        (SLATE_700,    False, "autos_ibama: dupla interpretação — fiscalização onde risco já é alto"),
        (SLATE_700,    False, "pib_agropecuario amplifica risco na fronteira do agronegócio"),
        (SLATE_700,    False, "std_desmat_roll3 captura instabilidade — regiões voláteis têm score maior"),
    ]
    y = CONT_T + 0.1
    for color, bold, text in insights:
        prefix = "" if bold else "  • "
        txt(sl, prefix + text, x_right, y, SLIDE_W - x_right - 0.3, 0.55,
            size=11 if bold else 10, bold=bold, color=color)
        y += 0.62 if bold else 0.57

    # SHAP por cluster
    shap_cl = Path(_SHAP_CLUSTER)
    if shap_cl.exists() and y < 6.3:
        txt(sl, "SHAP por cluster:", x_right, y + 0.05, 4.5, 0.3, size=10, bold=True, color=VERDE_ESCURO)
        sl.shapes.add_picture(str(shap_cl), Inches(x_right), Inches(y + 0.35),
                              Inches(SLIDE_W - x_right - 0.3), Inches(5.9 - y))


def slide_top_municipios(prs: Presentation, predictions_path: str) -> None:
    sl = _blank(prs)
    add_slide_bg(sl)
    add_content_header(sl, "Municípios de Maior Risco",
                       subtitle="Score 0–100 = rank percentílico anual · maior valor = maior risco relativo")
    add_footer(sl, 8)

    try:
        gdf = gpd.read_parquet(predictions_path)

        # ── Top 10 histórico (2025) ──────────────────────────────────────────
        top10_25 = (
            gdf[gdf["ano"] == 2025]
            .nlargest(10, "score_risco")[
                ["municipio", "uf", "score_risco", "taxa_desmatamento", "desmatamento_km2"]
            ]
        )
        rows_25 = [
            [r["municipio"], r["uf"],
             f"{r['score_risco']:.1f}",
             f"{r['taxa_desmatamento']:.3f}%",
             f"{r['desmatamento_km2']:.0f} km²"]
            for _, r in top10_25.iterrows()
        ]

        # ── Top 5 previsão 2026 ──────────────────────────────────────────────
        top5_26 = (
            gdf[gdf["ano"] == 2026]
            .nlargest(5, "score_risco")[
                ["municipio", "uf", "score_risco", "taxa_desmatamento_prevista"]
            ]
        )

        # Tabela 2025 à esquerda
        txt(sl, "🏆  Top 10 · Dados Reais 2025",
            MAR_L, CONT_T + 0.05, 7.5, 0.38, size=12, bold=True, color=VERDE_ESCURO)
        headers = ["Município", "UF", "Score", "Taxa", "Desmat."]
        add_table(sl, headers, rows_25, MAR_L, CONT_T + 0.5, 7.2, 5.55,
                  font_size=9.5, header_bg=VERDE_ESCURO)

        # Previsão 2026 à direita
        txt(sl, "📡  Top 5 · Previsão 2026",
            7.95, CONT_T + 0.05, 4.8, 0.38, size=12, bold=True, color=LARANJA)
        _rect(sl, 7.95, CONT_T + 0.5, 5.0, 0.42, LARANJA, line_color=LARANJA)
        for ci, h in enumerate(["Município", "UF", "Taxa est."]):
            txt(sl, h, 7.95 + ci * 1.65, CONT_T + 0.52, 1.6, 0.36,
                size=10, bold=True, color=BRANCO, align=PP_ALIGN.CENTER)

        for ri, (_, r) in enumerate(top5_26.iterrows()):
            bg = LARANJA_BG if ri % 2 == 0 else BRANCO
            y_r = CONT_T + 0.94 + ri * 0.62
            _rect(sl, 7.95, y_r, 5.0, 0.60, bg, line_color=SLATE_200)
            vals = [r["municipio"], r["uf"], f"{r['taxa_desmatamento_prevista']:.3f}%"]
            for ci, v in enumerate(vals):
                txt(sl, v, 7.95 + ci * 1.65 + 0.06, y_r + 0.10, 1.55, 0.42,
                    size=9, color=SLATE_900, align=PP_ALIGN.CENTER)

        # Nota metodológica
        txt(sl, "Score = rank percentílico · 100 = município mais crítico daquele ano · "
               "2026 usa LightGBM com features de 2025 como proxy",
            MAR_L, 6.72, CONT_W, 0.42, size=9, italic=True, color=SLATE_500)

    except Exception as e:
        txt(sl, f"Dados indisponíveis: {e}", MAR_L, CONT_T + 0.5, CONT_W, 1.0,
            size=13, color=VERMELHO)


def slide_dashboard(prs: Presentation) -> None:
    sl = _blank(prs)
    add_slide_bg(sl)
    add_content_header(sl, "Dashboard Interativo — HTML Single-File",
                       subtitle="Disponível em: caioestrella.github.io/desmatamento-amazonia")
    add_footer(sl, 9)

    # Abas do dashboard
    abas = [
        ("🗺  Mapa Coroplético",       VERDE_ESCURO, VERDE_BG,
         "Score 0–100 por município. Filtrável por ano (2008–2026) e estado. "
         "Leaflet.js + GeoJSON inline."),
        ("📊  Ranking & KPIs",          AZUL_MED, AZUL_BG,
         "Top-20 municípios de maior risco. KPIs dinâmicos: desmatamento total, "
         "taxa máxima, municípios acima de 0,2%/ano."),
        ("🔵  Clusters HDBSCAN",        LARANJA, LARANJA_BG,
         "Perfis estruturais dos 4 clusters. Cards descritivos + mapa colorido "
         "por tipo de município."),
        ("📈  Série Temporal",           ROXO_MED, ROXO_BG,
         "Evolução do desmatamento por estado (2008–2026). Chart.js interativo."),
        ("🧠  SHAP",                    RGBColor(0x06, 0x82, 0xC7), AZUL_BG,
         "Importância global das features + waterfall dos municípios mais críticos."),
    ]

    for i, (aba, accent, bg, desc) in enumerate(abas):
        row = i // 2
        col = i % 2
        x = MAR_L + col * 6.1
        y = CONT_T + 0.08 + row * 1.7
        if i == 4:
            x = MAR_L + 3.05  # centralizado
        _rect(sl, x, y, 5.85, 1.55, bg, line_color=SLATE_200)
        _rect(sl, x, y, 5.85, 0.34, accent)
        txt(sl, aba, x + 0.12, y + 0.04, 5.6, 0.28, size=11, bold=True, color=BRANCO)
        txt(sl, desc, x + 0.12, y + 0.42, 5.6, 1.06, size=9.5, color=SLATE_700)

    # Badge GitHub Pages
    _rect(sl, MAR_L, 6.6, CONT_W, 0.52, VERDE_BG, line_color=VERDE_MEDIO)
    txt(sl, "🌐  HTML único  ·  6,5 MB  ·  Sem servidor  ·  Funciona offline após carregamento  ·  "
           "Tema claro / escuro  ·  Responsivo para mobile",
        MAR_L + 0.2, 6.64, CONT_W - 0.4, 0.44, size=10, bold=True, color=VERDE_ESCURO)


def slide_proximos(prs: Presentation) -> None:
    sl = _blank(prs)
    add_slide_bg(sl)
    add_content_header(sl, "Próximos Passos & Limitações",
                       SLATE_700, subtitle="Trabalhos futuros e decisões metodológicas documentadas")
    add_footer(sl, 10)

    # Próximos passos
    txt(sl, "Trabalhos futuros", MAR_L, CONT_T + 0.08, 5.8, 0.36,
        size=12, bold=True, color=VERDE_ESCURO)
    proximos = [
        ("📅", "Validação temporal",
         "CV walk-forward (prever ano t com dados até t-1) — necessária para avaliar robustez pós-2019."),
        ("🛰", "Alertas DETER",
         "Integração com dados quinzenais do DETER/INPE para monitoramento em tempo quase-real."),
        ("📡", "Features climáticas",
         "Incorporar precipitação (CHIRPS) e índices de seca (SPEI) como proxies de pressão antrópica."),
        ("🔌", "API REST",
         "Endpoint público para consumo por órgãos como IBAMA, ICMBio e secretarias estaduais."),
    ]
    for i, (icon, title, desc) in enumerate(proximos):
        y = CONT_T + 0.52 + i * 0.88
        txt(sl, icon, MAR_L, y, 0.45, 0.55, size=16)
        txt(sl, title, MAR_L + 0.48, y + 0.03, 2.2, 0.35, size=11, bold=True, color=VERDE_ESCURO)
        txt(sl, desc, MAR_L + 0.48 + 2.2, y, 2.9, 0.55, size=10, color=SLATE_700)

    # Limitações
    txt(sl, "Limitações documentadas", 6.65, CONT_T + 0.08, 6.4, 0.36,
        size=12, bold=True, color=VERMELHO)
    limitacoes = [
        ("CV geográfica ≠ temporal", "Score pode ser otimista para anos de ruptura"),
        ("Score é ranking, não probabilidade", "Score 80 = 80º percentil naquele ano"),
        ("R² baixo em RR+TO",  "Regiões esparsas — dados históricos limitados"),
        ("Features 2026 = proxies 2025", "INPE/IBGE publicam com ~12 meses de defasagem"),
        ("SHAP em dados de treino", "Sem holdout dedicado para explicabilidade"),
    ]
    for i, (lim, obs) in enumerate(limitacoes):
        y = CONT_T + 0.52 + i * 0.82
        _rect(sl, 6.65, y, 0.25, 0.40, VERMELHO)
        txt(sl, lim, 7.0, y, 3.5, 0.38, size=10, bold=True, color=SLATE_900)
        txt(sl, obs, 7.0, y + 0.38, 6.0, 0.38, size=9, color=SLATE_700)

    # Contato
    _rect(sl, MAR_L, 6.55, CONT_W, 0.62, VERDE_BG, line_color=VERDE_MEDIO)
    txt(sl, "Caio Figueiredo  ·  Data Science Portfolio  ·  "
           "github.com/CaioEstrella/desmatamento-amazonia",
        MAR_L + 0.2, 6.60, CONT_W - 0.4, 0.52, size=11, bold=True, color=VERDE_ESCURO)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_pptx(predictions_path: str = _PREDICTIONS,
               output_path: str = _OUTPUT) -> None:
    prs = Presentation()
    prs.slide_width  = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    slides = [
        ("Capa",                  lambda: slide_capa(prs)),
        ("Problema & Objetivo",   lambda: slide_problema(prs)),
        ("Pipeline de Dados",     lambda: slide_dados(prs)),
        ("Feature Engineering",   lambda: slide_features(prs)),
        ("HDBSCAN",               lambda: slide_hdbscan(prs)),
        ("Modelo & Validação",    lambda: slide_modelo(prs)),
        ("SHAP",                  lambda: slide_shap(prs)),
        ("Top Municípios",        lambda: slide_top_municipios(prs, predictions_path)),
        ("Dashboard",             lambda: slide_dashboard(prs)),
        ("Próximos Passos",       lambda: slide_proximos(prs)),
    ]

    for i, (name, fn) in enumerate(slides, 1):
        fn()
        logger.info("  Slide %d/10: %s", i, name)

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
