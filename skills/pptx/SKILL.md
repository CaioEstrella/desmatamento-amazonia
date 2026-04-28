# SKILL: pptx

## Propósito

Gerar um **slide deck executivo** em formato `.pptx` (PowerPoint) usando a biblioteca
`python-pptx`. O deck deve ser profissional, visual e adequado para portfólio de
Data Science — comunicando resultados técnicos para uma audiência de negócio.

---

## Biblioteca

```python
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
```

`python-pptx` já está no `requirements.txt` do projeto. Não instalar nada.

---

## Paleta de cores

```python
VERDE_ESCURO  = RGBColor(0x16, 0x65, 0x34)   # #166534 (green-800)
VERDE_MEDIO   = RGBColor(0x22, 0xC5, 0x5E)   # #22c55e (green-500)
VERDE_CLARO   = RGBColor(0xBB, 0xF7, 0xD0)   # #bbf7d0 (green-100)
VERMELHO      = RGBColor(0xEF, 0x44, 0x44)   # #ef4444 (red-500)
LARANJA       = RGBColor(0xF9, 0x73, 0x16)   # #f97316 (orange-500)
AMARELO       = RGBColor(0xEA, 0xB3, 0x08)   # #eab308 (yellow-500)
SLATE_900     = RGBColor(0x0F, 0x17, 0x2A)   # #0f172a (texto principal)
SLATE_500     = RGBColor(0x64, 0x74, 0x8B)   # #64748b (texto secundário)
SLATE_100     = RGBColor(0xF1, 0xF5, 0xF9)   # #f1f5f9 (fundo claro)
BRANCO        = RGBColor(0xFF, 0xFF, 0xFF)
```

---

## Estrutura dos slides

### Slide 1 — Capa
- Layout: fundo `VERDE_ESCURO`, texto branco.
- Título: **"Preditor de Risco de Desmatamento"** (Pt 40, bold, branco).
- Subtítulo: **"Amazônia Legal · Machine Learning aplicado à conservação"** (Pt 20, verde-claro).
- Rodapé: **"Amazônia Legal · 772 municípios · 2008–2025"** (Pt 14, SLATE_500).
- Ícone/emoji 🌿 como elemento visual (inserir como text box grande, Pt 80).

### Slide 2 — Problema & Objetivo
- Layout: fundo branco, barra lateral esquerda `VERDE_ESCURO` (0.4" largura).
- Título: "O Problema" (Pt 28, bold, SLATE_900).
- Bullet points (Pt 16):
  - 🌳 Amazônia Legal: 5,2 milhões km² de floresta tropical
  - 📊 Desmatamento histórico: pico em 2004, redução até 2012, reaceleração após 2018
  - ⚠️ Necessidade: identificar municípios de alto risco **antes** do desmatamento
- Box destacado (fundo VERDE_CLARO): "**Objetivo:** Score de risco 0–100 por município, atualizável anualmente"

### Slide 3 — Fontes de Dados
- Título: "Pipeline de Dados" (Pt 28, bold).
- Tabela 5 linhas × 3 colunas: Fonte | Dados | Cobertura
  - INPE/PRODES | Desmatamento km² | 2008–2025
  - IBGE | Pop., PIB agro, área | 2008–2025
  - ICMBio | Unidades de conservação e TIs | Estático
  - IBAMA | Autos de infração | 2008–2025
  - TerraBrasilis WFS | Geometrias municipais | Atual
- Estilo de tabela: cabeçalho `VERDE_ESCURO` (texto branco), linhas alternadas.

### Slide 4 — Feature Engineering
- Título: "Features do Modelo" (Pt 28, bold).
- Dois grupos de bullets lado a lado:
  - **Temporais** (lag features):
    - taxa_desmat_lag1/2/3: histórico de desmatamento
    - taxa_desmat_roll3/5: média móvel 3 e 5 anos
    - std_desmat_roll3: volatilidade do desmatamento
  - **Espaciais**:
    - local_moran_i: autocorrelação espacial (Queen weights)
    - area_uc_pct / area_ti_pct: proteção territorial
    - autos_ibama: fiscalização ambiental
- Nota: "Data leakage prevention: shift(1) antes do rolling window"

### Slide 5 — Clusterização HDBSCAN
- Título: "Perfis de Municípios (HDBSCAN)" (Pt 28, bold).
- 4 cards lado a lado (um por cluster):
  - Cluster 0: "Baixo risco estável" — fundo VERDE_CLARO
  - Cluster 1: "Risco moderado crescente" — fundo amarelo claro
  - Cluster 2: "Alto risco — fronteira agrícola" — fundo laranja claro
  - Cluster 3: "Hotspot crítico" — fundo vermelho claro
- Cada card: nome, contagem de municípios representativos.
- Nota: "89 municípios classificados como ruído (12%)"

### Slide 6 — Modelo & Validação
- Título: "LightGBM + Optuna" (Pt 28, bold).
- Métricas em destaque (3 KPI boxes):
  - **R² = 0.76** (média spatial CV)
  - **RMSE = 0.109** (%/ano)
  - **55 trials** Optuna
- Tabela de métricas por fold (UF | RMSE | R²):
  - MA: 0.114 | 0.748
  - PA: 0.118 | 0.850
  - MT: 0.114 | 0.793
  - RR+TO: 0.065 | 0.548
  - AC+AM+AP+RO: 0.132 | 0.853
- Nota: "Validação espacial por UF — evita data leakage geográfico"

### Slide 7 — Análise SHAP
- Título: "Interpretabilidade — SHAP" (Pt 28, bold).
- Inserir imagem `reports/shap_beeswarm.png` (se existir) ocupando 60% do slide.
- Bullets de insights (Pt 14):
  - "taxa_desmat_lag1 é o preditor mais importante em todos os clusters"
  - "local_moran_i revela efeito de contágio regional"
  - "area_ti_pct tem efeito protetor consistente (SHAP negativo)"

### Slide 8 — Top Municípios de Risco
- Título: "Municípios de Maior Risco — 2025" (Pt 28, bold).
- Tabela top-10: Município | UF | Score | Taxa (%/ano)
  - Dados reais do predictions.parquet (último ano).
- Barra de cor no score (verde→vermelho).
- Nota: "Score = ranking percentílico global (0–100)"

### Slide 9 — Dashboard
- Título: "Dashboard Interativo" (Pt 28, bold).
- Screenshot ou mockup do dashboard (inserir `reports/dashboard_preview.png` se existir).
- Bullets:
  - "Streamlit: análise exploratória e filtros interativos"
  - "HTML/Leaflet: distribuição pública sem dependências"
  - "Atualização anual: re-executar pipeline em ~15 min"

### Slide 10 — Próximos Passos & Contato
- Título: "Próximos Passos" (Pt 28, bold).
- Bullets:
  - "Integração com alertas em tempo real (DETER/INPE)"
  - "Modelo por bioma (Cerrado, Caatinga)"
  - "API REST para consumo por órgãos ambientais"
- Box de contato no rodapé: nome do autor, e-mail, GitHub.

---

## Configuração do slide master

```python
prs = Presentation()
prs.slide_width  = Inches(13.33)   # widescreen 16:9
prs.slide_height = Inches(7.5)
```

Usar layouts em branco (`prs.slide_layouts[6]`) e construir todos os elementos
manualmente para máximo controle visual.

---

## Helpers obrigatórios

```python
def add_text_box(slide, text, left, top, width, height,
                 font_size=18, bold=False, color=SLATE_900,
                 align=PP_ALIGN.LEFT, bg_color=None):
    """Adiciona text box estilizado ao slide."""

def add_kpi_box(slide, label, value, left, top, width=2.5, height=1.4,
                value_color=VERDE_ESCURO, bg_color=SLATE_100):
    """Adiciona card KPI com label e valor em destaque."""

def add_table(slide, headers, rows, left, top, width, height,
              header_bg=VERDE_ESCURO, header_fg=BRANCO):
    """Adiciona tabela estilizada com cabeçalho colorido."""
```

---

## Arquivo de saída

`reports/slide_deck_executivo.pptx`

---

## Requisitos de qualidade

- [ ] Abre sem erros no PowerPoint e no LibreOffice Impress.
- [ ] Todos os 10 slides presentes e com conteúdo correto.
- [ ] Dados reais do projeto (não placeholders genéricos).
- [ ] Imagens inseridas quando os arquivos existirem em `reports/`.
- [ ] Tamanho widescreen 16:9 (13.33" × 7.5").
- [ ] Paleta de cores consistente em todos os slides.
