# Preditor de Risco de Desmatamento — Amazônia Legal

Pipeline de análise de dados e machine learning para predição de risco de desmatamento em municípios da Amazônia Legal brasileira, utilizando exclusivamente dados públicos governamentais via **mcp-brasil**.

---

## Dashboard interativo

**Acesse ao vivo, sem instalar nada:**

> ### [caioestrella.github.io/desmatamento-amazonia](https://caioestrella.github.io/desmatamento-amazonia/)

O dashboard é um arquivo HTML único, hospedado no GitHub Pages. Abre diretamente no navegador.

**O que você encontra:**
- **Mapa coroplético** interativo — score de risco 0–100 por município, filtrável por ano e estado
- **Ranking** — top-20 municípios de maior risco
- **Clusters HDBSCAN** — perfis estruturais dos municípios com mapa e cards descritivos
- **Série temporal** — evolução do desmatamento por estado (2008–2026)
- **SHAP** — importância global das features e waterfall dos municípios mais críticos
- **Dicionário** — definição de cada indicador e feature do modelo
- Suporte a **tema claro/escuro**

---

## Visão geral

O projeto atribui um **score de risco de desmatamento (0–100)** para cada um dos municípios da Amazônia Legal, combinando séries históricas do INPE/PRODES com dados socioeconômicos do IBGE, unidades de conservação do ICMBio e autos de infração do IBAMA. O modelo é explicável via SHAP e os resultados são apresentados em dashboard interativo com mapa de calor por município.

```
Dados públicos (mcp-brasil)
    └── INPE/PRODES · IBGE · ICMBio · IBAMA
          ↓
    ETL + Feature Engineering
    (lag features · rolling window · Moran's I local)
          ↓
    Clusterização (HDBSCAN)
    → perfis de municípios
          ↓
    LightGBM + Optuna
    (tunagem com spatial cross-validation)
          ↓
    Score 0–100 por município
    + explicabilidade SHAP
          ↓
    Dashboard HTML · Slide deck executivo
```

---

## Arquitetura do Pipeline

![Arquitetura do Pipeline](reports/arquitetura.png)

Para regenerar o diagrama:

```bash
python src/reports/build_arquitetura.py
```

---

## Stack

| Camada | Tecnologias |
|---|---|
| Dados | `mcp-brasil`, `pandas`, `geopandas` |
| Análise espacial | `libpysal`, `esda`, `splot` |
| Feature engineering | `feature-engine`, `pandas` |
| Clusterização | `hdbscan`, `scikit-learn` |
| Modelagem | `lightgbm`, `optuna` |
| Validação | `spatial-kfold` |
| Explicabilidade | `shap` |
| Visualização | `folium`, `plotly`, `Chart.js`, `Leaflet.js` |
| Dashboard | HTML single-file · GitHub Pages |

---

## Fontes de dados

Todos os dados são obtidos via **[mcp-brasil](https://github.com/Mcp-Brasil/mcp-brasil)**, um MCP server open-source que conecta agentes de IA a APIs públicas brasileiras.

| Fonte | Dados utilizados |
|---|---|
| **INPE / PRODES** | Alertas históricos de desmatamento por município |
| **IBGE** | Dados socioeconômicos municipais (PIB agropecuário, população, área) |
| **ICMBio** | Unidades de conservação e terras indígenas |
| **Portal da Transparência / IBAMA** | Autos de infração ambientais por município |

---

## Estrutura do repositório

```
desmatamento-amazonia/
├── CLAUDE.md               # Instruções para o Claude Code
├── index.html              # Redirect para o dashboard (GitHub Pages)
├── specs/
│   ├── requirements.md
│   ├── design.md
│   └── tasks.md
├── data/
│   ├── raw/                # Dados brutos via mcp-brasil (gitignore)
│   ├── processed/          # Dataset consolidado municipio×ano (gitignore)
│   └── outputs/            # Predições, métricas, SHAP (gitignore)
├── notebooks/              # EDA exploratória
├── src/
│   ├── ingestion/          # Coleta de dados via mcp-brasil
│   ├── features/           # ETL, lag features, Moran's I
│   ├── clustering/         # HDBSCAN e perfis de municípios
│   ├── model/              # LightGBM, Optuna, SHAP
│   ├── dashboard/          # build_html.py — gerador do dashboard
│   └── reports/            # build_pptx.py, build_arquitetura.py
└── reports/
    ├── dashboard.html          # Dashboard interativo (publicado no GitHub Pages)
    ├── arquitetura.png         # Diagrama do pipeline
    ├── clusters_description.md # Perfis textuais dos clusters HDBSCAN
    └── slide_deck_executivo.pptx
```

---

## Como executar o pipeline

### Pré-requisitos

- Python 3.10+
- Chave gratuita do Portal da Transparência (opcional): [api.portaldatransparencia.gov.br](https://api.portaldatransparencia.gov.br)

### Instalação

```bash
git clone https://github.com/CaioEstrella/desmatamento-amazonia.git
cd desmatamento-amazonia
pip install -r requirements.txt
```

### Configuração do mcp-brasil

Crie o arquivo `.claude/mcp.json`:

```json
{
  "mcpServers": {
    "mcp-brasil": {
      "command": "uvx",
      "args": ["--from", "mcp-brasil", "python", "-m", "mcp_brasil.server"],
      "env": {
        "TRANSPARENCIA_API_KEY": "sua-chave-aqui"
      }
    }
  }
}
```

### Executando o pipeline

```bash
# Coleta de dados
python src/ingestion/inpe.py
python src/ingestion/ibge.py
python src/ingestion/icmbio.py
python src/ingestion/ibama.py

# ETL e feature engineering
python src/features/etl.py
python src/features/engineering.py

# Clusterização
python src/clustering/hdbscan_profiles.py

# Modelagem (tunagem + treino final)
python src/model/tune.py      # Optuna — pode levar alguns minutos
python src/model/train.py
python src/model/evaluate.py  # SHAP

# Gerar dashboard HTML
python -m src.dashboard.build_html
```

### Outputs adicionais

```bash
# Slide deck executivo (PowerPoint)
python -m src.reports.build_pptx

# Diagrama de arquitetura
python src/reports/build_arquitetura.py
```

---

## Slide Deck Executivo

Apresentação PowerPoint com 10 slides widescreen (16:9) cobrindo contexto, metodologia, resultados do modelo, mapa de risco e recomendações. Gerada automaticamente a partir dos dados de predição.

Arquivo: `reports/slide_deck_executivo.pptx`

---

## Metodologia: por que spatial cross-validation?

Dados geoespaciais violam a suposição de independência entre observações — municípios vizinhos têm desmatamento correlacionado. Um split aleatório tradicional colocaria municípios vizinhos simultaneamente no treino e no teste, inflando artificialmente as métricas. Este projeto usa `spatial-kfold` com UF como grupo, garantindo que cada fold de validação contenha estados geograficamente separados do conjunto de treino.

O mesmo princípio se aplica à tunagem: a **função objetivo do Optuna usa a média do RMSE nos folds espaciais**, não um split aleatório.

---

## Desenvolvimento

Este projeto foi desenvolvido com **Claude Code** seguindo o padrão **Spec Driven Development (SDD)**: especificação completa em `specs/` antes de qualquer implementação, commits atômicos por tarefa e validação de outputs a cada etapa.

---

## Licença

MIT
