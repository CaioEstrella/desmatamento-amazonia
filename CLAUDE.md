# CLAUDE.md — Preditor de Risco de Desmatamento (Amazônia Legal)

## Contexto do projeto

**Objetivo:** Atribuir um score de risco de desmatamento (0–100) para cada município da Amazônia Legal, atualizável periodicamente, usando dados históricos e socioeconômicos públicos.

**Audiência:** Portfólio profissional de Data Science — o código deve ser legível, bem comentado e didático. Prefira clareza a otimização sempre que houver trade-off.

**Stack principal:** Python 3.12 · LightGBM · Optuna · SHAP · GeoPandas · libpysal/esda · HDBSCAN · Streamlit · mcp-brasil

**Escopo geográfico:** Amazônia Legal — 9 estados: AM, PA, MT, RO, AC, AP, RR, TO e MA (parcial).

---

## Regras obrigatórias

### Ambiente
- **Sempre use o `.venv` do projeto.** Nunca instale pacotes no Python global.
  - Windows: `.venv\Scripts\activate`
  - macOS/Linux: `source .venv/bin/activate`

### Feature engineering
- Lag features e rolling window devem ser implementados via **pandas puro**: `.shift()` e `.rolling()`. Não use bibliotecas externas para isso.

### Modelagem
- **Modelo único: LightGBM.** Não compare com outros algoritmos (XGBoost, Random Forest, etc.). O objetivo é demonstrar profundidade, não breadth.

### Tunagem de hiperparâmetros
- A tunagem é **obrigatória via Optuna**.
- A função objetivo do Optuna deve usar **spatial cross-validation** (UF como grupo via `spatial-kfold`) — nunca split aleatório.
- Mínimo de **50 trials** por estudo.
- Salvar o estudo Optuna em `data/outputs/optuna_study.pkl` para reproductibilidade.
- Salvar os melhores hiperparâmetros em `data/outputs/best_params.json`.

### Commits
- **Um commit por TASK concluída** (TASK-01, TASK-02, ...). Mensagens no formato: `feat(task-XX): <descrição curta>`.

### Fonte de dados
- O **mcp-brasil** é a única fonte de dados. Não faça scraping manual, não baixe arquivos CSV externos diretamente no código.
- O `mcp-brasil` é um **servidor MCP externo** — não está no `requirements.txt` nem no `.venv`. É um processo separado gerenciado pelo `uvx`, configurado em `.claude/mcp.json`.
- O código Python **nunca importa o mcp-brasil**. Os scripts de ingestão apenas leem os arquivos `.parquet` já gerados em `data/raw/` via tools MCP.
- Implemente cache local em `data/raw/` para não repetir chamadas às tools MCP desnecessariamente.

### Outputs visuais finais (pós-implementação)
- Dashboard HTML/React: usar a **frontend-design skill** do Claude Code.
- Slide deck executivo: usar a **pptx skill** do Claude Code.

---

## Estrutura do projeto

```
desmatamento-amazonia/
├── CLAUDE.md               ← este arquivo
├── README.md
├── requirements.txt
├── .gitignore
├── specs/
│   ├── requirements.md     ← requisitos funcionais e não funcionais
│   ├── design.md           ← arquitetura e design técnico
│   └── tasks.md            ← tasks atômicas com dependências
├── data/
│   ├── raw/                ← dados brutos do mcp-brasil (ignorado pelo git)
│   ├── processed/          ← dataset consolidado (ignorado pelo git)
│   └── outputs/            ← modelos, estudos Optuna, scores (ignorado pelo git)
├── notebooks/              ← EDA exploratória (TASK-09)
├── src/
│   ├── ingestion/          ← scripts de coleta via mcp-brasil (TASK-01 a TASK-05)
│   ├── features/           ← ETL e feature engineering (TASK-06 a TASK-08)
│   ├── clustering/         ← HDBSCAN (TASK-10)
│   ├── model/              ← LightGBM + Optuna + SHAP (TASK-11 a TASK-14)
│   └── dashboard/          ← Streamlit (TASK-15)
└── reports/                ← outputs visuais exportados
```

---

## Schema do dataset consolidado

O arquivo `data/processed/dataset.parquet` terá **uma linha por (município, ano)** com as colunas:

| Coluna | Tipo | Origem |
|---|---|---|
| `cod_ibge` | str | IBGE |
| `municipio` | str | IBGE |
| `uf` | str | IBGE |
| `ano` | int | — |
| `area_km2` | float | IBGE |
| `populacao` | float | IBGE |
| `pib_agropecuario` | float | IBGE |
| `desmatamento_km2` | float | INPE/PRODES |
| `taxa_desmatamento` | float | calculado |
| `area_uc_pct` | float | ICMBio |
| `area_ti_pct` | float | ICMBio/FUNAI |
| `autos_ibama` | int | IBAMA |
| `taxa_desmat_lag1` | float | calculado |
| `taxa_desmat_lag2` | float | calculado |
| `taxa_desmat_lag3` | float | calculado |
| `taxa_desmat_roll3` | float | calculado |
| `taxa_desmat_roll5` | float | calculado |
| `std_desmat_roll3` | float | calculado |
| `local_moran_i` | float | esda |
| `cluster_id` | int | HDBSCAN |
| `geometry` | geometry | IBGE shapefile |
| `score_risco` | float | modelo (0–100) |
