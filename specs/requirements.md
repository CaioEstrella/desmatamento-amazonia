# Requisitos — Preditor de Risco de Desmatamento (Amazônia Legal)

## 1. Objetivo

Construir um modelo de machine learning que atribui um **score de risco de desmatamento (0–100)** para cada município da Amazônia Legal, utilizável como portfólio profissional de Data Science. O score deve ser interpretável, reprodutível e atualizável periodicamente conforme novos dados forem disponibilizados.

---

## 2. Escopo geográfico

- **Região:** Amazônia Legal
- **Estados cobertos:** Amazonas (AM), Pará (PA), Mato Grosso (MT), Rondônia (RO), Acre (AC), Amapá (AP), Roraima (RR), Tocantins (TO) e Maranhão (MA — apenas municípios que integram a Amazônia Legal)
- **Unidade de análise:** Município × Ano (painel longitudinal)
- **Janela temporal:** dados históricos do PRODES/INPE (disponível desde 1988; usar a partir de 2000 para garantir cobertura ampla)

---

## 3. Requisitos funcionais

### RF-01 — Ingestão de dados
O sistema deve coletar dados de quatro fontes via tools MCP do mcp-brasil:

| Fonte | Dados coletados | Arquivo de saída |
|---|---|---|
| INPE/PRODES | Alertas históricos de desmatamento por município e ano (área em km²) | `data/raw/inpe_raw.parquet` |
| IBGE | Dados socioeconômicos municipais: PIB agropecuário, população, área territorial | `data/raw/ibge_raw.parquet` |
| ICMBio | Unidades de conservação (UCs) e terras indígenas (TIs) por município | `data/raw/icmbio_raw.parquet` |
| IBAMA / Portal Transparência | Autos de infração ambientais por município e ano | `data/raw/ibama_raw.parquet` |

Além disso, o shapefile de municípios do Brasil deve ser obtido via `geopandas` (IBGE) e salvo em `data/raw/municipios.gpkg`.

### RF-02 — Dataset consolidado
O sistema deve produzir um DataFrame `municipio × ano` em `data/processed/dataset.parquet` com todas as features necessárias para modelagem (ver schema completo no CLAUDE.md e em `specs/design.md`).

### RF-03 — Análise exploratória
O sistema deve produzir um notebook Jupyter em `notebooks/eda.ipynb` com:
- Distribuição histórica do desmatamento por estado e ano
- Mapa coroplético do desmatamento acumulado
- Moran scatterplot (Moran's I global) da taxa de desmatamento
- Histograma e boxplot das features numéricas
- Matriz de correlação

### RF-04 — Clusterização de municípios
O sistema deve segmentar os municípios em 4–6 perfis interpretáveis usando HDBSCAN, gerando:
- Coluna `cluster_id` no dataset
- Visualização dos clusters no mapa
- Descrição textual de cada perfil (ex: "expansão de fronteira agrícola", "pressão sobre UC")

### RF-05 — Modelo preditivo
O sistema deve treinar um modelo LightGBM com:
- Tunagem de hiperparâmetros via Optuna (≥ 50 trials, função objetivo com spatial CV)
- Spatial cross-validation usando UF como grupo (`spatial-kfold`)
- Métricas reportadas por fold: RMSE, MAE, R²
- Score normalizado 0–100 por percentil

### RF-06 — Explicabilidade
O sistema deve gerar, via SHAP:
- Gráfico de importância global das features (beeswarm plot)
- Waterfall plot para os 5 municípios de maior risco
- Summary por cluster

### RF-07 — Dashboard Streamlit
O sistema deve disponibilizar um dashboard interativo com:
- Mapa Folium de calor com score por município
- Filtros por estado e cluster
- Ranking top-20 municípios de maior risco
- Gráfico SHAP interativo

### RF-08 — Outputs de portfólio (pós-implementação)
- Dashboard HTML/React gerado via **frontend-design skill** do Claude Code
- Slide deck executivo gerado via **pptx skill** do Claude Code (clusters, mapa de risco, top features SHAP, limitações)

---

## 4. Requisitos não funcionais

### RNF-01 — Dados públicos
Apenas fontes de dados públicas e gratuitas. Nenhum dado pago ou proprietário.

### RNF-02 — Reprodutibilidade
- Ambiente reprodutível via `requirements.txt` e `.venv`
- Estudo Optuna salvo em `data/outputs/optuna_study.pkl`
- Seed fixo em todas as operações aleatórias (`random_state=42`)

### RNF-03 — Plataforma
- Python 3.10+
- Compatível com Windows 11
- Sem dependências de sistema operacional específico além do Python

### RNF-04 — Legibilidade
- Código comentado e legível (projeto de portfólio)
- Prefira clareza a otimização em todo trade-off
- Docstrings em todas as funções públicas

### RNF-05 — Modularidade
- Uma função por fonte de dados no módulo de ingestão
- Separação clara entre ingestão, ETL, feature engineering, modelagem e visualização

### RNF-06 — Cache e resiliência
- Dados brutos cacheados em `data/raw/` para evitar re-chamadas às APIs
- Retry logic nas chamadas às tools MCP

---

## 5. Fora do escopo

- Previsão em tempo real (o modelo é batch, atualizado periodicamente)
- Comparação entre algoritmos de ML (apenas LightGBM)
- Dados pagos ou restritos
- Deploy em produção (apenas execução local e portfólio)
- Cobertura de outros biomas fora da Amazônia Legal

---

## 6. Critérios de aceitação

| Entregável | Critério de aceitação |
|---|---|
| Dataset consolidado | ≥ 700 municípios, ≥ 10 anos de histórico, < 5% missing por coluna |
| Modelo | R² médio nos folds espaciais ≥ 0,65 |
| Score | Distribuição razoável 0–100; municípios conhecidos de alto risco no quartil superior |
| Dashboard | Carrega em < 10s localmente; mapa renderiza corretamente |
| SHAP | Top-5 features condizentes com literatura de desmatamento |
