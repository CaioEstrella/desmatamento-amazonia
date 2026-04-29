# Tasks — Preditor de Risco de Desmatamento (Amazônia Legal)

> Cada task deve ser executada na ordem listada. Nunca avançar sem validar o output da task anterior.
> Convenção de commit: `feat(task-XX): <descrição curta>`

---

## TASK-01 — Coletar dados INPE/PRODES

**Módulo:** `src/ingestion/inpe.py`

**Descrição:**
Criar o script de ingestão que usa as tools MCP do mcp-brasil para coletar os alertas históricos de desmatamento do PRODES por município e ano. Implementar cache local e retry logic.

**Inputs:**
- Tools MCP disponíveis via servidor `mcp-brasil` (configurado em `.claude/mcp.json`)

**Outputs:**
- `data/raw/inpe_raw.parquet`
  - Colunas: `cod_ibge` (str), `municipio` (str), `uf` (str), `ano` (int), `desmatamento_km2` (float)
  - Cobertura esperada: municípios da Amazônia Legal, anos 2000–atual

**Critério de conclusão:**
- Arquivo `data/raw/inpe_raw.parquet` existe e pode ser lido com `pd.read_parquet()`
- Contém dados de pelo menos 700 municípios e pelo menos 10 anos distintos
- Sem valores negativos em `desmatamento_km2`
- Script com docstring explicando parâmetros e comportamento de cache

**Dependências:** nenhuma

**Bloqueios conhecidos:**
- Se a tool MCP de desmatamento não estiver disponível, documentar aqui e aguardar orientação antes de prosseguir

---

## TASK-02 — Coletar dados IBGE socioeconômico

**Módulo:** `src/ingestion/ibge.py`

**Descrição:**
Coletar dados socioeconômicos municipais (PIB agropecuário, população, área territorial) via mcp-brasil.

**Inputs:**
- Tools MCP do mcp-brasil (IBGE)

**Outputs:**
- `data/raw/ibge_raw.parquet`
  - Colunas: `cod_ibge` (str), `municipio` (str), `uf` (str), `ano` (int), `populacao` (float), `pib_agropecuario` (float), `area_km2` (float)

**Critério de conclusão:**
- Arquivo existe e é legível
- Cobertura: todos os municípios da Amazônia Legal com pelo menos população e área preenchidos
- `area_km2` sem zeros (dado estrutural do município, não muda por ano)

**Dependências:** nenhuma (pode rodar em paralelo com TASK-01)

---

## TASK-03 — Coletar shapefile dos municípios

**Módulo:** `src/ingestion/ibge_geo.py`

**Descrição:**
Baixar o shapefile de municípios do Brasil via `geopandas` (fonte: IBGE). Filtrar apenas municípios da Amazônia Legal. Salvar em formato GeoPackage.

**Inputs:**
- URL pública do IBGE para shapefiles municipais (via `geopandas.read_file()`)
- Ou via tool MCP do mcp-brasil se disponível

**Outputs:**
- `data/raw/municipios.gpkg`
  - Colunas: `cod_ibge` (str), `municipio` (str), `uf` (str), `geometry` (Polygon)
  - Apenas municípios da Amazônia Legal

**Critério de conclusão:**
- Arquivo GeoPackage existe e pode ser lido com `gpd.read_file()`
- CRS definido (EPSG:4326 ou SIRGAS 2000)
- Contém geometrias válidas (sem nulos em `geometry`)
- Plot visual do mapa confirma contorno correto da Amazônia Legal

**Dependências:** nenhuma (pode rodar em paralelo com TASK-01 e TASK-02)

---

## TASK-04 — Coletar dados ICMBio (UCs e TIs)

**Módulo:** `src/ingestion/icmbio.py`

**Descrição:**
Coletar dados sobre unidades de conservação (UCs) e terras indígenas (TIs) por município via mcp-brasil. Calcular área sobreposta como percentual da área total do município.

**Inputs:**
- Tools MCP do mcp-brasil (ICMBio / FUNAI)
- `data/raw/ibge_raw.parquet` (para obter `area_km2` por município)

**Outputs:**
- `data/raw/icmbio_raw.parquet`
  - Colunas: `cod_ibge` (str), `area_uc_km2` (float), `area_ti_km2` (float)

**Critério de conclusão:**
- Arquivo existe com cobertura de pelo menos 80% dos municípios da Amazônia Legal
- Valores ≥ 0; pode haver municípios com 0 (sem UC/TI)

**Dependências:** TASK-02 (para ter `area_km2`)

---

## TASK-05 — Coletar autos de infração IBAMA

**Módulo:** `src/ingestion/ibama.py`

**Descrição:**
Coletar quantidade de autos de infração ambientais emitidos pelo IBAMA por município e ano via Portal da Transparência / mcp-brasil.

**Inputs:**
- Tools MCP do mcp-brasil (Portal da Transparência / IBAMA)
- Chave da API `TRANSPARENCIA_API_KEY` (opcional; cadastrar em api.portaldatransparencia.gov.br se necessário)

**Outputs:**
- `data/raw/ibama_raw.parquet`
  - Colunas: `cod_ibge` (str), `municipio` (str), `uf` (str), `ano` (int), `autos_ibama` (int)

**Critério de conclusão:**
- Arquivo existe e é legível
- Cobre pelo menos os anos 2005–atual
- Se API indisponível: documentar o bloqueio em `specs/tasks.md` e criar coluna `autos_ibama` zerada como fallback; anotar a limitação no README

**Dependências:** nenhuma

---

## TASK-06 — ETL — Consolidar fontes

**Módulo:** `src/features/etl.py`

**Descrição:**
Consolidar as cinco fontes brutas num único DataFrame `municipio × ano`. Tratar missing values, normalizar chaves, calcular colunas derivadas e filtrar apenas Amazônia Legal.

**Inputs:**
- `data/raw/inpe_raw.parquet`
- `data/raw/ibge_raw.parquet`
- `data/raw/ibge_raw.parquet` (para `area_km2`)
- `data/raw/icmbio_raw.parquet`
- `data/raw/ibama_raw.parquet`

**Outputs:**
- `data/processed/dataset.parquet`
  - Schema conforme seção 8 do `specs/design.md` (sem lag features, sem cluster_id, sem score_risco ainda)
  - Colunas obrigatórias nesta etapa: todas exceto `taxa_desmat_lag*`, `taxa_desmat_roll*`, `std_desmat_roll3`, `local_moran_i`, `cluster_id`, `score_risco`

**Critério de conclusão:**
- Arquivo existe, tem `len(df) > 5000` linhas (≥ 700 municípios × ≥ 7 anos)
- `df.isnull().mean() < 0.05` para todas as colunas obrigatórias
- `df["taxa_desmatamento"].min() >= 0`
- Teste de sanidade: municípios conhecidos (ex: Altamira-PA, São Félix do Xingu-PA) presentes

**Dependências:** TASK-01, TASK-02, TASK-03, TASK-04, TASK-05

---

## TASK-07 — Feature engineering — Lag e rolling window

**Módulo:** `src/features/engineering.py` (função `add_temporal_features`)

**Descrição:**
Adicionar lag features (1, 2, 3 anos) e rolling window (média 3 e 5 anos, desvio padrão 3 anos) da taxa de desmatamento. Usar exclusivamente pandas `.shift()` e `.rolling()`.

**Inputs:**
- `data/processed/dataset.parquet`

**Outputs:**
- `data/processed/dataset.parquet` atualizado com colunas:
  - `taxa_desmat_lag1`, `taxa_desmat_lag2`, `taxa_desmat_lag3`
  - `taxa_desmat_roll3`, `taxa_desmat_roll5`, `std_desmat_roll3`

**Critério de conclusão:**
- Colunas existem no parquet
- Primeiros anos de cada município têm NaN nos lags (comportamento correto do shift)
- Sem data leakage: verificar que `taxa_desmat_lag1` de 2010 = `taxa_desmatamento` de 2009 para o mesmo município

**Dependências:** TASK-06

---

## TASK-08 — Feature engineering — Moran's I local

**Módulo:** `src/features/engineering.py` (função `add_local_moran`)

**Descrição:**
Calcular o Local Moran's I da taxa de desmatamento para cada município em cada ano, usando `libpysal` (pesos Queen) e `esda.Moran_Local`. Adicionar como coluna `local_moran_i`.

**Inputs:**
- `data/processed/dataset.parquet` (com `taxa_desmatamento`)
- `data/raw/municipios.gpkg` (geometria para calcular pesos de vizinhança)

**Outputs:**
- `data/processed/dataset.parquet` atualizado com coluna `local_moran_i` (float)

**Critério de conclusão:**
- Coluna `local_moran_i` existe, sem todos os valores iguais a zero
- Municípios conhecidos de alto desmatamento com vizinhos de alto desmatamento (ex: sul do Pará) têm valor positivo alto
- Moran scatterplot gerado e salvo em `reports/moran_scatterplot.png`

**Dependências:** TASK-07

---

## TASK-09 — EDA — Notebook exploratório

**Módulo:** `notebooks/eda.ipynb`

**Descrição:**
Produzir notebook Jupyter com análise exploratória completa do dataset consolidado.

**Inputs:**
- `data/processed/dataset.parquet`

**Outputs:**
- `notebooks/eda.ipynb` executado com todas as células sem erros
- Visualizações salvas em `reports/`:
  - `reports/desmatamento_por_estado.png`
  - `reports/desmatamento_serie_temporal.png`
  - `reports/moran_scatterplot.png`
  - `reports/correlacao_features.png`
  - `reports/mapa_desmatamento.html`

**Critério de conclusão:**
- Notebook executa do início ao fim sem erros (`Kernel > Restart & Run All`)
- Todos os gráficos gerados têm títulos, eixos nomeados e são interpretáveis

**Dependências:** TASK-08

---

## TASK-10 — Clusterização HDBSCAN

**Módulo:** `src/clustering/hdbscan_model.py`

**Descrição:**
Segmentar municípios em perfis usando HDBSCAN sobre features estruturais (sem lag temporal). Adicionar `cluster_id` ao dataset. Documentar interpretação de cada cluster.

**Inputs:**
- `data/processed/dataset.parquet`

**Outputs:**
- `data/processed/dataset.parquet` atualizado com coluna `cluster_id`
- `reports/clusters_mapa.html` — mapa Folium com cores por cluster
- `reports/clusters_description.md` — descrição interpretativa de cada cluster

**Critério de conclusão:**
- Entre 4 e 6 clusters distintos (excluindo ruído `-1`)
- Silhouette score ≥ 0,20 (excluindo ruído) — acima de 0,30 é raro com HDBSCAN
  sobre dados geoespaciais contínuos; 0,20 é o limiar realista para municípios brasileiros
- Municípios do sul do Pará e norte do MT em clusters de "alta pressão"
- Descrição textual de cada cluster escrita e plausível

**Dependências:** TASK-09 (EDA informa a escolha de features para clustering)

---

## TASK-11 — Tunagem de hiperparâmetros LightGBM com Optuna

**Módulo:** `src/model/train.py`

**Descrição:**
Executar estudo Optuna com ≥ 50 trials, usando spatial cross-validation (UF como grupo) como função objetivo. Salvar o estudo e os melhores hiperparâmetros.

**Inputs:**
- `data/processed/dataset.parquet` (com todas as features, incluindo `cluster_id`)

**Outputs:**
- `data/outputs/optuna_study.pkl` — estudo Optuna completo
- `data/outputs/best_params.json` — melhores hiperparâmetros encontrados

**Critério de conclusão:**
- `optuna_study.pkl` existe e pode ser carregado com `joblib.load()`
- `study.best_trial.value` (RMSE) < RMSE de um modelo com parâmetros padrão (baseline)
- `best_params.json` contém todos os 9 hiperparâmetros tunados

**Dependências:** TASK-10

---

## TASK-12 — Treinar modelo LightGBM final

**Módulo:** `src/model/train.py`

**Descrição:**
Treinar o modelo LightGBM final com os melhores hiperparâmetros encontrados pelo Optuna. Executar spatial CV para medir métricas finais por fold.

**Inputs:**
- `data/processed/dataset.parquet`
- `data/outputs/best_params.json`

**Outputs:**
- `data/outputs/lgbm_model.pkl` — modelo treinado
- `data/outputs/metrics_by_fold.json` — RMSE, MAE, R² por fold espacial e média
- Tabela de métricas exibida no console

**Critério de conclusão:**
- R² médio nos folds ≥ 0,65
- Modelo carrega sem erros com `joblib.load()`
- Métricas reportadas por fold confirmam ausência de vazamento espacial

**Dependências:** TASK-11

---

## TASK-13 — SHAP — Explicabilidade

**Módulo:** `src/model/evaluate.py`

**Descrição:**
Gerar análise de explicabilidade com SHAP: importância global das features e plots individuais para municípios de maior risco.

**Inputs:**
- `data/outputs/lgbm_model.pkl`
- `data/processed/dataset.parquet`

**Outputs:**
- `reports/shap_beeswarm.png` — importância global
- `reports/shap_waterfall_top5.png` — waterfall dos 5 municípios de maior risco
- `reports/shap_by_cluster.png` — importância por cluster

**Critério de conclusão:**
- Top-5 features plausíveis segundo literatura (ex: taxa histórica, PIB agropecuário, distância de UC)
- Plots gerados sem erros e visualmente legíveis

**Dependências:** TASK-12

---

## TASK-14 — Normalizar score 0–100 e salvar predições finais

**Módulo:** `src/model/predict.py`

**Descrição:**
Gerar predições para todos os municípios/anos disponíveis, normalizar para o intervalo 0–100 por percentil e salvar.

**Inputs:**
- `data/outputs/lgbm_model.pkl`
- `data/processed/dataset.parquet`

**Outputs:**
- `data/outputs/predictions.parquet`
  - Colunas: `cod_ibge`, `municipio`, `uf`, `ano`, `score_risco` (float 0–100)
- `data/processed/dataset.parquet` atualizado com coluna `score_risco`

**Critério de conclusão:**
- `score_risco` está no intervalo [0, 100] para todos os registros
- Municípios historicamente conhecidos de alto desmatamento (São Félix do Xingu, Altamira, Novo Progresso) estão no decil superior
- Distribuição do score é razoavelmente uniforme (histograma gerado)

**Dependências:** TASK-13

---

## TASK-15 — Dashboard Streamlit

**Módulo:** `src/dashboard/app.py`

**Descrição:**
Implementar dashboard Streamlit com mapa coroplético interativo, filtros, ranking e gráfico SHAP.

**Inputs:**
- `data/outputs/predictions.parquet`
- `data/raw/municipios.gpkg`
- `reports/shap_beeswarm.png`

**Outputs:**
- `src/dashboard/app.py` — aplicação Streamlit executável

**Critério de conclusão:**
- `streamlit run src/dashboard/app.py` inicia sem erros
- Mapa Folium renderiza com escala de cores corretas
- Filtros por UF e cluster funcionam
- Ranking top-20 atualiza ao mudar filtros
- Testado localmente no browser

**Dependências:** TASK-14

---

## TASK-16 — Dashboard HTML/React de portfólio [pós-implementação]

**Ferramenta:** frontend-design skill do Claude Code

**Descrição:**
Usar a skill `frontend-design` para gerar uma versão HTML/React estática do dashboard com visual de portfólio profissional.

**Inputs:**
- `data/outputs/predictions.parquet`
- `reports/shap_beeswarm.png`
- Layout e conteúdo do dashboard Streamlit (TASK-15)

**Outputs:**
- `reports/portfolio_dashboard/index.html` — dashboard HTML estático

**Critério de conclusão:**
- Abre no browser sem servidor
- Visual de portfólio profissional

**Dependências:** TASK-15

---

## TASK-17 — Slide deck executivo [pós-implementação]

**Ferramenta:** pptx skill do Claude Code

**Descrição:**
Usar a skill `pptx` para gerar apresentação executiva com os resultados do projeto.

**Inputs:**
- `reports/clusters_mapa.html`
- `reports/shap_beeswarm.png`
- `data/outputs/metrics_by_fold.json`
- `reports/clusters_description.md`

**Outputs:**
- `reports/apresentacao_resultados.pptx`
  - 7 slides conforme estrutura definida em `specs/design.md` seção 7.3

**Critério de conclusão:**
- Arquivo `.pptx` abre sem erros no PowerPoint
- Todos os 7 slides presentes com conteúdo

**Dependências:** TASK-15

---

## Resumo de dependências

```
TASK-01 ─────────────────────────────┐
TASK-02 ──────────────┐              │
TASK-03 ──────────────┤              │
                      ▼              ▼
TASK-04 (deps: 02) → TASK-06 → TASK-07 → TASK-08 → TASK-09 → TASK-10
TASK-05 ─────────────┘
                                                              │
                                                              ▼
                                                          TASK-11
                                                              │
                                                          TASK-12
                                                              │
                                                          TASK-13
                                                              │
                                                          TASK-14
                                                              │
                                                          TASK-15
                                                         /       \
                                                    TASK-16   TASK-17
```
