# Design Técnico — Preditor de Risco de Desmatamento (Amazônia Legal)

## 1. Visão geral da arquitetura

```
tools MCP (mcp-brasil)
        │
        ▼
┌───────────────────┐
│  src/ingestion/   │  TASK-01 a TASK-05
│  (coleta de dados)│  → data/raw/*.parquet
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  src/features/    │  TASK-06 a TASK-08
│  etl.py           │  → data/processed/dataset.parquet
│  engineering.py   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  notebooks/       │  TASK-09
│  eda.ipynb        │  → relatórios EDA
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  src/clustering/  │  TASK-10
│  hdbscan_model.py │  → coluna cluster_id no dataset
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  src/model/       │  TASK-11 a TASK-14
│  train.py         │  → optuna_study.pkl
│  evaluate.py      │  → best_params.json
│  predict.py       │  → predictions.parquet
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  src/dashboard/   │  TASK-15
│  app.py           │  → Streamlit interativo
└───────────────────┘
```

---

## 2. Etapa 1 — Ingestão (`src/ingestion/`)

### Princípios
- Uma função por fonte de dados
- Cache local: se o arquivo `.parquet` já existir em `data/raw/`, pular a coleta
- Retry logic: até 3 tentativas com backoff exponencial em caso de falha das tools MCP

### Módulos

#### `src/ingestion/inpe.py`
```python
def collect_inpe(output_path: str = "data/raw/inpe_raw.parquet") -> pd.DataFrame:
    """Coleta alertas de desmatamento PRODES por município e ano via mcp-brasil."""
```
- **Input:** tools MCP (`get_prodes_desmatamento` ou equivalente)
- **Output:** `data/raw/inpe_raw.parquet`
  - Colunas: `cod_ibge`, `municipio`, `uf`, `ano`, `desmatamento_km2`

#### `src/ingestion/ibge.py`
```python
def collect_ibge(output_path: str = "data/raw/ibge_raw.parquet") -> pd.DataFrame:
    """Coleta dados socioeconômicos municipais (PIB agropecuário, população, área)."""
```
- **Output:** `data/raw/ibge_raw.parquet`
  - Colunas: `cod_ibge`, `municipio`, `uf`, `ano`, `populacao`, `pib_agropecuario`, `area_km2`

#### `src/ingestion/ibge_geo.py`
```python
def collect_shapefile(output_path: str = "data/raw/municipios.gpkg") -> gpd.GeoDataFrame:
    """Baixa shapefile dos municípios do Brasil via geopandas (IBGE)."""
```
- **Output:** `data/raw/municipios.gpkg`
  - Colunas: `cod_ibge`, `municipio`, `uf`, `geometry`

#### `src/ingestion/icmbio.py`
```python
def collect_icmbio(output_path: str = "data/raw/icmbio_raw.parquet") -> pd.DataFrame:
    """Coleta área de UCs e TIs por município via mcp-brasil."""
```
- **Output:** `data/raw/icmbio_raw.parquet`
  - Colunas: `cod_ibge`, `area_uc_km2`, `area_ti_km2`

#### `src/ingestion/ibama.py`
```python
def collect_ibama(output_path: str = "data/raw/ibama_raw.parquet") -> pd.DataFrame:
    """Coleta autos de infração IBAMA por município e ano via mcp-brasil."""
```
- **Output:** `data/raw/ibama_raw.parquet`
  - Colunas: `cod_ibge`, `municipio`, `uf`, `ano`, `autos_ibama`

---

## 3. Etapa 2 — ETL (`src/features/etl.py`)

### Função principal
```python
def build_dataset(raw_dir: str = "data/raw",
                  output_path: str = "data/processed/dataset.parquet") -> pd.DataFrame:
    """Consolida todas as fontes num único DataFrame municipio×ano."""
```

### Operações
1. **Carregar** cada `.parquet` de `data/raw/`
2. **Filtrar** apenas municípios da Amazônia Legal (9 UFs)
3. **Normalizar** nomes de municípios e código IBGE (chave de junção)
4. **Join** de todas as fontes por `(cod_ibge, ano)` — left join a partir do INPE/PRODES
5. **Calcular** `taxa_desmatamento = desmatamento_km2 / area_km2 * 100`
6. **Calcular** `area_uc_pct = area_uc_km2 / area_km2 * 100`
7. **Calcular** `area_ti_pct = area_ti_km2 / area_km2 * 100`
8. **Tratar missing values**: imputar 0 para `autos_ibama` ausente; interpolação linear para séries temporais
9. **Salvar** em `data/processed/dataset.parquet`

### Validações
- `cod_ibge` deve ter 7 dígitos
- `ano` entre 2000 e ano atual
- `taxa_desmatamento` deve ser ≥ 0
- Alerta se missing > 5% em qualquer coluna

---

## 4. Etapa 3 — Feature Engineering (`src/features/engineering.py`)

### Função principal
```python
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona lag features, rolling window e Moran's I local ao dataset."""
```

### 4.1 Lag features (pandas `.shift()`)
O DataFrame deve estar ordenado por `(cod_ibge, ano)` antes do shift:

```python
df = df.sort_values(["cod_ibge", "ano"])
df["taxa_desmat_lag1"] = df.groupby("cod_ibge")["taxa_desmatamento"].shift(1)
df["taxa_desmat_lag2"] = df.groupby("cod_ibge")["taxa_desmatamento"].shift(2)
df["taxa_desmat_lag3"] = df.groupby("cod_ibge")["taxa_desmatamento"].shift(3)
```

### 4.2 Rolling window (pandas `.rolling()`)
```python
roll = df.groupby("cod_ibge")["taxa_desmatamento"]
df["taxa_desmat_roll3"] = roll.transform(lambda x: x.shift(1).rolling(3).mean())
df["taxa_desmat_roll5"] = roll.transform(lambda x: x.shift(1).rolling(5).mean())
df["std_desmat_roll3"]  = roll.transform(lambda x: x.shift(1).rolling(3).std())
```
> O `.shift(1)` antes do rolling evita data leakage (não usa o valor do próprio ano).

### 4.3 Moran's I local (`libpysal` + `esda`)

```python
from libpysal.weights import Queen
from esda.moran import Moran_Local

def add_local_moran(df: pd.DataFrame, gdf: gpd.GeoDataFrame, year: int) -> pd.DataFrame:
    """Calcula o Local Moran's I da taxa de desmatamento para um dado ano."""
    subset = df[df["ano"] == year].merge(gdf[["cod_ibge", "geometry"]], on="cod_ibge")
    gdf_year = gpd.GeoDataFrame(subset, geometry="geometry")
    w = Queen.from_dataframe(gdf_year, silence_warnings=True)
    w.transform = "r"  # row-standardize
    moran_loc = Moran_Local(gdf_year["taxa_desmatamento"].fillna(0), w)
    return moran_loc.Is  # array de Is por município
```

- A coluna `local_moran_i` captura o efeito de vizinhança espacial — município com alto desmatamento cercado de outros com alto desmatamento tem valor positivo alto.
- Calcular para cada ano e fazer join ao dataset.

---

## 5. Etapa 4 — Clusterização (`src/clustering/`)

### Módulo: `src/clustering/hdbscan_model.py`

```python
def run_clustering(df: pd.DataFrame,
                   features: list[str],
                   output_path: str = "data/processed/dataset.parquet") -> pd.DataFrame:
    """Executa HDBSCAN nos municípios e adiciona cluster_id ao dataset."""
```

### Features de entrada (sem lag/rolling para evitar leakage temporal)
- `area_km2`, `populacao`, `pib_agropecuario`
- `taxa_desmatamento` (média histórica por município)
- `area_uc_pct`, `area_ti_pct`
- `autos_ibama` (total histórico por município)

> Agregar por município (média/soma sobre anos) antes de clusterizar — a clusterização é sobre perfil estrutural, não sobre o painel temporal.

### Parâmetros HDBSCAN
```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=15,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom"
)
```

### Validação
- Calcular silhouette score (excluindo ruído, `cluster_id == -1`)
- Plotar clusters no mapa (Folium ou GeoPandas)
- Documentar interpretação de cada cluster em `reports/clusters_description.md`

### Perfis esperados (hipóteses a validar)
1. **Expansão de fronteira agrícola** — alta área, alto PIB agropecuário, alto desmatamento crescente
2. **Pressão sobre UC/TI** — alto percentual de UC/TI, desmatamento pontual
3. **Baixo risco estrutural** — pequena área, baixo PIB agropecuário, desmatamento residual
4. **Desmatamento histórico consolidado** — alta taxa histórica mas estabilizada
5. **Municípios de fronteira ativa** — alto volume de autos IBAMA, desmatamento recente

---

## 6. Etapa 5 — Modelo preditivo (`src/model/`)

### 6.1 Definição do problema
- **Variável alvo:** `taxa_desmatamento` do ano corrente (regressão)
- **Features de entrada:** todas as colunas do dataset exceto `taxa_desmatamento`, `desmatamento_km2`, `score_risco` e `geometry`
- **Janela temporal:** anos a partir de 2003 (para ter pelo menos 3 anos de lag disponíveis)

### 6.2 Spatial Cross-Validation

```python
from spatial_kfold.folding import SpacialKFold

skf = SpacialKFold(n_splits=5, method="spatial_aware", space_col="uf")
```

- UF como grupo espacial — municípios da mesma UF ficam sempre no mesmo fold
- Isso evita vazamento espacial entre municípios vizinhos de estados diferentes
- 5 folds → treina em 4 UFs, valida na 5ª, rotacionando

### 6.3 Tunagem de hiperparâmetros (Optuna)

```python
import optuna
import lightgbm as lgb

def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
              folds: list) -> float:
    params = {
        "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "verbosity": -1,
    }
    rmse_scores = []
    for train_idx, val_idx in folds:
        model = lgb.LGBMRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        rmse_scores.append(mean_squared_error(y.iloc[val_idx], preds, squared=False))
    return float(np.mean(rmse_scores))

study = optuna.create_study(direction="minimize")
study.optimize(lambda t: objective(t, X, y, folds), n_trials=50)
joblib.dump(study, "data/outputs/optuna_study.pkl")
```

### 6.4 Treinamento final

```python
best_params = study.best_params
best_params["random_state"] = 42
best_params["verbosity"] = -1

model = lgb.LGBMRegressor(**best_params)
model.fit(X_train_full, y_train_full)
joblib.dump(model, "data/outputs/lgbm_model.pkl")
```

### 6.5 Métricas finais

Reportar por fold espacial e média:

| Fold (UF) | RMSE | MAE | R² |
|---|---|---|---|
| AM | ... | ... | ... |
| PA | ... | ... | ... |
| ... | ... | ... | ... |
| **Média** | ... | ... | ... |

### 6.6 Normalização do score 0–100

```python
from scipy.stats import rankdata

predictions = model.predict(X_all)
score_risco = (rankdata(predictions) - 1) / (len(predictions) - 1) * 100
```

- Score baseado em percentil: 0 = menor risco observado, 100 = maior risco
- Robusto a outliers e garante distribuição uniforme no intervalo

---

## 7. Etapa 6 — Outputs (`src/dashboard/`)

### 7.1 Dashboard Streamlit (`src/dashboard/app.py`)

**Componentes:**
- Sidebar: filtros por estado (UF) e cluster
- Mapa principal: Folium coroplético com `score_risco` (escala verde → vermelho)
- Tabela: top-20 municípios de maior risco (filtrado pelos seletores)
- Gráfico SHAP: importância global das features (beeswarm via `shap.plots`)
- Linha do tempo: evolução do score de risco para um município selecionado

**Execução:**
```bash
.venv/Scripts/streamlit run src/dashboard/app.py
```

### 7.2 Frontend HTML/React (pós-implementação)
Usar a **frontend-design skill** do Claude Code para gerar versão estática com visual de portfólio.

### 7.3 Slide deck executivo (pós-implementação)
Usar a **pptx skill** do Claude Code para gerar apresentação com:
- Slide 1: Contexto e problema
- Slide 2: Fontes de dados e metodologia
- Slide 3: Mapa de clusters
- Slide 4: Mapa de score de risco
- Slide 5: Top-10 features SHAP
- Slide 6: Métricas do modelo por fold espacial
- Slide 7: Limitações e próximos passos

---

## 8. Schema completo do dataset consolidado

Arquivo: `data/processed/dataset.parquet`

| Coluna | Tipo | Origem | Etapa |
|---|---|---|---|
| `cod_ibge` | str (7 dígitos) | IBGE | Ingestão |
| `municipio` | str | IBGE | Ingestão |
| `uf` | str (2 letras) | IBGE | Ingestão |
| `ano` | int (2000–atual) | — | Ingestão |
| `area_km2` | float | IBGE | ETL |
| `populacao` | float | IBGE | ETL |
| `pib_agropecuario` | float (R$ milhares) | IBGE | ETL |
| `desmatamento_km2` | float | INPE/PRODES | ETL |
| `taxa_desmatamento` | float (%) | calculado | ETL |
| `area_uc_pct` | float (%) | ICMBio | ETL |
| `area_ti_pct` | float (%) | ICMBio/FUNAI | ETL |
| `autos_ibama` | int | IBAMA | ETL |
| `taxa_desmat_lag1` | float | calculado | Feature Eng |
| `taxa_desmat_lag2` | float | calculado | Feature Eng |
| `taxa_desmat_lag3` | float | calculado | Feature Eng |
| `taxa_desmat_roll3` | float | calculado | Feature Eng |
| `taxa_desmat_roll5` | float | calculado | Feature Eng |
| `std_desmat_roll3` | float | calculado | Feature Eng |
| `local_moran_i` | float | esda.Moran_Local | Feature Eng |
| `cluster_id` | int (-1 = ruído) | HDBSCAN | Clustering |
| `geometry` | geometry | IBGE shapefile | Ingestão |
| `score_risco` | float (0–100) | modelo | Predição |

---

## 9. Decisões de design e trade-offs

| Decisão | Alternativas descartadas | Justificativa |
|---|---|---|
| LightGBM único | XGBoost, CatBoost, Random Forest | Profundidade > breadth para portfólio |
| Spatial CV por UF | K-Fold aleatório, time-series split | Evita vazamento espacial entre municípios vizinhos |
| HDBSCAN | K-Means, GMM | Não requer número de clusters pré-definido; lida bem com ruído |
| Score por percentil | Min-max, sigmoid | Robusto a outliers; interpretável como "posição relativa" |
| Moran's I local como feature | Sem feature espacial | Captura efeito de contágio do desmatamento entre municípios |
| Cache local em parquet | Re-chamar MCP a cada execução | Resiliência a falhas de API; velocidade de iteração |
