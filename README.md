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

## Obtendo os dados

Os scripts de ingestão chamam **APIs públicas diretamente** — não há dependência de nenhum agente de IA para rodar o pipeline. O papel do MCP é explicado abaixo.

### Opção 1 — Python puro (sem IA)

Execute os scripts diretamente. Cada um baixa os dados da API correspondente e salva em `data/raw/`:

```bash
python -m src.ingestion.ibge_geo   # Shapefile dos municípios — IBGE FTP
python -m src.ingestion.inpe       # Desmatamento anual — TerraBrasilis WFS
python -m src.ingestion.ibge       # Pop., PIB agro, área — IBGE Servicodados
python -m src.ingestion.icmbio     # UCs e Terras Indígenas — TerraBrasilis WFS
python -m src.ingestion.ibama      # Autos de infração — Dados Abertos IBAMA
```

Todos os scripts têm **cache local**: se o arquivo já existe em `data/raw/`, a API não é chamada novamente.

### Opção 2 — Com agente MCP (Claude Code, Cursor, Cline etc.)

O projeto inclui configuração para o **[mcp-brasil](https://github.com/Mcp-Brasil/mcp-brasil)**, um servidor MCP que expõe as mesmas APIs como ferramentas de IA. Isso permite usar um agente para explorar os dados interativamente, debugar o pipeline e interpretar os resultados — mas **não é obrigatório** para rodar o código.

Para ativar, crie `.claude/mcp.json` (ou o equivalente do seu cliente):

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

> A chave do Portal da Transparência é gratuita: [api.portaldatransparencia.gov.br](https://api.portaldatransparencia.gov.br). É usada apenas para os dados do IBAMA — os demais scripts funcionam sem ela.

### Endpoints de referência

Caso queira acessar os dados manualmente ou integrar com outra ferramenta:

| Fonte | Endpoint |
|---|---|
| INPE/PRODES — desmatamento | `https://terrabrasilis.dpi.inpe.br/geoserver/prodes-legal-amz/ows` (WFS 2.0) |
| INPE/PRODES — municípios | mesma base, camada `prodes-legal-amz:municipalities_legal_amazon` |
| ICMBio — UCs | mesma base, camada `prodes-legal-amz:conservation_units_legal_amazon` |
| ICMBio — TIs | mesma base, camada `prodes-legal-amz:indigenous_area_legal_amazon` |
| IBGE — socioeconômico | `https://servicodados.ibge.gov.br/api/v3/agregados/` |
| IBGE — shapefile | `https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/Brasil/BR/BR_Municipios_2022.zip` |
| IBAMA — autos de infração | `https://dadosabertos.ibama.gov.br/dados/SIFISC/auto_infracao/auto_infracao/auto_infracao_csv.zip` |

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
│   ├── raw/                # Dados brutos coletados pelos scripts de ingestão (gitignore)
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

**Limitação da CV:** a validação cruzada adotada avalia *generalização geográfica* — o modelo foi testado em estados que não entraram no treino. Ela não avalia generalização temporal (ex.: prever 2023 com dados até 2022). Dado o regime de desmatamento acelerado pós-2019, resultados podem ser otimistas para cenários de ruptura. A validação temporal walk-forward é identificada como trabalho futuro.

---

## Limitações conhecidas

Este projeto é um portfólio de Data Science; algumas decisões metodológicas foram feitas conscientemente dentro desse escopo.

| Limitação | Impacto | Observação |
|---|---|---|
| **Validação geográfica, não temporal** | Métricas podem ser otimistas para anos de ruptura (pós-2019) | CV walk-forward como trabalho futuro |
| **Score 0–100 é ranking percentílico anual** | Score 80 significa "80º percentil naquele ano", não probabilidade de evento; não é comparável entre anos | Escolha de design intencional — transparência exige comunicação explícita |
| **Previsão 2026 usa proxies de 2025** | Variáveis estruturais (PIB, UC, população) copiadas do ano anterior | Aceitável dado que estas variáveis variam lentamente |
| **R² inferior em RR e TO** | Fold 3 (Roraima + Tocantins) tem R²≈0,55 vs. média 0,76 | Dados escassos em regiões esparsas — esperado e documentado |
| **Sem análise causal** | Correlação ≠ causalidade; autos IBAMA podem ser efeito, não causa | Escopo de portfólio preditivo, não causal |
| **SHAP calculado sobre dados de treino** | Sem holdout dedicado para SHAP | O modelo foi treinado em todo o histórico disponível |
| **Sem intervalos de confiança na previsão** | Score pontual sem estimativa de incerteza | Regressão quantílica ou bootstrap seriam necessários |

---

## Desenvolvimento

Este projeto foi desenvolvido com **Claude Code** seguindo o padrão **Spec Driven Development (SDD)**: especificação completa em `specs/` antes de qualquer implementação, commits atômicos por tarefa e validação de outputs a cada etapa.

---

## Licença

MIT
