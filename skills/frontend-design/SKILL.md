# SKILL: frontend-design

## Propósito

Gerar um **dashboard HTML single-file** (sem build step, sem Node.js, sem bundler)
completamente funcional e pronto para abrir no browser. O arquivo final deve ser
auto-contido: todo CSS, JavaScript e dados embutidos em um único `.html`.

---

## Stack obrigatória

| Camada | Tecnologia | Versão CDN |
|---|---|---|
| UI framework | **Tailwind CSS** | via CDN (Play CDN) |
| Mapa coroplético | **Leaflet.js** | 1.9.x via CDN |
| Gráficos | **Chart.js** | 4.x via CDN |
| Dados geoespaciais | **GeoJSON embutido** | — |
| JavaScript | Vanilla ES2022 | — |

**Não usar React, Vue, Angular, webpack, npm ou qualquer ferramenta de build.**
O arquivo deve funcionar offline com `file://` após os CDNs serem carregados.

---

## Paleta de cores

```
Verde floresta (baixo risco)  : #22c55e  (Tailwind green-500)
Amarelo alerta (risco médio)  : #eab308  (Tailwind yellow-500)
Laranja alto risco            : #f97316  (Tailwind orange-500)
Vermelho crítico              : #ef4444  (Tailwind red-500)
Fundo escuro (sidebar/header) : #1e293b  (Tailwind slate-800)
Fundo principal               : #f8fafc  (Tailwind slate-50)
Texto principal               : #0f172a  (Tailwind slate-900)
Texto secundário              : #64748b  (Tailwind slate-500)
Borda                         : #e2e8f0  (Tailwind slate-200)
```

Escala de risco no mapa (score 0–100):
- 0–20  → `#22c55e`
- 21–40 → `#86efac`
- 41–60 → `#eab308`
- 61–80 → `#f97316`
- 81–100 → `#ef4444`

---

## Layout

```
+-------------------------------------------------------------+
|  HEADER  (fundo slate-800, texto branco, logo + título)     |
+------------+------------------------------------------------+
|            |  KPI CARDS (4 cards: desmat total, taxa média, |
|  SIDEBAR   |  score médio, mun score>=80)                   |
|  (filtros) +------------------------------------------------+
|            |  TABS: [Mapa] [Ranking] [Tendência] [Sobre]    |
|  • Ano     +------------------------------------------------+
|  • UF      |                                                |
|  • Cluster |  CONTEÚDO DA TAB ATIVA                        |
|            |                                                |
+------------+------------------------------------------------+
```

- **Header**: altura 56px, `bg-slate-800`, logo emoji 🌿, título "Risco de Desmatamento · Amazônia Legal".
- **Sidebar**: largura 240px, `bg-white border-r`, filtros com `<select>` e checkboxes estilizados.
- **KPI cards**: `grid grid-cols-4 gap-4`, cada card com ícone, valor destacado e label.
- **Tabs**: `border-b` com botões, tab ativa com `border-b-2 border-green-500 text-green-700`.

---

## Tab: Mapa

- **Leaflet.js** com tiles `CartoDB.Positron`.
- Municípios desenhados como `L.geoJSON` com `fillColor` baseado em `score_risco`.
- **Popup** ao clicar: nome do município, UF, score, taxa de desmatamento, desmatamento_km2.
- **Legenda** fixa no canto inferior direito com a escala de cores.
- Zoom inicial: 5, centro: `[-6, -55]`.
- GeoJSON embutido: último ano disponível, apenas propriedades essenciais.

---

## Tab: Ranking

- Tabela HTML estilizada (`table-auto w-full`).
- Top-20 municípios por `score_risco` do ano selecionado.
- Coluna `Score` com badge colorido (mesma escala do mapa).
- Linhas alternadas (`even:bg-slate-50`).
- Cabeçalho fixo com `sticky top-0`.

---

## Tab: Tendência

- **Chart.js** `line chart` com série histórica de `taxa_desmatamento` média por UF.
- Uma linha por UF selecionada, cores distintas.
- Eixo X: anos. Eixo Y: taxa média (%/ano).
- Segundo gráfico: `bar chart` empilhado com desmatamento total (km²) por UF e ano.
- Ambos responsivos (container 400px de altura).

---

## Tab: Sobre

- Card descritivo: objetivo, fontes de dados, metodologia (LightGBM + Optuna + SHAP + HDBSCAN).
- Métricas do modelo: R²=0.76, RMSE=0.109.
- Rodapé com data de geração.

---

## Dados embutidos

```html
<script>
  const GEODATA   = { /* GeoJSON FeatureCollection — último ano */ };
  const ALL_DATA  = { /* { ano: [ {cod_ibge, score_risco, taxa_desmatamento, ...} ] } */ };
  const SERIES    = { /* { uf: { ano: taxa_media } } */ };
  const METRICS   = { /* métricas do modelo por fold */ };
</script>
```

O script Python de geração deve:
1. Ler `data/outputs/predictions.parquet`.
2. Simplificar geometrias com `gdf.simplify(0.01)`.
3. Exportar `ALL_DATA` com todos os anos (para filtro dinâmico funcionar).
4. Exportar `GEODATA` com GeoJSON do último ano (geometrias + properties).
5. Exportar `SERIES` com agregação histórica por (UF, ano).
6. Embutir tudo no template HTML e salvar em `reports/dashboard.html`.

---

## Comportamento dinâmico (JavaScript)

- Ao mudar **Ano**: atualizar KPIs, recolorir mapa, reordenar ranking.
- Ao mudar **UF**: filtrar série temporal e ranking.
- O mapa usa GeoJSON fixo do último ano para geometrias; cores mudam via `ALL_DATA`.
- Estado dos filtros em objeto `state = { ano, ufs }`.

---

## Arquivo de saída

`reports/dashboard.html` — tamanho alvo < 5 MB.

---

## Requisitos de qualidade

- [ ] Abre sem erros no Chrome/Edge com `file://`.
- [ ] Mapa renderiza todos os municípios com cor correta.
- [ ] Filtro de ano atualiza mapa, KPIs e ranking dinamicamente.
- [ ] Sem `console.error` no DevTools.
- [ ] HTML válido.
- [ ] Responsivo em telas 1280px+.
