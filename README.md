# AlfaMoney: réplicas dos artigos científicos citados no Quantitativo

Este repositório replica as estratégias dos artigos acadêmicos que o substack
[Quantitativo](https://www.quantitativo.com) ("Quant Trading Rules") implementou entre 2024 e 2026.
As fontes de dados são a **Norgate Data**, lida pela base do **AmiBroker** (plugin Norgate) ou pela
API Python da Norgate, e o **Portfolio123**, usado onde se aplica.

> **Status:** o código foi validado com testes e dados sintéticos (`pytest`). Ele **ainda não foi
> executado com dados reais**. A Norgate exige o Norgate Data Updater (Windows), os arquivos AFL
> precisam ser rodados no AmiBroker e o P123 exige uma chave de API. Os números das tabelas abaixo são os reportados pelos artigos e pelo
> Quantitativo, não resultados deste código.

## Artigos replicados

| # | Post do Quantitativo | Artigo | Módulo | Dados |
|---|---|---|---|---|
| 1 | Intraday Momentum for ES and NQ | Zarattini, Aziz & Barbon (2024), *Beat the Market*, SSRN 4824172 | `strategies/intraday_momentum.py` | barras de 1 minuto (CSV externo)¹ |
| 2 | Learning to Rank | Poh, Lim, Zohren & Roberts (2021), JFDS, arXiv 2012.07149 | `strategies/learning_to_rank.py` | Norgate (Russell 3000 com deslistadas); P123 opcional |
| 3 | Coding Trend Factor | Han, Zhou & Zhu (2016), *A Trend Factor*, JFE 122(2) | `strategies/trend_factor.py` | Norgate; P123 (upload do fator + bucket test) |
| 4 | The Bitter Lesson | Takeuchi & Lee (2013), *Deep Learning … Momentum*, Stanford CS229 | `strategies/deep_momentum.py::run_takeuchi_lee` | Norgate |
| 5 | Short-Term Basis Reversal | Rossi, Zhang & Zhu (2025), SSRN 5250499 | `strategies/basis_reversal.py` | Norgate: contratos futuros individuais |
| 6 | The Unintended Consequences of Rebalancing | Harvey, Mazzoleni & Melone (2025), NBER w33554 | `strategies/rebalancing.py` | Norgate: ES e ZN contínuos |
| 7 | The Derivative Payoff Bias | Baltussen, Terstegge & Whelan (2023), SSRN 4562800 | `strategies/third_friday.py` | Norgate: SPY diário (abertura ≈ SOQ) |
| 8 | From Defense to Offense | Carlson (2025), *Defense First*, SSRN 5334772 | `strategies/defense_first.py` | Norgate: ETFs total return |
| 9 | Volume Shocks and Overnight Returns | Cartea, Cucuringu, Jin & Wilson (2025), SSRN 5156605 | `strategies/volume_shocks.py` | Norgate; P123 (só pesquisa semanal) |
| 10 | Asset Embeddings | Gabaix, Koijen, Richmond & Yogo (2025), NBER w33651 | `strategies/asset_embeddings.py` | Norgate + 13F da SEC²; proxy por PCA de retornos |
| 11 | Uncertainty | Han (2022), *Bimodal Characteristic Returns*, Mgmt Sci 68(10); Han & Qin (2025) | `strategies/deep_momentum.py::run_bimodal` | Norgate; P123 para market cap |
| 12 | Modern Statistical Arbitrage | Du, Walter & Ulrich (2026), *Cross-Market Alpha*, arXiv 2601.06499 | `strategies/stat_arb.py` | Norgate (diário); P123 (Alpha#46 semanal) |
| 13 | Network Momentum | Pu, Roberts, Dong & Zohren (2023), arXiv 2308.11294 | `strategies/network_momentum.py` | Norgate: cerca de 60 futuros contínuos |

¹ A Norgate não tem dados intradiários. Use barras de 1 minuto do pregão regular (RTH) vindas de
IBKR, Databento, FirstRate ou Kibot.
² Nem a Norgate nem o P123 têm posições por gestor (13F). Os data sets "Form 13F" ficam no SEC EDGAR.

**Posts não replicados, por não se basearem em artigo acadêmico:**
- os de reversão à média (2,11 Sharpe, First Principles, ML / Bouncing Back, Mean Reversion Curve);
- Fear & Greed (Connors) e Momentum L/S de 2024;
- Triangulated Stat Arb (série do Robot Wealth);
- Informational Edge;
- Portfolio Optimization: inspirado em Chen et al. (*Diffusion Factor Models*), mas o post não
  replica o método.

As newsletters "Quantitativo weekly" resumem centenas de artigos e ficaram fora do escopo.

## Regras implementadas e resultados de referência

| Estratégia | Regra central | Artigo | Quantitativo |
|---|---|---|---|
| Intraday momentum | Noise area de 14 dias, decisões a cada :00/:30, stop em max(banda, VWAP), zerado no fechamento, alvo de vol 2% e alavancagem ≤4x | SPY 2007–24: 19,6% a.a., Sharpe 1,33 | NQ ingênuo Sharpe 0,91; 90 dias / 3% / 8x: Sharpe 1,25; ES+NQ+long: Sharpe 1,57–1,67 |
| Learning to Rank | 21 features (Baz et al.), label = decil do mês seguinte, LambdaMART, long/short em decis, vol 15% | Sharpe ≈ 2,16 | R3000, 150% bruto, re-treino anual com janela de 15 anos: ≈18% a.a., DD 13% |
| Trend factor | MA/P em 11 janelas, regressão cross-section, média de 12 meses dos betas, quintis Q5–Q1 | 1,63%/mês | R3000, preço ≥ $5 |
| Takeuchi-Lee | 33 inputs com z-score, MLP 33-40-4-50-2, P(retorno > mediana), decis | ≈45,9% a.a., acurácia 53% | acurácia ≈52%, "a far cry" do artigo |
| Basis reversal | b = r1 − r2 semanal; aposta contra b da semana anterior; ≥30 dias até o vencimento | ≈18% a.a., Sharpe 1,42 | Feeder Cattle sozinho: Sharpe 0,91. Checagem da QuantReturns: Sharpe 0,92 (30 dias) e 1,58 (60 dias) |
| Rebalancing | 60/40 com drift; sinais de threshold (δ de 0,1% a 2,5%) e de calendário (últimos 5 dias úteis); spread ES−ZN por 1 dia | Sharpe > 1 | Sharpe 0,94 |
| 3rd Friday | Long do fechamento de quinta à abertura de sexta; short da abertura ao fechamento de sexta | +18,5 bp na abertura do vencimento | (código, sem backtest visível) |
| Defense First | TLT/GLD/DBC/UUP pelo momentum médio de 1/3/6/12 meses, pesos 40/30/20/10, filtro contra BIL → SPY | 10,9% a.a., Sharpe 0,89 | Sharpe 0,95 |
| Volume shock | VS = V / EMA(V, meia-vida 60) − 1; compra no fechamento, vende na abertura | prêmio overnight no quintil alto | decil topo long-only; colapsa com custos; biotech: 36% a.a., Sharpe 1,52 |
| Asset embeddings | Embeddings 13F → 50 clusters → 10 com maior reversão → quintis long/short intra-cluster diários | (artigo não traz regra de trading) | Sharpe 2,59, 18,5% a.a. |
| Bimodal (Han) | Classificação em 10 decis; ranking pelo retorno esperado Σ p_k μ_k | VW Sharpe 1,66; EW > 2,8 | "bem abaixo" |
| Alpha#46 | (MA3+MA6+MA12+MA24)/(4·Close), demeaned por dia, rebalanceamento diário | 17 de 191 sinais significativos | sozinho: Sharpe < 0,5 com DD 55,7%; combo de 17: Sharpe 1,15–1,76 |
| Network momentum | 8 features, grafo de Kalofolias (ensemble de 5 janelas), Ã·X, regressão linear agrupada | Sharpe 1,51 | Sharpe ≈ 2,1 (vs 1,13 só com features próprias) |

## Usando a base do AmiBroker (Norgate Data)

Há dois caminhos. Os dois usam a mesma base Norgate que o AmiBroker já tem.

### 1. Exportar a base do AmiBroker e rodar as 13 estratégias em Python

Assim as estratégias que o AFL não comporta (ML, regressões, grafos, contratos futuros
individuais) rodam sobre os dados da sua base do AmiBroker.

1. Copie a pasta `amibroker/` para `C:\Program Files\AmiBroker\Formulas\AlfaMoney\`.
2. No AmiBroker, abra `AlfaMoney_Export_Norgate.afl` numa janela de Analysis e configure:
   **Range = All quotes**, **Apply to = Filter → Watchlist**. Rode **Explore** três vezes, uma
   para cada grupo:

   | Watchlist | Parâmetro *Asset type* | Pasta de saída |
   |---|---|---|
   | `Russell 3000 Current & Past` (ou `S&P 500 Current & Past`) | Equities | `...\ami_export\equities` |
   | uma watchlist com SPY, TLT, GLD, DBC, UUP, BIL | ETF/Other | `...\ami_export\etfs` |
   | uma watchlist com os futuros contínuos (`&ES_CCB`, …) e os contratos (`ZC-2024Z`, …) | Futures | `...\ami_export\futures` |

   O exportador grava, para cada símbolo:
   - OHLCV e o preço não ajustado (`NorgateOriginalCloseTimeSeries`);
   - a participação histórica nos índices S&P 500, Russell 1000/3000/2000 e Nasdaq-100
     (`NorgateIndexConstituentTimeSeries`);
   - as barras com *padding* são descartadas.

   Configure o ajuste de preços do plugin Norgate como *total return* antes de exportar.
3. Rode o Python apontando para a exportação:

```bash
python scripts/run_norgate.py --all --source amibroker --ami-export C:\AlfaMoney\ami_export --cost-bps 5
```

A primeira leitura junta os CSVs num cache `_alfamoney_combined.parquet` dentro da pasta.

### 2. Backtests nativos no AmiBroker (AFL)

Estratégias que cabem no backtester de portfólio do AmiBroker:

| Arquivo | Estratégia | Rodar em |
|---|---|---|
| `amibroker/Alpha46_StatArb.afl` | Alpha#46 long/short em decis, rebalanceamento diário (Modern Stat Arb) | `S&P 500 Current & Past` ou `Russell 3000 Current & Past` |
| `amibroker/VolumeShock_Overnight.afl` | Volume shock: compra no fechamento e vende na abertura seguinte, decil topo, teto por % do ADV | `Russell 3000 Current & Past` |
| `amibroker/DefenseFirst.afl` | Defense First: 40/30/20/10 com filtro contra BIL → SPY, rebalanceamento mensal | watchlist TLT GLD DBC UUP SPY BIL |
| `amibroker/ThirdFriday_Spike.afl` | 3rd Friday: trade A (fechamento → abertura) ou B (abertura → fechamento) | SPY |

As três primeiras usam `include/AlfaMoney_CBT.afl`, um custom backtester genérico que rebalanceia
para pesos-alvo ou para quantis long/short. Ele também fecha posições que saem do índice ou são
deslistadas.

> Os AFL foram escritos para AmiBroker 6.x com o plugin Norgate, mas **não puderam ser executados
> neste ambiente**. Faça um *Check* (Ctrl+F7) e compare com o resultado em Python antes de confiar
> nos números.

O Alpha#46 em AFL usa pesos iguais por perna; a versão Python pondera pelo sinal demeaned.

## Uso

```bash
pip install -r requirements.txt
python -m pytest -q                                  # testes com dados sintéticos

# Norgate via exportação do AmiBroker
python scripts/run_norgate.py --all --source amibroker --ami-export C:\AlfaMoney\ami_export

# Norgate via API Python (Windows, com o NDU aberto)
python scripts/run_norgate.py --all --start 1995-01-01 --cost-bps 5
python scripts/run_norgate.py --strategy network_momentum basis_reversal
python scripts/run_norgate.py --strategy intraday_momentum --minute-csv NQ_1min.csv

# Portfolio123
export P123_API_ID=...  P123_API_KEY=...
python scripts/run_p123.py alpha46
python scripts/run_p123.py trend_factor               # sinal calculado com Norgate, testado no P123
```

Os resultados vão para `results/`: séries de retorno e `summary.csv` com CAGR, Sharpe, MaxDD,
turnover e exposição.

## Quando usar Norgate ou Portfolio123

- **Norgate:** preços diários sem viés de sobrevivência (inclui deslistadas), composição histórica
  dos índices, futuros contínuos e contratos individuais. Atende a todas as estratégias de preço e
  de futuros.
- **Portfolio123:** fundamentos point-in-time (market cap para filtros de tamanho, dummies de
  tamanho e pesos por valor) e backtests de screens/rankings semanais ou mensais. Também roda
  sinais calculados em Python e enviados como *custom stock factor*.
  - **Não serve para:** operações overnight, rebalanceamento diário e futuros.

## Ressalvas importantes

- **Regras não confirmadas:** o site quantitativo.com estava bloqueado no ambiente onde o código foi
  escrito. As regras vêm dos resumos dos artigos e de trechos dos posts. Antes de tirar conclusões,
  confira os parâmetros contra os PDFs. Os principais pontos em aberto estão comentados nos módulos:
  - definição exata das 20 features diárias do Takeuchi-Lee;
  - corte em quantis do basis reversal;
  - lista dos 17 sinais GTJA;
  - parâmetros α/β do grafo do network momentum.
- **Volume shocks:** com barras diárias, o volume do dia inteiro só é conhecido após o leilão de
  fechamento. O backtest usa a versão do artigo; para operar de fato, use o volume acumulado até
  15:45 (`volume=`).
- **Proxies:**
  - Asset embeddings sem 13F usa PCA de retornos, o que **não** é o método do artigo.
  - Tamanho sem P123 usa o volume financeiro médio como proxy de market cap.
- **Custos:** o padrão é zero. Use `--cost-bps` (ações e ETFs) e `cost_per_spread` / `cost_per_side`
  (futuros).
- **Velocidade:** `load_equity_panel` consulta cada símbolo da Norgate (ativos e deslistados). A
  primeira execução é lenta e depois usa o cache em `.cache/`.
