# Hydra Longo-Eixo: Plano Multi-ROI (3 Retangulos)

Data: 2026-03-23  
Escopo: melhorar deteccao/localizacao de longoeixo em panoramicas com estrategia de 3 ROIs, reduzindo compressao vertical e reduzindo saida por rede de 64 canais para 20-24.

> Atualizacao: este plano foi supersedido pela especificacao de janelas fixas e rede lateral compartilhada em:
> `docs/planos/HYDRA_LONGOEIXO_FIXED_WINDOWS_LATERAL_SHARED_SPEC.md`

## 1) Motivacao Tecnica

Problemas do setup atual (panoramica inteira -> 64 canais):
- perda de detalhe fino por resize global (principalmente estrutura radicular);
- competicao entre muitas classes no mesmo head;
- confusao entre dentes distantes na arcada.

Hipotese:
- usar 3 ROIs anatomicas aumenta resolucao efetiva por dente;
- cada rede especializada em subconjunto reduz ambiguidade e melhora gradiente;
- reduzir canais para 20-24 por rede melhora estabilidade de treino e inferencia.

## 2) Definicao Final dos 3 Retangulos

Pontos anatomicos usados (API `v1/panoramics/anatomic_points`):
- `Condilo - Esquerdo`
- `Condilo - Direito`
- `E.N.A.`
- `Mentoniano`

Convencao (coordenadas da imagem):
- `CE = (x_ce, y_ce)` condilo esquerdo
- `CD = (x_cd, y_cd)` condilo direito
- `ENA = (x_ena, y_ena)`
- `MEN = (x_men, y_men)`

Retangulos:

1. `R_LEFT` (lateral esquerda anatomica)
- `x1 = min(x_ce, x_ena)`
- `x2 = max(x_ce, x_ena)`
- `y1 = y_ce`
- `y2 = y_men`

2. `R_RIGHT` (lateral direita anatomica)
- `x1 = min(x_cd, x_ena)`
- `x2 = max(x_cd, x_ena)`
- `y1 = y_cd`
- `y2 = y_men`

3. `R_CENTER` (centro)
- `x_mid_left = (x_ce + x_ena) / 2`
- `x_mid_right = (x_cd + x_ena) / 2`
- `x1 = min(x_mid_left, x_mid_right)`
- `x2 = max(x_mid_left, x_mid_right)`
- `y1 = min(y_ce, y_cd)`  (linha dos condilos)
- `y2 = y_men`

Observacoes:
- todos os limites sao clampados para `[0, W] x [0, H]`;
- se `y2 <= y1`, forca-se altura minima de 1 px.

## 3) Particao de Dentes por Retangulo (Sem Redundancia)

Objetivo: cada dente pertence a apenas um head/rede (sem duplicar labels).

Dicionario oficial:

```python
TOOTH_SETS_BY_RECT = {
    "R_LEFT": [  # pre-molares + molares (lado esquerdo anatomico)
        "24", "25", "26", "27", "28",
        "34", "35", "36", "37", "38",
    ],
    "R_RIGHT": [  # pre-molares + molares (lado direito anatomico)
        "14", "15", "16", "17", "18",
        "44", "45", "46", "47", "48",
    ],
    "R_CENTER": [  # incisivos + caninos
        "11", "12", "13", "21", "22", "23",
        "31", "32", "33", "41", "42", "43",
    ],
}
```

Contagem de canais:
- `R_LEFT`: 10 dentes x 2 pontos = 20 canais
- `R_RIGHT`: 10 dentes x 2 pontos = 20 canais
- `R_CENTER`: 12 dentes x 2 pontos = 24 canais

## 4) Proposta de Arquitetura de Treino

Treino:
- 3 modelos especialistas (left, center, right), cada um com seu proprio head.
- loss mascarada por conjunto de dentes da ROI correspondente.
- normalizacao/resize por ROI (nao pela panoramica inteira).

Inferencia:
- obter pontos anatomicos;
- montar 3 ROIs;
- rodar 3 modelos especialistas;
- reprojetar coordenadas ROI -> coordenadas globais da panoramica;
- merge final por mapeamento deterministico dente->rede (sem concorrencia).

## 5) Planos de Melhoria da Rede de Longo-Eixo

Melhorias imediatas:
- adotar pipeline multi-ROI acima;
- manter augmentacoes geometricas leves por ROI;
- auditar cobertura de dentes por retangulo antes do treino.

Melhorias seguintes:
- fallback para modelo global quando pontos anatomicos faltarem;
- score de confianca por dente para filtrar outliers;
- calibracao por grupo dental (incisivo/canino/premolar/molar).

## 6) Validacao Recomendada

Comparar baseline atual vs multi-ROI:
- erro mediano por dente (px);
- recall de deteccao por dente;
- taxa de falha por grupo dental (inc/can vs pre/mol);
- taxa de "dente fora do retangulo esperado" no dataset.

Critico:
- validar cobertura em todas as 999 amostras pareadas antes de iniciar treino longo.
