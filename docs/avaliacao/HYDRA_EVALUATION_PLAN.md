# HYDRA Evaluation Plan

Plano de avaliação para comparar predições do modelo Hydra U-Net MultiTask com ground truth.

## 1) Métricas de Heatmap (64 canais)

Objetivo: medir qualidade global da predição densa dos mapas.

- MSE por canal
- Dice por canal
- Dice médio global

Saídas sugeridas:
- média e desvio padrão por canal
- média macro dos 64 canais

## 2) Métricas de Localização de Pontos (principal)

Objetivo: medir erro geométrico dos pontos previstos.

Procedimento:
1. Extrair ponto previsto por canal via `argmax`.
2. Extrair ponto GT por canal via `argmax` (ou ponto original equivalente no target final).
3. Calcular distância euclidiana em pixels no espaço de 256x256.

Métricas:
- erro médio (px)
- erro mediano (px)
- P90 (px)
- taxa dentro de tolerância:
  - `<= 3 px`
  - `<= 5 px`
  - `<= 10 px`

## 3) Métricas de Presença Dentária (32 classes)

Objetivo: medir classificação presença/ausência por dente.

- AUC macro
- F1 macro
- Precision macro
- Recall macro
- acurácia por dente

Observação:
- usar `sigmoid(logits)` + threshold (inicialmente 0.5, depois calibrável por validação).

## 4) Métricas Combinadas por Dente

Objetivo: avaliar coerência entre classificação de presença e localização.

- erro de ponto **condicionado a GT-presença=1**
- taxa de falso ponto em dentes ausentes (GT-presença=0)
- taxa de ponto válido quando presença prevista é positiva

## 5) Relatórios Recomendados

## 5.1 Resumo global

- tabela final com:
  - MSE global
  - Dice global
  - erro médio/mediano/P90
  - AUC macro / F1 macro presença

## 5.2 Relatório por dente

- métricas por dente (`11..18, 21..28, 31..38, 41..48`):
  - F1 presença
  - erro de ponto (quando presente)
  - taxa `<=5px`

## 5.3 Relatório por quadrante

- agregação por quadrante para análise clínica:
  - superior direito
  - superior esquerdo
  - inferior esquerdo
  - inferior direito

## 5.4 Curvas por época

- `train/val`:
  - total loss
  - heatmap loss
  - presence loss
  - erro de ponto val
  - F1/AUC val

## 5.5 Inspeção visual

- overlays de comparação `pred vs gt`
- atenção por camada (mean/max)
- before/after augmentação

## 6) Artefatos de Saída da Avaliação

Gerar no mínimo:
- `metrics_summary.json`
- `metrics_per_tooth.csv`
- `metrics_per_quadrant.csv`
- `pred_vs_gt_samples/` (painéis)

## 7) Critérios iniciais de aceitação (fase protótipo)

Sugestão para primeira iteração local (100 casos):
- tendência clara de queda de loss em treino/val
- erro mediano de ponto em queda por época
- F1 de presença acima do baseline aleatório com estabilidade

Após isso, escalar para 16k e recalibrar metas.
