# HYDRA Evaluation Criteria (Oficial)

Este documento define **o que medimos**, **por que medimos**, e **quais métricas valem para decisão** no Hydra.

## 1) Contexto da tarefa

A Hydra é multitarefa:
- `heatmap_logits` (64 canais): localização de 2 pontos por dente.
- `presence_logits` (32): presença/ausência por dente.

No uso operacional, a cabeça de presença faz gating dos pontos:
- se o dente é previsto ausente, os pontos desse dente não são usados.

## 2) Princípio de avaliação

Como pontos de dentes ausentes são irrelevantes para decisão final (gating por presença), as métricas principais de localização devem focar em **dentes presentes**.

## 3) Métricas principais (headline)

## 3.1 Presença (32 dentes)

Objetivo: qualidade da decisão presença/ausência.

Usar como principais:
- `presence_f1_macro`
- `presence_auc_macro`

Suporte:
- `precision_macro`, `recall_macro`, `accuracy_macro`

## 3.2 Localização (present-only)

Objetivo: precisão geométrica dos pontos onde o dente realmente existe.

Usar como principais:
- `point_error_median_px` (global, somente presentes)
- `point_within_5px_rate` (global, somente presentes)
- `dice_macro_present_channels` (somente canais presentes)

Suporte:
- `point_error_mean_px`, `point_error_p90_px`
- `mse_present_global_mean`, `dice_present_global_mean`

### Visão operacional (pred_presence)

Para refletir o uso real (com gating por presença predita), reportar também:
- `point_error_median_px_when_pred_presence_pos`
- `point_within_5px_rate_when_pred_presence_pos`

Essas métricas avaliam os pontos apenas quando a própria rede decidiu que o dente está presente.

## 3.3 Métricas combinadas

Objetivo: qualidade da decisão final ponta-a-ponta (presença + ponto).

Usar:
- `false_point_rate_gt_absent_global`
- `valid_point_rate_when_pred_presence_pos_global`

## 4) Métricas secundárias (diagnóstico)

As métricas globais de heatmap em todos os canais (`mse_global_mean`, `dice_global_mean`, `dice_macro_64`) permanecem úteis para debug, mas **não devem ser o critério primário de qualidade geométrica** quando há gating por presença.

## 5) Split oficial para decisão final

Para report final:
- split fixo `train/val/test = 70/15/15`
- treino usa `train` + `val` (early stopping/hyperparams)
- decisão final usa somente `test`

## 5.1 Política de calibração de presença no `eval.py`

Quando a cabeça de presença não for confiável/ativa (ou quando quisermos comparar regras de gating), a presença pode ser inferida por score de heatmap no `eval.py`.

Opções de score suportadas:
- `heatmap` (simples): média dos picos dos 2 canais do dente.
- `heatmap_composite`: combinação de pico, energia local, nitidez, consistência do par e plausibilidade de distância.

Política oficial:
- calibrar thresholds **somente no `val`** (comparando score com GT),
- congelar esses thresholds por dente (`11..48`) em JSON,
- aplicar os thresholds congelados no `test`.

Regra de ouro:
- nunca calibrar no `test` (evitar vazamento de avaliação),
- o `test` deve medir somente generalização.

Comando recomendado de calibração (split `val`):
- `eval.py --split val --presence-source heatmap --calibrate-presence-thresholds --presence-threshold 0.1`

Comando recomendado de aplicação (split `test`):
- `eval.py --split test --presence-source heatmap --presence-thresholds-json <json_calibrado_no_val>`

## 6) Critério de promoção de experimento

Um experimento é candidato a promoção quando, no `test`:
1. melhora `presence_f1_macro` (ou mantém com variação mínima aceitável),
2. melhora `point_error_median_px` e/ou `point_within_5px_rate` em presentes,
3. não piora de forma relevante `false_point_rate_gt_absent_global`.

## 7) Decisão atual registrada

Decisão vigente:
- priorizar métricas `present-only` para localização,
- manter métricas globais por canal apenas como apoio diagnóstico,
- manter avaliação final no split `test`.
- para presença inferida por heatmap: calibrar no `val` e aplicar no `test`.
