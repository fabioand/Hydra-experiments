# Índice de Scripts de Métricas

Guia rápido para saber **qual script usar**, **como rodar** e **quais artefatos esperar**.

## Hydra - avaliação principal

| Script | Comando típico | Saídas principais | Quando usar |
|---|---|---|---|
| `eval.py` | `.venv/bin/python eval.py --config hydra_train_config_full70_15_15.json --run-name <RUN> --split test` | `runs/<RUN>/eval/metrics_summary.json`, `metrics_per_tooth.csv`, `metrics_per_quadrant.csv`, `pred_vs_gt_samples/` | Avaliação oficial da run (presença + geometria + combinado). |
| `longoeixo/scripts/run_hydra_eval.sh` | `bash longoeixo/scripts/run_hydra_eval.sh <RUN>` | Mesmas saídas do `eval.py` no split `val` | Atalho para avaliação rápida no fluxo padrão. |
| `longoeixo/scripts/run_hydra_smoke.sh` | `bash longoeixo/scripts/run_hydra_smoke.sh smoke_demo` | Treino smoke + `eval/` da run smoke | Validação E2E rápida do pipeline. |

### `eval.py` - modo de presença por heatmap com calibração

Quando a cabeça de presença não estiver sendo usada para decisão, o `eval.py` suporta inferir presença via score de heatmap:
- `--presence-source heatmap`
- score por dente: média dos picos dos 2 canais do dente.
- `--presence-source heatmap_composite`
- score por dente (composto): pico + energia local da vizinhança + nitidez + balanceamento do par + plausibilidade de distância.

Calibração oficial no `val` (gera thresholds por dente):

```bash
.venv/bin/python eval.py \
  --config hydra_train_config_full70_15_15_ec2.json \
  --run-name <RUN> \
  --split val \
  --presence-source heatmap \
  --presence-threshold 0.1 \
  --calibrate-presence-thresholds
```

Aplicação oficial no `test` (com JSON calibrado no `val`):

```bash
.venv/bin/python eval.py \
  --config hydra_train_config_full70_15_15_ec2.json \
  --run-name <RUN> \
  --split test \
  --presence-source heatmap \
  --presence-thresholds-json runs/<RUN>/eval/presence_thresholds_heatmap_calibrated_val.json
```

## Hydra - presença (foco em erro)

| Script | Comando típico | Saídas principais | Quando usar |
|---|---|---|---|
| `eval_presence_top_errors.py` | `.venv/bin/python eval_presence_top_errors.py --config hydra_train_config_full70_15_15.json --run-name <RUN> --split test --top-k 300` | `runs/<RUN>/eval_presence/presence_errors_summary.json`, `presence_errors_per_sample.csv`, `presence_top_errors_topK.csv`, histograma CSV/PNG | Rankear radiografias com maior suspeita de erro de presença. |
| `visualize_presence_top_errors_overlay.py` | `.venv/bin/python visualize_presence_top_errors_overlay.py --config hydra_train_config_full70_15_15.json --run-name <RUN> --top-k 200` | `runs/<RUN>/eval_presence_overlay/index.html` + `images/` | Revisão visual dos top erros de presença (GT vs predição). |

## Auditorias de anotação (dataset/GT)

| Script | Comando típico | Saídas principais | Quando usar |
|---|---|---|---|
| `audit_axis_inversion.py` | `.venv/bin/python audit_axis_inversion.py --config hydra_train_config_full70_15_15.json --split test --top-k 300` | `annotation_audit/axis_inversion_<split>/axis_inversion_summary.json`, CSVs, HTML e imagens | Detectar suspeita de inversão `p1/p2` por dente. |
| `audit_morphology_suspects.py` | `.venv/bin/python audit_morphology_suspects.py --config hydra_train_config_full70_15_15.json --split test --baseline-split train --top-k 300` | `annotation_audit/morphology_<split>/morphology_audit_summary.json`, CSVs, histogramas, HTML e overlays | Encontrar suspeitos morfológicos/geométricos no GT. |

## Métricas de dataset

| Script | Comando típico | Saídas principais | Quando usar |
|---|---|---|---|
| `longoeixo/scripts/count_dentition_coverage.py` | `.venv/bin/python longoeixo/scripts/count_dentition_coverage.py --imgs-dir longoeixo/imgs --json-dir longoeixo/data_longoeixo --out-json /tmp/dentition_summary.json` | Resumo no terminal + JSON/CSV opcionais | Medir cobertura dentária e histograma de dentes presentes. |
| `longoeixo/scripts/calc_dataset_horizontal_stats.py` | `.venv/bin/python longoeixo/scripts/calc_dataset_horizontal_stats.py --imgs-dir longoeixo/imgs --json-dir longoeixo/data_longoeixo --out-dir /tmp/hstats --top-k 200` | Estatísticas no terminal + ranking CSV + `index.html` com mosaico | Diagnóstico de distribuição geométrica/shape do dataset. |

## DAE - avaliação

| Script | Comando típico | Saídas principais | Quando usar |
|---|---|---|---|
| `dae_longoeixo/eval_dae.py` | `.venv/bin/python dae_longoeixo/eval_dae.py --config dae_longoeixo/dae_train_config.json --run-name <RUN_DAE> --split test` | `runs/<RUN_DAE>/eval/metrics_summary.json`, `metrics_per_tooth.csv`, `per_sample_errors.csv`, `pred_vs_gt_samples/` | Avaliar imputação do DAE (MSE/MAE/erro de ponto). |
| `dae_longoeixo/scripts/run_dae_eval.sh` | `bash dae_longoeixo/scripts/run_dae_eval.sh <RUN_DAE> test` | Mesmas saídas do `eval_dae.py` | Atalho para avaliação DAE. |
| `dae_longoeixo/scripts/run_dae_smoke.sh` | `bash dae_longoeixo/scripts/run_dae_smoke.sh dae_smoke` | Treino smoke + `eval/` do DAE | Validação E2E rápida do pipeline DAE. |

## Métricas de treino (online)

| Script | Comando típico | Saídas principais | Quando usar |
|---|---|---|---|
| `train.py` | `.venv/bin/python train.py --config hydra_train_config_full70_15_15.json --run-name <RUN>` | `runs/<RUN>/metrics.csv`, `tensorboard/`, `best.ckpt`, `last.ckpt`, `train_visuals/` | Acompanhar treino Hydra por época e selecionar `best.ckpt` por menor `val_total_loss`. |
| `dae_longoeixo/train_dae.py` | `.venv/bin/python dae_longoeixo/train_dae.py --config dae_longoeixo/dae_train_config.json --run-name <RUN_DAE>` | `runs/<RUN_DAE>/metrics.csv`, `tensorboard/`, `best.ckpt`, `last.ckpt`, `train_visuals/` | Acompanhar treino DAE por época e selecionar `best.ckpt` por menor `val_total_loss`. |

## Observações práticas

- Para comparação final de experimento, prefira `--split test` no `eval.py`.
- Para diagnosticar presença, use `eval_presence_top_errors.py` + `visualize_presence_top_errors_overlay.py` em sequência.
- Para rastreabilidade, sempre use `--run-name` explícito.
