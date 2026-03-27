# HYDRA Experiment Note - FirstTest100

Data: 2026-03-20
Run: `FirstTest100`

## 1) Objetivo

Primeiro experimento de validação com 100 amostras para verificar:
- estabilidade do pipeline
- convergência inicial
- comportamento das losses por tarefa (heatmap vs presença)

## 2) Setup

- Dataset: 100 exemplos (`longoeixo/imgs` + `longoeixo/gaussian_maps_stack64`)
- Split: 80/20 reprodutível (`longoeixo/splits.json`)
- Modelo: `HydraUNetMultiTask` (backbone `resnet34`)
- Loss:
  - heatmap: `0.8*MSE + 0.2*SoftDice`
  - presença: `BCEWithLogits`
  - total: `1.0*heatmap + 0.3*presence`
- Device local: auto (`cuda -> mps -> cpu`)
- Config desta rodada:
  - `epochs=30`
  - `batch_size=4`
  - `early_stopping_patience=8`

## 3) Caminhos de artefatos da run

- Run dir:
  - `longoeixo/experiments/hydra_unet_multitask/runs/FirstTest100`
- Checkpoints:
  - `best.ckpt`
  - `last.ckpt`
- Métricas por época:
  - `metrics.csv`
- TensorBoard:
  - `tensorboard/`
- Viewer visual (augmentação + atenção):
  - `train_visuals/index.html`

## 4) Leitura rápida dos resultados iniciais

Observação principal:
- `train_total` caiu de forma consistente
- `val_heatmap` também caiu gradualmente
- `val_presence` apresentou oscilação alta (instável)

Ponto de melhor validação global observado:
- `val_total` mínimo em torno da `epoch 10` (`~0.53597`)

Interpretação:
- A tarefa de localização (heatmap) está aprendendo com estabilidade.
- A tarefa de presença ainda tem variância alta com apenas 100 casos.

## 5) Conclusão do experimento

`FirstTest100` cumpre papel de baseline funcional do pipeline.

Próximo passo recomendado:
- escalar para `FirstTest1000` para reduzir ruído em validação de presença e obter sinal mais confiável para ajuste fino.
