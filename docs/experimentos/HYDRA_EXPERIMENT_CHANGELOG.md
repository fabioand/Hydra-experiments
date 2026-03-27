# HYDRA Experiment Changelog (curto)

## FirstTest100
- Pipeline: precomputed (`JPG + stack64.npy`).
- Config legado (sem `data.source_mode`).
- DataLoader sem paralelismo (`num_workers=0`).
- Objetivo: validar pipeline E2E inicial e monitoramento.

## SecondTest100
- Pipeline: on-the-fly (`JPG + JSON`) no DataLoader.
- Subset reprodutível de 100 casos (`longoeixo/subsets/N100_seed123`, split 80/20).
- DataLoader com paralelismo configurado (`num_workers=4`, `prefetch_factor=2`, `persistent_workers=true`).
- Geração de heatmap no grid final de treino (regra oficial).
- Resultado observado: melhora clara nas curvas de treino/val vs First.

## Ajuste após Second (avaliado no Third)
- `sigma_px_target` reduzido para `5.0` em `longoeixo/presets/unet256_stack64_preset.json`.
- Resultado no `ThirdTest100_sigma50`: não superou o `SecondTest100` (val_total e val_presence levemente piores; val_heatmap muito próximo).

## Ajuste atual (pré-Fourth)
- `sigma_px_target` ajustado para `5.2` (compromisso entre 5.0 e 5.5).

## Próxima run sugerida
- Nome: `FourthTest1000_sigma52`.
- Escopo: 1000 amostras com subset reprodutível (`N1000_seed123`) via script de treino local.
- Estado atual dos dados locais: 999 pares `JPG+JSON` disponíveis; fallback preparado: `FourthTest999_sigma52`.

## FifthTest999_absentHM0 (preparado)
- Objetivo: remover contribuição de heatmap para dentes ausentes e deixar a decisão de presença concentrada na head de classificação.
- Mudança de loss: novo parâmetro opcional `training.absent_heatmap_weight`.
  - `1.0` (default): comportamento legado.
  - `0.0`: ignora canais ausentes na loss de heatmap.
- Config pronta: `hydra_train_config_fifth999_absenthm0.json`.
- Preset ativo na config: `FifthTest999_absentHM0`.
