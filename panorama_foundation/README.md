# Panorama Foundation

Base de projeto para pretreino auto-supervisionado em panoramicas e transferencia do encoder para:

- classificacao
- regressao
- segmentacao (U-Net com skip-connections)

Objetivo imediato: preparar o primeiro treino teste local pequeno do autoencoder com encoder `ResNet34`.

Nota de arquitetura:
- autoencoder de fundacao: **sem skip-connections** (gargalo real)
- segmentacao downstream: U-Net **com skip-connections**

## Estrutura

- `docs/PROJECT_PLAN.md`: plano, objetivos, abordagens e especificacoes.
- `configs/ae_local_smoke.json`: config inicial para treino local rapido.
- `dataset.py`: descoberta de imagens, split reprodutivel e dataset.
- `models.py`: modelos AE + cabecas de transferencia.
- `train_autoencoder.py`: loop de treino do AE.
- `train_transfer_skeleton.py`: esqueleto para treino downstream (freeze -> unfreeze).
- `scripts/run_ae_local.sh`: atalho para treino local.

## Primeiro teste local

1. Ajuste `images_dir` em `configs/ae_local_smoke.json`.
2. Rode:

```bash
./panorama_foundation/scripts/run_ae_local.sh
```

Saidas esperadas:

- `panorama_foundation/experiments/ae/runs/<RUN_NAME>/best.ckpt`
- `panorama_foundation/experiments/ae/runs/<RUN_NAME>/best_encoder.ckpt`
- `panorama_foundation/experiments/ae/runs/<RUN_NAME>/metrics.csv`
- `panorama_foundation/experiments/ae/runs/<RUN_NAME>/train_visuals/index.html`
- `panorama_foundation/experiments/ae/runs/<RUN_NAME>/train_visuals/manifest.jsonl`

## Inspecao visual durante treino

No fim de cada epoca (controlado por `visuals.interval` no config), o treino salva:

- `augmentation`: before/after augmentacao + diff
- `reconstruction`: input/target/reconstruction/erro
- `attention`: erro de reconstrucao vs atencao agregada (`mean` e `max`) em camadas intermediarias

Para abrir viewer local:

```bash
cd panorama_foundation/experiments/ae/runs/<RUN_NAME>/train_visuals
python3 -m http.server 8080
```

Depois abra `http://localhost:8080/index.html`.

## Smoke visual rapido

Para validar apenas o pipeline de artefatos visuais (poucas batches):

```bash
./panorama_foundation/scripts/run_ae_visual_smoke.sh --run-name visual_smoke_check
```

Para servir o viewer da run mais recente:

```bash
./panorama_foundation/scripts/serve_latest_visuals.sh 8080
```

## Pretext Modes (v1)

Configuravel em `pretext.mode` no JSON:

- `identity` (baseline atual)
- `denoise`
- `inpaint`
- `hybrid` (denoise + inpaint)

Run pronta da nova versao:

```bash
./panorama_foundation/scripts/run_ae_full999_hybrid.sh \
  --run-name AE_FULL999_RESNET34_NOSKIP_HYBRID_V1_MPS \
  --device mps
```
