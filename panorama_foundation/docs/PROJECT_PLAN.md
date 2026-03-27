# Panorama Autoencoder Foundation - Plano de Execucao

## 1) Objetivos

1. Treinar um autoencoder de imagem para radiografia panoramica com:
   - encoder `ResNet34` (mesma linha do Hydra/longoeixo)
   - decoder compativel para reconstrucao da panoramica, **sem skip-connections**
2. Reaproveitar o encoder pretreinado em tarefas supervisionadas:
   - classificacao
   - regressao
   - segmentacao (encoder + decoder U-Net com skips)
3. Padronizar treino em duas fases downstream:
   - fase A: encoder congelado
   - fase B: fine-tuning completo (encoder descongelado)

## 2) Escopo da Fase Atual

Fase atual (esta entrega):

- pasta dedicada do projeto
- documentacao tecnica do plano
- esqueleto executavel do pretreino AE local
- esqueleto de transferencia (class/reg/seg) com estrategia freeze/unfreeze

Fase seguinte (proxima iteracao):

- rodar primeiro experimento local pequeno
- validar curva de reconstrucao e overfit controlado
- escolher checkpoint base para transferencia

## 3) Abordagem Tecnica

## 3.1 Autoencoder de Reconstrucao

- entrada: imagem panoramica grayscale `1x256x256`
- encoder: `ResNetEncoder(variant="resnet34")`
- decoder: upsampling convolucional **sem concat de skips**
- saida: `recon_logits (B,1,H,W)` e `recon = sigmoid(recon_logits)`
- loss inicial: combinacao `L1 + MSE`

Racional:

- `ResNet34` ja validada no ecossistema local
- sem skips evita atalho de copia direta e forca uso do bottleneck
- loss simples/estavel para primeiro ciclo

## 3.2 Transfer Learning

Partindo do encoder pretreinado:

- classificacao/regressao:
  - `encoder -> GAP -> FC`
- segmentacao:
  - `encoder -> decoder U-Net -> seg head`

Regime de treino downstream:

1. Warm-up com encoder congelado (heads aprendem rapido).
2. Fine-tuning completo com LR menor no encoder (discriminative LR).

## 4) Especificacoes

## 4.1 Dados

- fonte: pasta de imagens panoramicas (`jpg/jpeg/png/bmp/tif/tiff`)
- split: reprodutivel com JSON salvo em disco
- preprocess:
  - grayscale
  - resize para `256x256`
  - normalizacao para `[0,1]`
- augmentacao de treino (leve):
  - `ShiftScaleRotate`
  - `RandomBrightnessContrast`
  - ruido gaussiano leve
- proibido por default: flip horizontal/vertical

## 4.2 Modelo

- backbone principal: `resnet34`
- fallback rapido: `resnet18`
- API base:
  - `PanoramicResNetAutoencoder`
  - `PanoramicEncoderClassifier`
  - `PanoramicEncoderRegressor`
  - `PanoramicUNetSegmenter`

## 4.3 Treino

- otimizador: `AdamW`
- scheduler: `CosineAnnealingLR`
- AMP: habilitado apenas em CUDA
- checkpoints:
  - `best.ckpt`
  - `last.ckpt`
  - `best_encoder.ckpt` (somente encoder)
- logs:
  - `metrics.csv`
  - TensorBoard opcional
  - `train_visuals/manifest.jsonl + index.html`
- early stopping por `val_loss`

## 4.4 Protocolo Downstream (recomendado)

1. Epochs 1..N1:
   - `freeze_encoder()`
   - treinar somente decoder/head
2. Epochs N1+1..N2:
   - `unfreeze_encoder()`
   - reduzir LR do encoder
   - manter LR maior no head/decoder

## 5) Entregaveis desta fase

1. Estrutura `panorama_foundation/`.
2. Documentacao tecnica com plano e specs.
3. Script de treino local do AE com config de smoke.
4. Esqueleto de transferencia pronto para implementacao incremental.
5. Callback visual por epoca no padrao Hydra (augmentacao, reconstrucao, atencao).

## 6) Critérios de Pronto (fase atual)

1. Projeto roda localmente com dataset pequeno.
2. Gera pasta de run com checkpoints e `metrics.csv`.
3. Exporta checkpoint do encoder para reutilizacao downstream.
