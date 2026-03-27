# HYDRA Training Runbook (Local -> EC2)

Guia operacional para treino do modelo Hydra U-Net MultiTask com dataset pequeno local (100) e escala para EC2 (16k), mantendo mesma árvore de diretórios.

## 1) Estrutura esperada de pastas

Na raiz do repo:

- `longoeixo/imgs`
- `longoeixo/data_longoeixo`

Na EC2, a estrutura deve ser idêntica.

## 2) Pipeline de dados (decisão oficial)

1. Usar `longoeixo/imgs` + `longoeixo/data_longoeixo` como fonte canônica.
2. Gerar `Y_heatmap` on-the-fly por amostra (em memória) a partir do JSON.
3. Projetar pontos para `256x256` e gerar gaussianas diretamente no grid final de treino.
4. Derivar `Y_presence` no mesmo pipeline (JSON ou pares de canais).

Motivo:
- evita pré-armazenamento massivo de `stack64` original.
- evita variação artificial de largura gaussiana entre exames com resoluções originais diferentes.
- mantém fidelidade geométrica sem custo de storage inviável para escala.

Regra prática:
- parametrizar `sigma` em pixels do target final (`sigma_px_target`), não em pixels absolutos da imagem original.

## 3) Augmentações (decisão oficial)

### Proibido
- `horizontal flip`
- `vertical flip`

Risco: quebra lateralidade/semântica dos canais odontológicos.

### Permitido (leve)
- rotação pequena
- scale leve
- translação leve
- jitter de brilho/contraste (apenas imagem)
- ruído leve (apenas imagem):
  - gaussiano aditivo fraco
  - poisson leve
  - speckle leve

Regra: transformações geométricas sempre iguais em imagem e máscara.

## 4) Preset e parâmetros iniciais

Arquivo: `longoeixo/presets/unet256_stack64_preset.json`

Valores base:
- entrada: `1x256x256`
- target: `64x256x256`
- optimizer: `AdamW`
- learning rate: `3e-4`
- weight decay: `1e-4`
- batch size inicial: `8`
- epochs: `120`
- early stopping patience: `20`
- amp: `true`
- loss: `0.8*MSE + 0.2*SoftDice` (heatmap)
- multitask: adicionar `BCEWithLogits` para presença (32 saídas)

## 5) Setup portátil (local/EC2)

Na raiz do repo:

```bash
bash longoeixo/setup_env.sh
```

## 6) Geração de targets stack64 (opcional)

Use apenas para inspeção/debug/smoke.

Todos os exemplos:

```bash
bash longoeixo/run_generate_stack64.sh longoeixo/gaussian_maps_stack64
```

Smoke test:

```bash
bash longoeixo/run_generate_stack64.sh longoeixo/gaussian_maps_stack64_smoke 10
```

## 7) Preprocess 256 de validação (1 amostra)

```bash
.venv/bin/python longoeixo/scripts/unet256_preprocess.py \
  --image longoeixo/imgs/ararasradiodonto.radiomemory.com.br_373829_11994.jpg \
  --mask longoeixo/gaussian_maps_stack64/ararasradiodonto.radiomemory.com.br_373829_11994.npy \
  --out-x /tmp/x_256.npy \
  --out-y /tmp/y_256.npy \
  --out-overlay /tmp/overlay_256.png
```

## 8) Plano de evolução do modelo

### Fase A - protótipo (100 casos)
- validar dataset/dataloader/loss/logs
- medir overfit controlado
- ajustar sigma/loss weights

### Fase B - escala (16k na EC2)
- subir para backbone ResNet34
- treino completo com early stopping
- monitorar métricas por dente e por quadrante

## 9) Checklist antes de iniciar treino longo

1. Garantir shape de target `(64,256,256)` no batch final do DataLoader.
2. Garantir `Y_presence` com shape `(32,)` consistente com ordem canônica.
3. Confirmar ausência de flip no pipeline.
4. Validar visualmente overlays em lote pequeno.
5. Configurar paralelismo do DataLoader (`num_workers`, prefetch, workers persistentes).
6. Salvar configuração de experimento (seed, versão de preset, commit).

## 10) Convenções para reproducibilidade

- Fixar seeds (Python/Numpy/framework).
- Versionar preset usado por experimento.
- Salvar:
  - hiperparâmetros
  - métricas por época
  - melhor checkpoint
  - data split usado

## 10.1 Paralelismo recomendado (on-the-fly)

Para reduzir gargalo de geração de target durante treino:
- usar `DataLoader` com múltiplos workers (processos).
- habilitar prefetch para manter batches prontos.
- usar workers persistentes para evitar custo de spawn a cada época.

Modelo mental:
- CPU prepara próximos batches (incluindo target on-the-fly) enquanto acelerador treina no batch atual.

## 11) Inspecao visual antes/depois da augmentacao

Gerar paineis com overlay (imagem + max das 64 gaussianas) antes e depois da augmentacao:

```bash
# exemplo com 10 amostras
.venv/bin/python longoeixo/scripts/capture_aug_samples.py \
  --masks-dir longoeixo/gaussian_maps_stack64_10 \
  --out-dir longoeixo/aug_inspection_10 \
  --num-samples 10
```

Saida: imagens lado a lado (Before/After) para auditoria visual de consistencia do target.

Nota (modo on-the-fly):
- manter essa auditoria durante o treino com baixa frequência (ex.: a cada 5 ou 10 épocas) para reduzir custo de I/O e ainda garantir sanidade da geração.

## 12) Captura automatica por epoca (augmentacao + atencao)

Modulo: `hydra_training_callbacks.py`

Comportamento atual no `train.py`:
- no fim de cada epoca, o treino pega um mini-batch de debug do `train_loader`;
- chama `capture_epoch_visuals(...)` com:
  - `x_before/y_before` (antes da augmentacao),
  - `x_after/y_after` (apos augmentacao),
  - `model` (para extrair features intermediarias e gerar attention maps).

Saida por run (atual):
- `longoeixo/experiments/hydra_unet_multitask/runs/<RUN_NAME>/train_visuals/epoch_XXXX/augmentation/*.png`
- `longoeixo/experiments/hydra_unet_multitask/runs/<RUN_NAME>/train_visuals/epoch_XXXX/attention/*.png`
- `longoeixo/experiments/hydra_unet_multitask/runs/<RUN_NAME>/train_visuals/manifest.jsonl`
- `longoeixo/experiments/hydra_unet_multitask/runs/<RUN_NAME>/train_visuals/index.html`

Detalhes de visual:
- `augmentation`: painel lado a lado (`Before Aug` vs `After Aug`) com overlay `img + max(mask64)` em vermelho.
- `attention`: painel lado a lado (`GT Overlay` vs `Attention`) por camada e por modo de agregacao (`mean` e `max`).

Exemplo de chamada (referencia):

```python
from hydra_training_callbacks import capture_epoch_visuals

# x_before/y_before: batch antes da augmentacao
# x_after/y_after: batch apos a augmentacao
capture_epoch_visuals(
    out_dir=run_dir / "train_visuals",
    epoch=epoch,
    model=model,
    x_before=x_before,
    y_before=y_before,
    x_after=x_after,
    y_after=y_after,
    interval=5,      # recomendado em on-the-fly (ex.: 5 ou 10)
    max_samples=8,
)
```

## 13) Monitoramento de treino: TensorBoard + HTML Viewer

Objetivo operacional:
- usar TensorBoard para curvas escalares (loss/lr por epoca);
- usar o HTML Viewer para auditoria visual (augmentacao e attention por amostra/camada).

### 13.1 TensorBoard (curvas)

Suba o TensorBoard apontando para a pasta de runs:

```bash
cd /caminho/do/repo/hydra
.venv/bin/tensorboard \
  --logdir longoeixo/experiments/hydra_unet_multitask/runs \
  --host 127.0.0.1 \
  --port 6006
```

Com tunel SSH:

```bash
ssh -N -L 6006:127.0.0.1:6006 <usuario>@<ec2-host>
```

Abra:
- `http://localhost:6006`

### 13.2 Viewer HTML dinamico na EC2

Agora o callback grava automaticamente:
- `index.html` (viewer)
- `manifest.jsonl` (indice de artefatos)
- artefatos por epoca em `epoch_XXXX/...`

No loop de treino, use `capture_epoch_visuals(...)`.

Para abrir na EC2:

```bash
# na EC2, dentro da pasta de artefatos
cd /caminho/do/repo/hydra/longoeixo/experiments/hydra_unet_multitask/runs/<RUN_NAME>/train_visuals
python3 -m http.server 8080
```

Do seu computador local (tunel SSH):

```bash
ssh -N -L 8080:localhost:8080 <usuario>@<ec2-host>
```

Depois abra:
- `http://localhost:8080/index.html`

O viewer permite filtrar por: epoca, amostra, grupo, modo e camada.

## 14) Passo a passo para rodar experimento com hiperparametro/opcao diferente

Objetivo: criar uma configuracao nova, rodar treino e comparar sem sobrescrever um experimento anterior.

### 14.1 Onde estao os scripts

- treino local: `longoeixo/scripts/run_hydra_train_local.sh`
- avaliacao rapida (config padrao): `longoeixo/scripts/run_hydra_eval.sh`
- treino direto com config escolhida: `train.py`
- avaliacao direta com config escolhida: `eval.py`

### 14.2 Crie uma config nova a partir de uma base

Exemplo (base oficial 70/15/15):

```bash
cp hydra_train_config_full70_15_15.json hydra_train_config_myexp.json
```

Edite `hydra_train_config_myexp.json` e altere o que deseja testar, por exemplo:
- `training.lr`
- `training.batch_size`
- `training.lambda_presence`
- `training.absent_heatmap_weight`
- `training.num_workers`

### 14.3 Rode o treino com nome de run explicito

```bash
.venv/bin/python train.py \
  --config hydra_train_config_myexp.json \
  --run-name MyExp_lr1e4_bs8
```

Isso cria artefatos em:
- `longoeixo/experiments/hydra_unet_multitask/runs/MyExp_lr1e4_bs8/`

### 14.4 Acompanhe durante o treino

Metricas por epoca:

```bash
tail -f longoeixo/experiments/hydra_unet_multitask/runs/MyExp_lr1e4_bs8/metrics.csv
```

Viewer visual da run:

```bash
cd longoeixo/experiments/hydra_unet_multitask/runs/MyExp_lr1e4_bs8/train_visuals
python3 -m http.server 8080
```

Abra:
- `http://localhost:8080/index.html`

### 14.5 Avalie a run com a mesma config

```bash
.venv/bin/python eval.py \
  --config hydra_train_config_myexp.json \
  --run-name MyExp_lr1e4_bs8 \
  --split val
```

Saidas esperadas:
- `.../runs/MyExp_lr1e4_bs8/eval/summary.json`
- `.../runs/MyExp_lr1e4_bs8/eval/metrics_per_tooth.csv`

### 14.6 Opcional: usar script rapido padrao

Se voce quer apenas um treino com a config default (`hydra_train_config.json`):

```bash
bash longoeixo/scripts/run_hydra_train_local.sh
```
