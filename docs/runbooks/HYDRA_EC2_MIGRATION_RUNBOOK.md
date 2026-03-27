# HYDRA EC2 Migration Runbook

Plano operacional para:
- primeira migração do treino Hydra para EC2;
- atualizações recorrentes (código/config);
- monitoramento remoto no navegador local.

Este guia assume que a EC2 já possui a árvore equivalente:
- `longoeixo/imgs`
- `longoeixo/data_longoeixo`

## 1) Objetivo operacional

Rodar treino no terminal da EC2 com persistência (mesmo após desconexão SSH) e acompanhar progresso do seu navegador local em tempo real.

## 2) Padrão de execução recomendado

- Treino: `tmux` (ou `screen`) na EC2.
- Logs:
  - métricas por época em `metrics.csv`;
  - TensorBoard em `.../runs/<run>/tensorboard`.
- Visuals por época: `train_visuals` com `index.html`.
- Monitoramento local via túnel SSH:
  - TensorBoard (`localhost:6006`);
  - Viewer HTML (`localhost:8080`).

## 3) Primeira implantação (one-time)

## 3.1 Preparar código na EC2

No servidor:

```bash
cd /caminho/do/repo/hydra
git fetch --all
git checkout <branch-ou-tag>
```

## 3.2 Preparar ambiente Python

```bash
cd /caminho/do/repo/hydra
bash longoeixo/setup_env.sh
```

## 3.3 Sanidade mínima de dados

```bash
cd /caminho/do/repo/hydra
ls longoeixo/imgs | head
ls longoeixo/data_longoeixo | head
```

## 3.4 Sanidade de config da run

Exemplo (Fifth com heatmap ausente mascarado):

```bash
cd /caminho/do/repo/hydra
cat hydra_train_config_fifth999_absenthm0.json
```

Conferir principalmente:
- `paths.imgs_dir`
- `paths.json_dir`
- `paths.splits_path`
- `paths.output_dir`
- `training.absent_heatmap_weight`

## 4) Execução do treino na EC2

## 4.1 Subir sessão tmux

```bash
cd /caminho/do/repo/hydra
tmux new -s hydra
```

## 4.2 Rodar treino

```bash
cd /caminho/do/repo/hydra
.venv/bin/python train.py \
  --config hydra_train_config_fifth999_absenthm0.json \
  --run-name FifthTest999_absentHM0
```

Atalhos `tmux`:
- destacar: `Ctrl+b` depois `d`
- voltar: `tmux attach -t hydra`

## 5) Monitoramento no navegador local

## 5.1 TensorBoard (recomendado para curvas)

Na EC2:

```bash
cd /caminho/do/repo/hydra
.venv/bin/tensorboard \
  --logdir longoeixo/experiments/hydra_unet_multitask/runs \
  --host 127.0.0.1 \
  --port 6006
```

No seu computador local (novo terminal):

```bash
ssh -N -L 6006:127.0.0.1:6006 <usuario>@<ec2-host>
```

Abrir no navegador local:
- `http://localhost:6006`

## 5.2 Viewer de imagens por época (augmentação/atenção)

Na EC2:

```bash
cd /caminho/do/repo/hydra/longoeixo/experiments/hydra_unet_multitask/runs/FifthTest999_absentHM0/train_visuals
python3 -m http.server 8080 --bind 127.0.0.1
```

No seu computador local:

```bash
ssh -N -L 8080:127.0.0.1:8080 <usuario>@<ec2-host>
```

Abrir no navegador local:
- `http://localhost:8080/index.html`

## 6) Acompanhamento por terminal (rápido)

No servidor:

```bash
cd /caminho/do/repo/hydra
tail -f longoeixo/experiments/hydra_unet_multitask/runs/FifthTest999_absentHM0/metrics.csv
```

## 7) Pós-treino e avaliação

Rodar avaliação da run treinada:

```bash
cd /caminho/do/repo/hydra
.venv/bin/python eval.py \
  --config hydra_train_config_fifth999_absenthm0.json \
  --run-name FifthTest999_absentHM0
```

Artefatos esperados:
- `.../runs/FifthTest999_absentHM0/best.ckpt`
- `.../runs/FifthTest999_absentHM0/metrics.csv`
- `.../runs/FifthTest999_absentHM0/eval/summary.json`
- `.../runs/FifthTest999_absentHM0/eval/metrics_per_tooth.csv`

## 8) Rotina de atualizações (repeatable)

Para cada nova atualização de código/config:

1. Encerrar treino atual com segurança (ou manter em sessão separada).
2. `git fetch --all` + `git checkout <branch/tag/commit>`.
3. Atualizar dependências se necessário:
```bash
cd /caminho/do/repo/hydra
bash longoeixo/setup_env.sh
```
4. Criar novo `--run-name` (nunca sobrescrever run anterior).
5. Rodar treino e monitorar novamente via túnel.

## 9) Convenções para não perder histórico

- Uma run, um nome único.
- Não reutilizar o mesmo `run-name` com config diferente.
- Salvar sempre:
  - config usada;
  - commit/branch;
  - métrica final principal (ex.: `presence_f1_macro`);
  - decisão (promover/descartar experimento).

## 10) Troubleshooting rápido

- TensorBoard não abre:
  - confirmar processo ativo na EC2 e porta 6006;
  - refazer túnel SSH;
  - testar `--host 127.0.0.1`.

- Viewer não abre:
  - confirmar `python3 -m http.server` no `train_visuals` da run:
    `longoeixo/experiments/hydra_unet_multitask/runs/<RUN_NAME>/train_visuals`;
  - garantir que `index.html` existe.

- Treino parou ao fechar SSH:
  - execute sempre dentro de `tmux`.

- GPU não está sendo usada:
  - validar logs `[DEVICE] using cuda`;
  - checar driver/CUDA na AMI.

## 11) Checklist curto (copiar e usar)

```bash
cd /caminho/do/repo/hydra
git fetch --all
git checkout <branch-ou-tag>
bash longoeixo/setup_env.sh
tmux new -s hydra
.venv/bin/python train.py --config hydra_train_config_fifth999_absenthm0.json --run-name FifthTest999_absentHM0
```

Em paralelo, para monitorar localmente:

```bash
# tunnel TensorBoard
ssh -N -L 6006:127.0.0.1:6006 <usuario>@<ec2-host>

# tunnel viewer
ssh -N -L 8080:127.0.0.1:8080 <usuario>@<ec2-host>
```
