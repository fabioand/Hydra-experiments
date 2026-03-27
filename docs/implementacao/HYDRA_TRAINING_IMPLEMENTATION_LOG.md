# HYDRA Training Implementation Log

Data de consolidação: 2026-03-20

## 1) Escopo implementado

Foi implementado o pipeline local end-to-end do Hydra U-Net MultiTask, cobrindo:

1. Dataset + DataLoader (JPG + stack64 NPY)
2. Split reprodutível com persistência
3. Treino (`train.py`) com checkpoints e logging
4. Capturas visuais por época (augmentação + atenção)
5. Avaliação (`eval.py`) com artefatos do plano
6. Organização por run/experimento com nome manual ou timestamp

## 2) Arquivos implementados/alterados

### Novos arquivos
- `hydra_data.py`
- `train.py`
- `eval.py`
- `hydra_train_config.json`
- `longoeixo/scripts/run_hydra_smoke.sh`
- `longoeixo/scripts/run_hydra_train_local.sh`
- `longoeixo/scripts/run_hydra_eval.sh`

### Artefatos gerados em execução
- `longoeixo/splits.json`
- `longoeixo/experiments/...`

## 3) Decisões de abordagem

### 3.1 Ground truth stack64 pré-gerado (offline)
Decisão mantida conforme os documentos do projeto:
- geração de gaussianas no tamanho original via `generate_gaussian_point_maps.py`
- treino consome `.npy` stack64 + JPG
- resize para `256x256` é feito no DataLoader

Motivos:
- maior reprodutibilidade
- menor custo no loop de treino
- compatibilidade direta local/EC2 com a mesma árvore

### 3.2 Presença derivada do stack64
`Y_presence` é derivado por regra fixa:
- dente `i` usa canais `2*i` e `2*i+1`
- presença = 1 se qualquer um desses canais tiver pico > `eps`

### 3.3 Augmentação
Implementada conforme preset:
- sem flip horizontal/vertical
- transformação geométrica aplicada em imagem + máscara
- intensidade/ruído aplicada apenas na imagem

### 3.4 Organização de experimentos por run
Foi adotado modelo por execução:
- `--run-name` opcional em `train.py`
- fallback automático para timestamp (`YYYY-MM-DD_HH-MM-SS`)
- cada run grava seus próprios artefatos

Estrutura:
- `.../experiments/<exp>/runs/<run_name>/best.ckpt`
- `.../experiments/<exp>/runs/<run_name>/last.ckpt`
- `.../experiments/<exp>/runs/<run_name>/metrics.csv`
- `.../experiments/<exp>/runs/<run_name>/tensorboard/`
- `.../experiments/<exp>/runs/<run_name>/train_visuals/`
- `.../experiments/<exp>/runs/<run_name>/eval/`

Além disso:
- `latest_run.txt` em `.../experiments/<exp>/` para facilitar avaliação da run mais recente.

## 4) Decisões de plataforma (Local/EC2)

### 4.1 Compatibilidade
Sem caminhos hardcoded de máquina no código do pipeline.
Todos os caminhos vêm do config e são resolvidos relativos à raiz do repo.

### 4.2 Seleção de device
`device=auto` implementado em `train.py` e `eval.py` com prioridade:
1. `cuda`
2. `mps`
3. `cpu`

Isso mantém:
- EC2/NVIDIA: CUDA automático
- Mac Apple Silicon: MPS automático quando disponível

### 4.3 Observação sobre ambiente Codex vs terminal local
No ambiente de execução do agente (sandbox), MPS apareceu indisponível.
No terminal local do usuário, MPS foi validado como disponível e funcional (`mps:0`).

## 5) Validações executadas

### 5.1 Smoke test end-to-end
Executado com treino curto + avaliação.
Validados:
- shapes corretos de tensor
- geração de checkpoints
- logging CSV e TensorBoard
- visualização por época (`train_visuals`)
- outputs de avaliação

### 5.2 Coerência de ordem de canais e presença
Checagens realizadas:
- `channel_order_64.txt` compatível com ordem canônica
- ordem canônica do gerador e do modelo idênticas
- presença derivada dos canais coerente com JSON de origem em amostras verificadas

Resultado:
- sem inconsistências encontradas nas checagens realizadas

## 6) Comandos operacionais atuais

### Treino local (run nomeada)
```bash
./longoeixo/scripts/run_hydra_train_local.sh FirstTest100
```

### Treino local (run com timestamp)
```bash
./longoeixo/scripts/run_hydra_train_local.sh
```

### Eval de run específica
```bash
./longoeixo/scripts/run_hydra_eval.sh FirstTest100
```

### Smoke (treino + eval na mesma run)
```bash
./longoeixo/scripts/run_hydra_smoke.sh smoke_demo
```

## 7) Pendências e evolução sugerida

1. Opcional: modo híbrido de cache lazy (`gera stack64 se faltar`) sem perder compatibilidade com árvore original.
2. Ajustar deprecações de AMP (`torch.cuda.amp` -> API nova do PyTorch).
3. Escalar para 1000+ amostras para reduzir variância da head de presença.
