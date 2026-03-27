# DAE de Coordenadas de Long-Eixo

Este documento consolida o planejamento e a implementação do pipeline de **denoising autoencoder** para completar coordenadas de dentes faltantes a partir de entradas com knockout de dentes.

## 1) Objetivo

Treinar uma rede que:

- recebe um vetor de coordenadas de long-eixo (32 dentes, 2 pontos por dente)
- durante treino, recebe a entrada corrompida com alguns dentes nocauteados (`(0,0)` para os 2 pontos do dente)
- prediz o vetor completo de coordenadas (target sempre completo)

## 2) Alinhamento com o estado atual do projeto

Este pipeline foi desenhado no mesmo padrão operacional da U-Net Hydra:

- split reprodutível salvo em JSON
- treino com `best.ckpt` e `last.ckpt`
- logs em CSV + TensorBoard
- avaliação com artefatos (`metrics_summary.json`, CSV por dente, ranking por amostra)
- callbacks visuais por época
- scripts `run_*` para treino, eval e smoke

## 3) Regra de dados

Amostras elegíveis:

- apenas exames com dentição completa no padrão canônico
- 32 dentes (`11..18, 21..28, 31..38, 41..48`)
- 2 pontos válidos por dente

Representação:

- cada amostra vira vetor `float32` com 128 valores (`32 dentes x 2 pontos x (x,y)`)
- normalização por imagem de origem: `x/(W-1)` e `y/(H-1)`
- faixa final: `[0,1]`

### 3.1 Modo opcional de supervisão parcial (novo)

Agora o pipeline também suporta treino com dataset incompleto (ex.: 999 locais, 16k na EC2):

- `sample_filter=any_with_min_teeth` inclui qualquer exame com pelo menos `min_teeth_present`.
- dentes sem anotação válida entram com:
  - coordenadas placeholder `(0,0),(0,0)`,
  - `gt_available_mask=0`.
- a loss ignora automaticamente dentes sem GT (`mask` de supervisão).

Configuração atual em `dae_train_config.json`:

- `data.sample_filter = any_with_min_teeth`
- `data.min_teeth_present = 1`

## 4) Estratégia de corrupção (denoising)

No `Dataset`:

- sorteia subconjunto de dentes para knockout
- para cada dente nocauteado: zera os 4 valores (`x1,y1,x2,y2`)
- `Y` permanece sempre com coordenadas completas

Preset atual (`dae_coords_mlp_v1`):

- treino: knockout entre 2 e 10 dentes
- validação/eval: mesma faixa, porém determinístico
- augmentação extra de treino: jitter horizontal leve por dente, com reforço em vizinhos dos dentes nocauteados (simula pequenas migrações)

## 5) Arquitetura da rede

Modelo implementado: `CoordinateDenoisingAutoencoder` (MLP)

- entrada: `128` coords corrompidas + máscara opcional de pontos preservados (`+64`) -> `192`
- encoder MLP: `512 -> 256 -> latent(128)`
- decoder MLP simétrico
- saída: `128` coordenadas completas
- ativação de saída: `sigmoid` (mantém faixa `[0,1]`)

Loss implementada: `DaeImputationLoss`

- `total = w_knocked * MSE_knocked + w_observed * MSE_observed + w_all * MSE_all`
- pesos iniciais: `w_knocked=0.85`, `w_observed=0.15`, `w_all=0.0`
- regularização opcional de arco por centros: `w_arc_spacing * L_arc_spacing`
  - `L_arc_spacing` penaliza variação entre distâncias adjacentes de centros vizinhos
  - aplicado por quadrante, com máscara de GT disponível
- regularização opcional ancorada em GT observado: `w_anchor_rel * L_anchor_rel`
  - para dentes nocauteados, compara relação vetorial com vizinhos observados em GT
  - usa vizinhos imediatos no quadrante, sem depender do vizinho predito

## 6) Monitoramento e acompanhamento

Treino (`train_dae.py`) grava:

- `metrics.csv` por época
- TensorBoard (`runs/<run>/tensorboard`)
- checkpoints: `best.ckpt`, `last.ckpt`
- visuais de imputação por época em `train_visuals/`

Escalares no TensorBoard:

- `loss/train_total`, `loss/val_total`
- `mse/train_knocked`, `mse/val_knocked`
- `mse/train_observed`, `mse/val_observed`
- `mae/train_knocked`, `mae/val_knocked`
- `point_dist/train_knocked_px`, `point_dist/val_knocked_px`
- `lr`

## 7) Avaliação

Script: `eval_dae.py`

Saídas:

- `metrics_summary.json`
- `metrics_per_tooth.csv`
- `per_sample_errors.csv`
- `pred_vs_gt_samples/` (painéis visuais)

Métricas principais:

- MSE/MAE global
- MSE/MAE em coordenadas nocauteadas
- erro de ponto em pixels (média/mediana/P90)
- taxa dentro de tolerância (`<=3px`, `<=5px`, `<=10px`) nos pontos nocauteados
- métricas por dente (all vs knocked)

## 8) Estrutura da pasta

```text
dae_longoeixo/
  __init__.py
  dae_data.py
  dae_model.py
  dae_visuals.py
  train_dae.py
  eval_dae.py
  dae_train_config.json
  presets/
    dae_coords_mlp_v1.json
  scripts/
    run_dae_train_local.sh
    run_dae_eval.sh
    run_dae_smoke.sh
```

## 9) Execução

Treino padrão:

```bash
./dae_longoeixo/scripts/run_dae_train_local.sh
```

Treino nomeado:

```bash
./dae_longoeixo/scripts/run_dae_train_local.sh DAE_Full_001
```

Treino com limite de amostras:

```bash
./dae_longoeixo/scripts/run_dae_train_local.sh DAE_Subset_300 300
```

Treino local usando as 999 disponíveis (modo parcial):

```bash
./dae_longoeixo/scripts/run_dae_train_local.sh DAE_999_PARTIAL 999
```

Avaliação da run mais recente:

```bash
./dae_longoeixo/scripts/run_dae_eval.sh
```

Avaliação de run específica:

```bash
./dae_longoeixo/scripts/run_dae_eval.sh DAE_Full_001 val
```

Smoke test completo:

```bash
./dae_longoeixo/scripts/run_dae_smoke.sh dae_smoke_local
```

## 10) Próximas evoluções sugeridas

1. Expandir para múltiplos níveis de corrupção por batch (misturar leve/médio/severo).
2. Adicionar codificação de geometria global (ex.: centro/escala da arcada) como feature auxiliar.
3. Testar variante com Transformer MLP-Mixer para capturar relações de longo alcance entre dentes.
4. Integrar calibragem de incerteza por dente para priorizar revisão humana nos casos críticos.
