# Catálogo de Artefatos de Testes, Avaliações e Métricas

Data de consolidação: **2026-03-27**  
Escopo: workspace `/Users/fabioandrade/hydra`

Este catálogo inventaria os artefatos produzidos por scripts de teste/avaliação/métricas/visualização, agrupando por trilha do projeto, com:
- o que foi medido/experimentado;
- qual script gera cada família de saída;
- objetivo e cuidados de interpretação;
- histórico de versões (runs/pastas) e indicação do mais novo.

## 1) Resumo Executivo do Inventário

Contagens detectadas no workspace (arquivos-chave):
- `longoeixo` Hydra principal (`eval/metrics_summary.json`): **9**
- MultiROI model eval (`eval_multiroi_model/metrics_summary.json`): **7**
- MultiROI presença hist (`eval_multiroi_presence_hist/summary.json`): **3**
- MultiROI presença por imagem (`per_image_presence_errors.csv`): **3**
- Qualitativos MultiROI (`qualitative_*/index.html`): **22**
- DAE eval (`dae_longoeixo/.../eval/metrics_summary.json`): **10**
- DAE per-sample (`dae_longoeixo/.../eval/per_sample_errors.csv`): **10**
- Foundation AE (`panorama_foundation/.../runs/*/summary.json`): **7**
- AE radiograph filter (`ae_radiograph_filter/outputs/**/summary.json`): **3**
- RM API tools audit (`radiomemory_api_tools/outputs/**/audit_summary.json`): **2**

## 2) Trilha Hydra Principal (U-Net multitarefa)

### 2.1 Scripts e propósito

- `train.py`
  - Objetivo: treino da Hydra multitarefa (heatmap + presença).
  - Saídas: `runs/<RUN>/metrics.csv`, `best.ckpt`, `last.ckpt`, `train_visuals/index.html`.
  - Cuidado: comparação de performance final deve usar `eval.py` no split oficial, não só `val_total_loss`.

- `eval.py`
  - Objetivo: avaliação oficial por split (`train/val/test/all`), com métricas de presença e localização.
  - Saídas: `eval/metrics_summary.json`, `metrics_per_tooth.csv`, `metrics_per_quadrant.csv`, `pred_vs_gt_samples/`.
  - Cuidado: política oficial de calibração de presença no `val` e aplicação congelada no `test`.

- `longoeixo/scripts/run_hydra_eval.sh`, `run_hydra_smoke.sh`, `run_hydra_train_local.sh`
  - Objetivo: atalhos operacionais de treino/eval/smoke.
  - Cuidado: rastreabilidade via `--run-name` explícito.

### 2.2 Artefatos e histórico (runs)

Raiz: `longoeixo/experiments/hydra_unet_multitask/runs/`  
Exemplos de versões detectadas (mais recente primeiro):
- `Full70_15_15_mseOnly_noPres_absentHM1_20260325_013241`
- `Full70_15_15_noPresenceHMfull`
- `RM_API_Longaxis_vs_Full70_valcal_test`
- `FifthTest999_absentHM0__cmp_heatmap_comp_cal`
- `FifthTest999_absentHM0__cmp_heatmap_cal`
- `FifthTest999_absentHM0__cmp_heatmap_comp_fixed01`
- `FifthTest999_absentHM0__cmp_heatmap_fixed01`
- `FifthTest999_absentHM0__cmp_logits`
- `FifthTest999_absentHM0`
- `FourthTest999_sigma52`

Mais novo nesta trilha (por mtime de artefato-chave):
- `longoeixo/experiments/hydra_unet_multitask/runs/Full70_15_15_mseOnly_noPres_absentHM1_20260325_013241/metrics.csv`

## 3) Trilha MultiROI (janelas fixas + lateral compartilhada)

Contrato oficial de inferência: `longoeixo/scripts/multiroi_composed_inference.py`  
Consumidores oficiais ativos no catálogo:
- `infer_multiroi_overlay_mosaic_lib.py`
- `eval.py` (modo `multiroi_model`)
- `eval_multiroi_presence_hist.py`
- `infer_multiroi_overlay_pred_gt_errors_lib.py`

### 3.1 Famílias de experimento/artefato

1. **Avaliação MultiROI (métricas completas)**
- Script: `eval.py` (modo `multiroi_model`).
- Saídas: `eval_multiroi_model/metrics_summary.json`, `metrics_per_tooth.csv`, `metrics_per_quadrant.csv`, `pred_vs_gt_samples/`.
- Histórico detectado:
  - `multiroi_eval_all999_thr01_normcols_v2` (mais novo)
  - `multiroi_eval_all999_thr01_center16kstable3`
  - `multiroi_eval_all999_thr005`
  - `multiroi_eval_all999_thr_calibrated`
  - `multiroi_eval_all999_thr01`

2. **Histograma de erro de presença por imagem**
- Script: `longoeixo/scripts/eval_multiroi_presence_hist.py`.
- Saídas: `eval_multiroi_presence_hist/summary.json`, `per_image_presence_errors.csv`, histogramas PNG/JSON/CSV.
- Histórico detectado:
  - `multiroi_presence_hist_all999_thr01_ic` (mais novo)
  - `multiroi_presence_hist_all999_thr01`
  - `smoke_multiroi_presence_hist_5`

3. **Mosaicos qualitativos MultiROI**
- Script principal: `longoeixo/scripts/infer_multiroi_overlay_mosaic_lib.py`.
- Saídas: `qualitative_*/index.html`, `summary.json`, `overlays/`, `predictions_json/`.
- Histórico de versões por pasta (mais novo primeiro):
  - `qualitative_multiroi_ge3_errors_fnfp_sorted` (mais novo geral da família)
  - `qualitative_multiroi_one_error_fnfp_cards`
  - `qualitative_multiroi_200_lib_newdefault_center16kstable3`
  - `qualitative_multiroi_pred_gt_labels_all_gt3_thr01`
  - `qualitative_multiroi_pred_gt_tpfpfn_all_gt3_thr01`
  - `qualitative_multiroi_pred_gt_errors_pm_gt2_thr01`
  - `qualitative_multiroi_pred_gt_errors_gt4_thr01`
  - `qualitative_multiroi_200_lib_newdefault`
  - `qualitative_multiroi_50_lib_newdefault`
  - `qualitative_multiroi_20260325_003514`
  - e demais variantes históricas `qualitative_multiroi_*`.

4. **Overlay Pred vs GT filtrado por erro**
- Script: `longoeixo/scripts/infer_multiroi_overlay_pred_gt_errors_lib.py`.
- Entrada: `per_image_presence_errors.csv`.
- Saídas: HTML + overlays com TP/FP/FN por exame filtrado.

5. **Análises de troca FN/FP por pares (novas)**
- Scripts:
  - `longoeixo/scripts/analyze_presence_swap_pairs.py`
  - `longoeixo/scripts/count_swap_pairs_from_hist_list.py`
- Objetivo: quantificar trocas em pares definidos (pré-molares adjacentes e 2º/3º molares), com base em erros de presença.
- Saídas: `swap_pairs_summary.json`, `swap_pairs_by_pair.csv`, `swap_pairs_by_exam.csv`.
- Cuidado: manter checkpoints/thresholds consistentes com o run de referência para evitar divergência de contagem.

### 3.2 Mais novos na trilha MultiROI

- Mais novo qualitativo:  
  `longoeixo/experiments/hydra_roi_fixed_shared_lateral/qualitative_multiroi_ge3_errors_fnfp_sorted/index.html`

- Mais novo eval MultiROI:  
  `longoeixo/experiments/hydra_roi_fixed_shared_lateral/center24_sharedflip_nopres_absenthm1/runs/multiroi_eval_all999_thr01_normcols_v2/eval_multiroi_model/metrics_summary.json`

- Mais novo histograma de presença:  
  `longoeixo/experiments/hydra_roi_fixed_shared_lateral/center24_sharedflip_nopres_absenthm1/runs/multiroi_presence_hist_all999_thr01_ic/eval_multiroi_presence_hist/summary.json`

## 4) Trilha DAE de Coordenadas (imputação)

### 4.1 Scripts e propósito

- `dae_longoeixo/train_dae.py`
  - Objetivo: treino do DAE de coordenadas.
  - Saídas: `metrics.csv`, `best.ckpt`, `last.ckpt`, `train_visuals/index.html`.

- `dae_longoeixo/eval_dae.py`
  - Objetivo: avaliação de imputação (erro por ponto/dente/amostra).
  - Saídas: `eval/metrics_summary.json`, `metrics_per_tooth.csv`, `per_sample_errors.csv`, `pred_vs_gt_samples/`.

- Wrappers: `dae_longoeixo/scripts/run_dae_train_local.sh`, `run_dae_eval.sh`, `run_dae_smoke.sh`, `run_dae_train_curves.sh`, `run_dae_eval_curves.sh`.

### 4.2 Histórico de versões detectado

- `coords_dae/runs/`: `DAE_142_FULL32`, `DAE_999_FIRST`, `dae_smoke_autotest`
- `coords_dae_partial/runs/`: `DAE_999_PARTIAL`, `DAE_999_PARTIAL_ARC`, `DAE_999_PARTIAL_ARC_ANCHOR`
- `coords_dae_curves/runs/`: `DAE_CURVES_TEETH_ONLY`
- Há também famílias smoke auxiliares: `coords_dae_smoke`, `coords_dae_partial_smoke`, `coords_dae_curves_smoke`, `coords_dae_curves_reconall_smoke`.

Mais novo da trilha:
- `dae_longoeixo/experiments/coords_dae_curves/runs/DAE_CURVES_TEETH_ONLY/metrics.csv`
- visual correlato: `.../imputation_on_real_missing/index.html`

## 5) Trilha Foundation AE (panorâmicas)

### 5.1 Scripts e propósito

- `panorama_foundation/train_autoencoder.py`
  - Objetivo: treino AE base/foundation.
  - Saídas: `runs/<RUN>/metrics.csv`, `summary.json`, `train_visuals/index.html`.

- `panorama_foundation/train_transfer_skeleton.py`
  - Objetivo: esqueleto de transferência.

- Wrappers:
  - `scripts/run_ae_visual_smoke.sh`
  - `scripts/run_ae_full999.sh`
  - `scripts/run_ae_full999_hybrid.sh`
  - `scripts/run_ae_local.sh`

### 5.2 Histórico de versões detectado

- `ae_full999/runs/`:
  - `AE_FULL999_RESNET34_NOSKIP_ALL_EPOCH_VIS_V2_MPS` (mais novo por mtime de pasta)
  - `AE_FULL999_RESNET34_NOSKIP_BASELINE_V2_60E_MPS`
  - `AE_FULL999_RESNET34_NOSKIP_BASELINE_V1`
  - `AE_FULL999_RESNET34_NOSKIP_BASELINE_V1_MPS`

- `ae_visual_smoke/runs/`:
  - `visual_smoke_identity_v1` (mais novo)
  - `visual_smoke_noskips`
  - `visual_smoke_check`

Mais novo da trilha (arquivo-chave):
- `panorama_foundation/experiments/ae_full999/runs/AE_FULL999_RESNET34_NOSKIP_BASELINE_V2_60E_MPS/summary.json`

## 6) Trilha AE Radiograph Filter

### 6.1 Scripts e propósito

- `ae_radiograph_filter/scripts/run_filter.py`
  - Objetivo: scoring/flag de radiografias por erro de reconstrução.
  - Saída: `summary.json`.

- `ae_radiograph_filter/scripts/run_batch_enhance_and_html.py`
  - Objetivo: geração de painéis e HTML (enhance + inspeção visual).
  - Saídas: `index.html`, `summary.json`.

- Wrappers: `run_sample_local.sh`, `run_batch100_local.sh`.

### 6.2 Histórico de saídas detectado

- `AE_FILTER_SMOKE20/summary.json`
- `AE_ENHANCE_SMOKE12/index.html`, `summary.json`
- `AE_ENHANCE_LOCAL100_FS03_FA03_TIMED/index.html`, `summary.json` (mais novo da trilha)

## 7) Trilha Radiomemory API Tools / Auth

### 7.1 API Tools (probes e auditorias)

Scripts:
- `radiomemory_api_tools/probe_panoramic_longaxis.py`
- `radiomemory_api_tools/audit_longoeixo_roi_partition.py`
- overlays/renderers (`panoramic_*_overlay.py`, `render_*`).

Artefatos detectados:
- `outputs/roi_partition_audit_smoke12/` (`audit_summary.json/csv`, `audit_details.json`)
- `outputs/roi_partition_audit_full999/` (`audit_summary.json/csv`, `audit_details.json`)
- relatórios de janelas seguras/fixas (`*report.json`)
- respostas de probe (`panorogram_simple_response_test.json`, etc.)

Mais novo detectado:
- `radiomemory_api_tools/outputs/panorogram_simple_response_test.json`

### 7.2 Auth / smoke API

Scripts:
- `radiomemory_auth/rm_ia_smoke_test.py`
- `radiomemory_auth/rm_ia_batch_runner.py`
- `radiomemory_auth/rm_ia_client.py`

Objetivo: validação de conectividade e contrato de endpoints RM API.

## 8) Auditorias de dataset e utilitários de diagnóstico

Scripts detectados:
- `audit_axis_inversion.py`
- `audit_morphology_suspects.py`
- `longoeixo/scripts/calc_dataset_horizontal_stats.py`
- `longoeixo/scripts/count_dentition_coverage.py`
- `visualize_presence_top_errors_overlay.py`

Situação no inventário atual:
- scripts presentes e documentados;
- nesta varredura não foram encontrados artefatos recentes dessas famílias no workspace ativo (fora dos que já estão embutidos em runs específicos).

## 9) Histórico de Versões (síntese)

### 9.1 Macro-linha do tempo observada

- **2026-03-20 ~ 2026-03-23**: consolidação Hydra principal + auditorias + DAE inicial + Foundation AE.
- **2026-03-24 ~ 2026-03-26**: forte iteração em MultiROI (evals, mosaicos, overlays GT/Pred, presença hist).
- **2026-03-27**: análises adicionais de presença/trocas FN-FP e mosaicos temáticos (`1 erro` e `>=3 erros`).

### 9.2 “Mais novo” por trilha (arquivo-âncora)

- Hydra principal:  
  `longoeixo/experiments/hydra_unet_multitask/runs/Full70_15_15_mseOnly_noPres_absentHM1_20260325_013241/metrics.csv`

- MultiROI qualitativo:  
  `longoeixo/experiments/hydra_roi_fixed_shared_lateral/qualitative_multiroi_ge3_errors_fnfp_sorted/index.html`

- MultiROI métricas:  
  `.../runs/multiroi_eval_all999_thr01_normcols_v2/eval_multiroi_model/metrics_summary.json`

- MultiROI presença hist:  
  `.../runs/multiroi_presence_hist_all999_thr01_ic/eval_multiroi_presence_hist/summary.json`

- DAE:  
  `dae_longoeixo/experiments/coords_dae_curves/runs/DAE_CURVES_TEETH_ONLY/metrics.csv`

- Foundation AE:  
  `panorama_foundation/experiments/ae_full999/runs/AE_FULL999_RESNET34_NOSKIP_BASELINE_V2_60E_MPS/summary.json`

- AE Filter:  
  `ae_radiograph_filter/outputs/AE_ENHANCE_LOCAL100_FS03_FA03_TIMED/summary.json`

- RM API Tools:  
  `radiomemory_api_tools/outputs/panorogram_simple_response_test.json`

## 10) Observações de Governança e Cuidados

1. `scripts antigos (NAO É PRA USAR!!!)/` deve permanecer fora do fluxo operacional; scripts ativos estão em raiz e `longoeixo/scripts`.
2. Para comparações quantitativas, manter consistência de:
   - checkpoints (`center/lateral`),
   - thresholds de presença,
   - split avaliado.
3. Quando comparar contagens de “pares trocados”, distinguir:
   - contagem de **ocorrências de pares** (um exame pode contribuir em múltiplos pares),
   - contagem de **exames únicos**.
4. `docs/avaliacao/METRICS_SCRIPTS_INDEX.md` cita `eval_presence_top_errors.py` na raiz; hoje esse script está apenas em área arquivada e o fluxo foi coberto por scripts MultiROI atuais. Recomendado atualizar o índice.

---

## Apêndice A) Scripts-chave por tipo

- Avaliação principal: `eval.py`, `dae_longoeixo/eval_dae.py`
- Smoke/eval wrappers: `longoeixo/scripts/run_hydra_*.sh`, `dae_longoeixo/scripts/run_dae_*.sh`
- MultiROI inferência oficial: `longoeixo/scripts/multiroi_composed_inference.py`
- MultiROI qualitativo: `infer_multiroi_overlay_mosaic_lib.py`, `infer_multiroi_overlay_pred_gt_errors_lib.py`
- MultiROI presença/hist: `eval_multiroi_presence_hist.py`
- Análise de pares FN/FP: `analyze_presence_swap_pairs.py`, `count_swap_pairs_from_hist_list.py`
- Auditoria dataset: `audit_axis_inversion.py`, `audit_morphology_suspects.py`
- Dataset stats: `calc_dataset_horizontal_stats.py`, `count_dentition_coverage.py`
- AE filter: `ae_radiograph_filter/scripts/run_filter.py`, `run_batch_enhance_and_html.py`
- RM API tools: `probe_panoramic_longaxis.py`, `audit_longoeixo_roi_partition.py`
