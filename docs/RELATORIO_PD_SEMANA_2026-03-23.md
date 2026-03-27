# Relatório de P&D da Semana (Hydra Workspace)

Período analisado: **2026-03-20 a 2026-03-23**.
Base de evidência: documentação do repositório, artefatos de runs e timestamps/commits.

## 1) Resumo executivo

Nesta semana o laboratório evoluiu em 4 trilhas principais:

1. **Hydra Longo-Eixo (multitarefa)**: pipeline E2E foi consolidado (treino/eval/visualização), escalado de smoke/100 para ~999 casos, com melhoria contínua de `val_total_loss` e experimentos de presença/gating.
2. **Auditoria e qualidade de anotação**: scripts e plano robusto para detectar inconsistências (presença, inversão de eixo, morfologia, cascata), além de plano de dashboards.
3. **Foundation Autoencoder (panorâmicas)**: projeto de pretreino foi estruturado e executado com múltiplas runs (baseline/full999/híbrido), gerando checkpoints de encoder para transferência.
4. **DAE de coordenadas + ferramentas RM API**: pipeline de imputação de dentes ausentes foi implementado e validado, e a camada de integração/teste com APIs Radiomemory foi formalizada.

Resultado geral: o workspace saiu de um estado de setup/prototipagem para um estado de **plataforma de experimentação reproduzível**, com artefatos comparáveis e runbooks operacionais.

## 2) Linha do tempo inferida (docs + timestamps + git)

- **2026-03-20**: implementação do pipeline Hydra local E2E, docs de arquitetura/runbook/avaliação e primeiras runs (`FirstTest100`, smoke).
- **2026-03-21**: split oficial 70/15/15, experimento `FifthTest999_absentHM0`, auditorias de anotação, planos de dashboards/two-stage crop, DAE (smoke e full), e hardening para EC2.
- **2026-03-22**: trilha Foundation AE (várias runs full999 e híbrido), trilha AE filter para triagem de radiografias, integração RM Auth/Tools e comparativos com RM API longaxis.
- **2026-03-23**: ajuste utilitário de inspeção de reconstrução (`show_single_reconstruction_cv2.py`).

Indicador de intensidade (arquivos técnicos atualizados fora de runs/outputs):
- 2026-03-20: **28**
- 2026-03-21: **43**
- 2026-03-22: **35**
- 2026-03-23: **1**

## 3) Projetos e iniciativas trabalhadas

## 3.1 Hydra Longo-Eixo (principal)

Escopo entregue:
- treino multitarefa (`heatmap + presença`),
- avaliação padronizada,
- geração visual por época (augmentação + atenção),
- scripts de execução local/smoke/EC2,
- split reprodutível (incluindo 70/15/15).

Referências:
- `docs/arquitetura/HYDRA_UNET_MULTITASK_SPEC.md`
- `docs/implementacao/HYDRA_TRAINING_IMPLEMENTATION_LOG.md`
- `docs/runbooks/HYDRA_TRAINING_RUNBOOK.md`
- `docs/experimentos/HYDRA_EXPERIMENT_CHANGELOG.md`

Evidência de evolução (melhor `val_total_loss` por run):
- `FirstTest100`: **0.535970** (epoch 10)
- `SecondTest100`: **0.280176** (epoch 29)
- `ThirdTest100_sigma50`: **0.426239** (epoch 26)
- `FourthTest999_sigma52`: **0.247389** (epoch 30)
- `FifthTest999_absentHM0`: **0.227570** (epoch 30)

Leitura: houve ganho consistente ao escalar dataset e ajustar estratégia; `sigma=5.0` foi pior que o segundo baseline, e `absentHM0` aparece como melhor run de loss global até aqui.

## 3.2 Avaliação, auditoria e governança de métricas

Entregas:
- critérios oficiais de avaliação e promoção,
- índice operacional de scripts,
- auditorias de presença/top erros,
- auditorias de inversão de eixo e suspeitas morfológicas,
- plano de dashboards estáticos com manifest.

Referências:
- `docs/avaliacao/HYDRA_EVALUATION_CRITERIA.md`
- `docs/avaliacao/METRICS_SCRIPTS_INDEX.md`
- `docs/planos/HYDRA_ANNOTATION_AUDITOR_PLAN.md`
- `docs/planos/HYDRA_METRICS_DASHBOARDS_IMPLEMENTATION_PLAN.md`

## 3.3 Panorama Foundation (AE fundacional)

Entregas:
- projeto estruturado (`dataset/models/train/callbacks/configs/scripts`),
- runs smoke e full999,
- plano oficial para pretexto híbrido denoise+inpaint,
- checkpoints de encoder para transferência.

Referências:
- `panorama_foundation/README.md`
- `panorama_foundation/docs/PROJECT_PLAN.md`
- `panorama_foundation/docs/AE_HYBRID_DENOISE_FOUNDATION_PLAN.md`

Resultados (`best_val_loss`):
- `AE_FULL999_RESNET34_NOSKIP_BASELINE_V2_60E_MPS`: **0.05291** (melhor full999 observado)
- `AE_FULL999_RESNET34_NOSKIP_ALL_EPOCH_VIS_V2_MPS`: **0.05538**
- `AE_FULL999_RESNET34_NOSKIP_BASELINE_V1`: **0.05682**
- `visual_smoke_hybrid_v1`: **0.23147** (pior que smoke identity nesta amostra)

Leitura: baseline full999 evoluiu bem; modo híbrido ainda parece precisar calibração antes de virar default.

## 3.4 DAE de coordenadas (imputação de dentes ausentes)

Entregas:
- pipeline completo (`train/eval/smoke`),
- métricas por dente e por amostra,
- visuais de imputação,
- plano estratégico de dois AEs (imputação + QA).

Referências:
- `dae_longoeixo/README_DAE_COORDS_PLAN.md`
- `dae_longoeixo/README_TWO_AE_STRATEGY.md`

Resultados principais (eval):
- `DAE_142_FULL32`: `point_error_median_px_knocked` ~ **8.17 px**, `within_5px` ~ **0.258**
- `DAE_999_FIRST`: `point_error_median_px_knocked` ~ **8.16 px**, `within_5px` ~ **0.286**

Leitura: DAE full teve desempenho estável e competitivo vs smoke, com sinais de maturidade para rodada maior.

## 3.5 RM API / autenticação / probes

Entregas:
- runbook formal de entrypoints,
- guia de serviços RM,
- cliente/smoke/batch runner,
- comparativo de presença com endpoint de longaxis.

Referências:
- `docs/runbooks/RADIOMEMORY_API_ENTRYPOINTS_RUNBOOK.md`
- `radiomemory_auth/RM_IA_SERVICOS_GUIA.md`
- `radiomemory_auth/README.md`
- `radiomemory_api_tools/README.md`

Exemplo de benchmark API longaxis (`test`, 150 amostras):
- `presence_fixed.f1_macro`: **0.9837**
- `presence_fixed.auc_macro`: **0.9856**

## 3.6 AE Radiograph Filter (triagem por erro de reconstrução)

Entregas:
- pipeline de scoring/flag por percentil,
- geração de painéis top/low error,
- modo enhance + relatório HTML.

Referências:
- `ae_radiograph_filter/README.md`
- `ae_radiograph_filter/outputs/AE_FILTER_SMOKE20/summary.json`

Exemplo (`AE_FILTER_SMOKE20`):
- `num_images_scored`: 20
- `mae_threshold` (P90): 0.04236
- `num_flagged`: 2 (10%)

## 4) Onde estão os dados de métricas, testes e evidências

Hydra (runs e eval):
- `longoeixo/experiments/hydra_unet_multitask/runs/<RUN>/metrics.csv`
- `longoeixo/experiments/hydra_unet_multitask/runs/<RUN>/eval/metrics_summary.json`
- `longoeixo/experiments/hydra_unet_multitask/runs/<RUN>/eval/metrics_per_tooth.csv`
- `longoeixo/experiments/hydra_unet_multitask/runs/<RUN>/train_visuals/index.html`

Comparativos de presença/gating e RM API:
- `longoeixo/experiments/hydra_unet_multitask/runs/FifthTest999_absentHM0__cmp_*/eval/`
- `longoeixo/experiments/hydra_unet_multitask/runs/*/eval_rm_api_longaxis/metrics_summary.json`

DAE:
- `dae_longoeixo/experiments/coords_dae/runs/<RUN>/metrics.csv`
- `dae_longoeixo/experiments/coords_dae/runs/<RUN>/eval/metrics_summary.json`

Foundation AE:
- `panorama_foundation/experiments/ae_full999/runs/<RUN>/summary.json`
- `panorama_foundation/experiments/ae_full999/runs/<RUN>/metrics.csv`

Análises auxiliares:
- `analysis_inputs/metrics_per_tooth_Full70_15_15_absentHM0.csv`
- `analysis_inputs/plots_per_tooth_full70_nompl/*.png`

## 5) Como rodar e reproduzir rapidamente

Hydra:
```bash
bash longoeixo/scripts/run_hydra_train_local.sh <RUN_NAME>
bash longoeixo/scripts/run_hydra_eval.sh <RUN_NAME>
bash longoeixo/scripts/run_hydra_smoke.sh <RUN_NAME>
```

DAE:
```bash
bash dae_longoeixo/scripts/run_dae_train_local.sh <RUN_DAE>
bash dae_longoeixo/scripts/run_dae_eval.sh <RUN_DAE> test
bash dae_longoeixo/scripts/run_dae_smoke.sh <RUN_DAE_SMOKE>
```

Foundation AE:
```bash
bash panorama_foundation/scripts/run_ae_visual_smoke.sh --run-name visual_smoke_check
bash panorama_foundation/scripts/run_ae_full999.sh --run-name AE_FULL999_TEST
bash panorama_foundation/scripts/run_ae_full999_hybrid.sh --run-name AE_FULL999_HYBRID_TEST --device mps
```

AE filter e RM tools:
```bash
bash ae_radiograph_filter/scripts/run_sample_local.sh
python3 radiomemory_auth/rm_ia_smoke_test.py
python3 radiomemory_api_tools/probe_panoramic_longaxis.py
```

## 6) Achievements da semana

1. Estabelecimento de um **framework reproduzível de pesquisa** para Hydra (código + runbooks + critérios + artefatos).
2. Ganho de performance no core Hydra (de 0.5359 para 0.2276 em `val_total_loss` do baseline inicial ao melhor run da semana).
3. Criação de trilha de **data quality/auditoria** com planos e scripts já em estágio avançado.
4. Fundação AE consolidada com múltiplas runs e checkpoint de encoder reutilizável.
5. DAE funcional com métricas objetivas para imputação de coordenadas.
6. Integração de API RM padronizada com runbooks e utilitários operacionais.

## 7) Próximos passos recomendados (prioridade)

1. **Hydra: validar promoção no split `test` oficial** com critério completo (`presence_f1_macro`, `point_error_median_px`, `within_5px`, `false_point_rate`).
2. **Dashboards**: implementar Fase 1 do plano (`dashboard_registry.py` + integração em 4 scripts + `index.html` básicos).
3. **Two-stage crop**: iniciar Fase 1 (utilitários bbox/transform + testes round-trip), antes de treinar Stage-2.
4. **Foundation AE híbrido**: calibrar severidade de corrupção e repetir comparação justa com baseline full999.
5. **DAE**: ampliar avaliação em split `test` e iniciar experimento de mistura de níveis de corrupção.
6. **RM API**: consolidar contrato do endpoint definitivo usado em produção e manter smoke periódico automatizado.

## 8) Riscos e pontos de atenção

- Há muitos artefatos de run no workspace; manter naming/versionamento consistente por experimento é essencial.
- Parte dos resultados está em `val`; para decisão de promoção, consolidar somente em `test` (sem calibrar no test).
- Existe diferença de comportamento entre modos de presença (logits vs heatmap/composite); decisões precisam seguir critério oficial para evitar regressão operacional.

---

Documento preparado para onboarding rápido de continuidade de P&D.
