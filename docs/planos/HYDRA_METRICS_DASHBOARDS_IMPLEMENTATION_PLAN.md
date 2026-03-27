# HYDRA Metrics Dashboards - Plano de Implementação Segura

Data: 2026-03-21  
Status: Proposta técnica para implementação

## 1) Objetivo

Criar, por experimento, dois painéis HTML para centralizar métricas e auditorias:

1. `dashboard_runs`: métricas de runs (treino/eval), com tabelas e gráficos úteis.
2. `dashboard_audits`: auditorias de anotação e presença, com agregados e links para visuais especializados.

Escopo inicial: experimento `longoeixo/experiments/hydra_unet_multitask`.

## 2) Decisão de arquitetura

Abordagem escolhida: **híbrida estática com manifest atualizado pelos scripts**.

- Dashboards são HTML estático + JS (sem backend dedicado).
- Servidos com `python3 -m http.server`.
- Cada script avaliador registra seus artefatos em um `manifest.json`.
- O dashboard lê o manifest em tempo de abertura e renderiza tabelas/gráficos.

Motivo:
- simples para EC2/local;
- sem necessidade de processo web persistente;
- sem “regerar painel manualmente” se o script já registrar no manifest.

## 3) Estrutura de diretórios proposta

Dentro do experimento:

```text
longoeixo/experiments/hydra_unet_multitask/
  dashboard_runs/
    index.html
    manifest.json
  dashboard_audits/
    index.html
    manifest.json
```

Opcional futuro:
- `longoeixo/experiments/index.html` (launcher para escolher experimento).

## 4) Contrato de dados (manifest)

## 4.1 Regras gerais

- Um item no manifest representa uma “execução de relatório” (run eval, auditoria, overlay etc.).
- Campos mínimos:
  - `id` (string única)
  - `kind` (ex.: `hydra_eval`, `presence_eval`, `axis_audit`)
  - `experiment` (ex.: `hydra_unet_multitask`)
  - `run_name` (quando aplicável)
  - `split` (`train|val|test|all` quando aplicável)
  - `created_at_utc` (ISO8601)
  - `summary` (objeto leve para cards/tabelas)
  - `artifacts` (paths relativos para JSON/CSV/HTML)

- Paths no manifest devem ser relativos à raiz do experimento (`.../hydra_unet_multitask/`) para portabilidade.
- Escrita atômica do manifest (tmp + rename) para evitar corrupção.
- Política de deduplicação: chave natural `(kind, run_name, split, primary_artifact_path)`.

## 4.2 Exemplo de item (`hydra_eval`)

```json
{
  "id": "hydra_eval__Full70_15_15_absentHM0__test__2026-03-21T20:14:11Z",
  "kind": "hydra_eval",
  "experiment": "hydra_unet_multitask",
  "run_name": "Full70_15_15_absentHM0",
  "split": "test",
  "created_at_utc": "2026-03-21T20:14:11Z",
  "summary": {
    "presence_f1_macro": 0.8912,
    "presence_auc_macro": 0.9641,
    "point_error_median_px": 7.82,
    "point_within_5px_rate": 0.431,
    "false_point_rate_gt_absent_global": 0.071
  },
  "artifacts": {
    "metrics_summary_json": "runs/Full70_15_15_absentHM0/eval/metrics_summary.json",
    "metrics_per_tooth_csv": "runs/Full70_15_15_absentHM0/eval/metrics_per_tooth.csv",
    "metrics_per_quadrant_csv": "runs/Full70_15_15_absentHM0/eval/metrics_per_quadrant.csv",
    "pred_vs_gt_samples_dir": "runs/Full70_15_15_absentHM0/eval/pred_vs_gt_samples"
  }
}
```

## 5) Cobertura de scripts

## 5.1 Painel `dashboard_runs`

Integrar registro automático em:

- `eval.py` (`kind=hydra_eval`)
- `dae_longoeixo/eval_dae.py` (`kind=dae_eval`) se quiser exibir DAE junto
- (opcional) `train.py` para snapshot de treino (`kind=hydra_train_metrics`)

## 5.2 Painel `dashboard_audits`

Integrar registro automático em:

- `eval_presence_top_errors.py` (`kind=presence_eval`)
- `visualize_presence_top_errors_overlay.py` (`kind=presence_overlay_html`)
- `audit_axis_inversion.py` (`kind=axis_inversion_audit`)
- `audit_morphology_suspects.py` (`kind=morphology_audit`)
- `longoeixo/scripts/count_dentition_coverage.py` (`kind=dentition_coverage`) quando houver `--out-json`
- `longoeixo/scripts/calc_dataset_horizontal_stats.py` (`kind=dataset_horizontal_stats`)

## 5.3 Scripts que já geram HTML

Não substituir. O dashboard deve linkar para:
- overlays de presença;
- HTML de axis inversion;
- HTML de morphology;
- mosaico de horizontal stats;
- `train_visuals/index.html` por run.

## 6) UX dos dashboards

## 6.1 `dashboard_runs`

- Cards headline por run:
  - `presence_f1_macro`, `presence_auc_macro`
  - `point_error_median_px`
  - `point_within_5px_rate`
  - `false_point_rate_gt_absent_global`
- Tabela comparativa de runs com ordenação.
- Histogramas/distribuições de métricas (por run selecionada) a partir dos CSVs quando disponíveis.
- Links diretos para:
  - `metrics_summary.json`
  - `metrics_per_tooth.csv`
  - `train_visuals/index.html`

## 6.2 `dashboard_audits`

- Cards por auditoria com data/split/run.
- Tabela de “top findings” por tipo de auditoria.
- Gráficos de suporte:
  - histograma de erros de presença;
  - contagem de inversão de eixo;
  - distribuição de `suspect_score` morfológico.
- Links para HTML detalhado e CSVs completos.

## 7) Segurança e robustez de implementação

## 7.1 Integridade de manifest

- Atualização com lock de arquivo (quando disponível) ou estratégia write-then-rename.
- Validar schema mínimo antes de gravar.
- Se manifest inválido/corrompido, salvar backup e recriar.

## 7.2 Resiliência do dashboard

- Se artefato listado não existir, exibir aviso não bloqueante no card.
- Se JSON/CSV estiver malformado, ignorar item com erro e seguir renderização.
- Não depender de acesso a internet (bibliotecas JS locais ou sem dependências externas).

## 7.3 Compatibilidade de caminhos

- Sempre usar paths relativos ao experimento no manifest.
- Resolver links com base no diretório do dashboard.
- Não hardcode de host/porta.

## 7.4 Performance

- Carregar manifest completo e dados pesados sob demanda (lazy fetch por card expandido).
- Limitar número de pontos em gráficos por padrão.

## 8) Plano de rollout

Fase 1 (MVP seguro):
1. Criar utilitário comum `dashboard_registry.py` (append/update manifest).
2. Integrar registro em `eval.py`, `eval_presence_top_errors.py`, `audit_axis_inversion.py`, `audit_morphology_suspects.py`.
3. Criar `dashboard_runs/index.html` e `dashboard_audits/index.html` com tabela + cards + links.
4. Validar em uma run real e uma auditoria real.

Fase 2:
1. Integrar scripts restantes (`visualize_presence_top_errors_overlay.py`, `count_dentition_coverage.py`, `calc_dataset_horizontal_stats.py`, opcional DAE).
2. Adicionar histogramas e filtros avançados.
3. Adicionar launcher de experimentos (opcional).

## 9) Testes mínimos obrigatórios

## 9.1 Funcionais

- Rodar `eval.py` e verificar item novo em `dashboard_runs/manifest.json`.
- Rodar `audit_axis_inversion.py` e verificar item novo em `dashboard_audits/manifest.json`.
- Abrir ambos os dashboards via `http.server` e validar renderização dos novos cards.

## 9.2 Regressão

- Garantir que scripts antigos continuam gerando os mesmos artefatos originais.
- Garantir que falha no registro do manifest não interrompe a geração principal do avaliador.

## 9.3 Caminhos

- Validar links no dashboard em:
  - ambiente local;
  - EC2 com túnel SSH.

## 10) Critérios de aceite

- Painéis existem por experimento e abrem com `python3 -m http.server`.
- Novo resultado de avaliação/auditoria aparece automaticamente no painel correspondente após execução do script.
- Usuário consegue navegar de card para JSON/CSV/HTML detalhado sem editar caminhos manualmente.
- Não há regressão nos scripts atuais de métricas.

## 11) Riscos e mitigação

- Risco: acoplamento frágil entre estrutura de saída e dashboard.  
Mitigação: manifest versionado (`schema_version`) e parser tolerante.

- Risco: crescimento do manifest ao longo do tempo.  
Mitigação: política opcional de retenção por run (ex.: manter último por `(kind, run, split)`).

- Risco: dados inconsistentes entre scripts.  
Mitigação: padronizar nomes de métricas-chave no `summary`.

## 12) Próximo passo recomendado

Implementar Fase 1 com PR pequeno:
- utilitário de registry + integração em 4 scripts;
- dois `index.html` básicos;
- validação com 1 run e 1 auditoria.
