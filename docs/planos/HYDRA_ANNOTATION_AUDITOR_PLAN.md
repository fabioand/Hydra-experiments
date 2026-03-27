# HYDRA Annotation Auditor - Plano de Projeto

Projeto para detectar e priorizar anotacoes potencialmente ruins no dataset, com foco em:
- erros de presenca/ausencia;
- erros geometricos de longoeixo;
- erros de deslocamento de dente (cascata);
- erros mistos (`p1` de um dente e `p2` de outro);
- anomalias morfologicas/espaciais na geometria das anotacoes.

## 1) Objetivo

Criar um avaliador que gere rankings acionaveis para revisao humana, com rastreabilidade por radiografia e por dente.

Saida esperada:
- lista curta de casos suspeitos (top-N) para revisao;
- evidencia visual (overlay) quando aplicavel;
- metricas para acompanhar melhora apos correcoes.

## 2) Escopo funcional

## 2.1 Auditoria de Presenca/Ausencia

Entrada:
- `presence_logits` / `sigmoid`;
- GT de presenca.

Por radiografia:
- `fn_count`, `fp_count`, `presence_error_count`;
- `bce_presence_mean`;
- dentes com erro (`fn_teeth`, `fp_teeth`).

Ranking:
- `presence_suspect_score = w_fn*FN + w_fp*FP + w_bce*BCE`.

Artefatos:
- `presence_errors_per_sample.csv`
- `presence_top_errors_topK.csv`
- histograma por numero de dentes errados.

## 2.2 Auditoria Geometrica (uso real)

Entrada:
- pontos preditos por canal;
- GT de pontos.

Metrica operacional:
- avaliar geometria em dentes com `pred_presence=1` (uso real com gating).

Metrica diagnostica:
- avaliar tambem em `GT_presence=1` para separar erro de classificacao vs erro geometrico.

Saidas:
- erro por dente e por radiografia (`median/p95/mean`);
- taxa dentro de 5px/10px.

## 2.3 Detector de Deslocamento em Cascata (Novo)

Ideia:
- comparar erro do dente correto (`shift=0`) com erro apos deslocar indice no quadrante (`shift=-3..+3`).

Calculo base:
- `error_shift0`: erro medio no mapeamento correto;
- `error_best_shift`: menor erro medio entre shifts testados;
- `cascade_gain = error_shift0 - error_best_shift`;
- `support`: quantidade de dentes validos no quadrante;
- `cascade_score = cascade_gain * sqrt(support)`.

Sinal de suspeita forte:
- `best_shift != 0` e `cascade_gain` alto com `support` bom.

## 2.4 Detector de Erro Misto p1/p2 (Novo)

Problema alvo:
- anotacao onde `p1` corresponde a um dente e `p2` a outro.

Estrategia:
- avaliar custo de correspondencia:
1. direto (`p1->p1`, `p2->p2`);
2. cruzado (`p1->p2`, `p2->p1`);
3. parcial por ponto (best shift de `p1` e `p2` separados).

Flags:
- `swap_flag` (cruzado melhor que direto);
- `mixed_neighbor_flag` (`best_shift_p1 != best_shift_p2`);
- `best_shift_p1`, `best_shift_p2`.

## 2.5 Auditoria de Eixo Invertido p1/p2

Problema alvo:
- anotador inverte orientacao do eixo (ordem dos pontos), quebrando convencao `p1=cuspide` -> `p2=raiz`.

Regra operacional usada no auditor:
- dentes superiores: suspeita de inversao quando `p1.y < p2.y`;
- dentes inferiores: suspeita de inversao quando `p1.y > p2.y`.

Saidas:
- ranking por radiografia com `inverted_count` e `inverted_rate`;
- lista de dentes invertidos por caso;
- mosaico HTML com overlay:
  - verde = eixo consistente;
  - vermelho = suspeita de inversao;
  - bolinha em `p1` e label do dente.

## 2.6 Auditoria Morfologica/Geometrica Heuristica (Novo)

Objetivo:
- achar suspeitos de erro de anotacao usando consistencia anatomica espacial, mesmo sem depender da preditividade do modelo.

### 2.6.1 Heuristicas principais

1. Sequencia horizontal no quadrante
- verificar inversoes locais da ordem dos dentes adjacentes no eixo `x`.
- detectar quando um dente "passa" do vizinho na horizontal.

2. Altura inconsistente com arcada
- superior muito baixo (outlier de `y` para dentes superiores).
- inferior muito alto (outlier de `y` para dentes inferiores).

3. Espacamento horizontal entre vizinhos
- distancia `|dx|` entre adjacentes muito pequena/grande para o padrao do par.

4. Comprimento do longoeixo
- `||p2-p1||` muito curto ou muito longo para o dente.

5. Inclinacao do longoeixo
- razao `|dx|/|dy|` muito fora do padrao do dente.

6. Separacao entre arcadas no caso
- gap vertical entre medianas de superior e inferior muito pequeno/grande.

### 2.6.2 Geometria e normalizacao estatistica

Features por dente:
- centro do dente: `cx=(x1+x2)/2`, `cy=(y1+y2)/2`;
- comprimento de eixo: `axis_len = hypot(dx,dy)`;
- inclinacao: `axis_tilt = |dx|/(|dy|+eps)`.

Features por pares adjacentes (por quadrante):
- `dx_pair = cx(next)-cx(curr)`;
- `abs_dx_pair = |dx_pair|`.

Feature por radiografia:
- `jaw_gap = median_y_lower - median_y_upper`.

Baseline robusto:
- por dente/par, usar mediana e escala robusta (MAD escalado).
- converter para `z_robusto` e marcar outliers por threshold.

### 2.6.3 Score de suspeita morfologica

Por radiografia:
- contagem de flags por indicador + severidade (excesso de z acima do limiar).
- score combinado ponderado para ranking final.

Saidas:
- ranking global (`top_morphology_suspects`),
- detalhamento por dente (`morphology_features_per_tooth`),
- histogramas por indicador.

## 3) Artefatos finais do auditor

## 3.1 Tabelas

- `top_presence_suspects.csv`
- `top_geometry_suspects.csv`
- `top_cascade_suspects.csv`
- `top_mixed_p1p2_suspects.csv`
- `top_axis_inversion_suspects.csv`
- `top_morphology_suspects.csv`
- `joint_priority_suspects.csv` (intersecao/score combinado)

## 3.2 Visuais

- overlays GT vs predicao para top presenca;
- overlays para top cascata/misto;
- mosaico HTML de inversao de eixo;
- (opcional) overlays para top morfologia.

## 3.3 Resumo

- `annotation_auditor_summary.json` com:
  - distribuicao de scores;
  - contagem de casos suspeitos por tipo;
  - thresholds usados.

## 4) Criterios de priorizacao para revisao humana

Prioridade alta:
1. casos com erro de presenca alto (`FN` principalmente);
2. casos com `cascade_score` alto e `best_shift != 0`;
3. casos com `mixed_neighbor_flag = true`;
4. casos com alta anomalia morfologica (multiplos indicadores);
5. intersecao entre listas (presenca + cascata + geometria + morfologia).

## 5) Plano de implementacao

Fase 1 (implementada):
- avaliador de presenca com ranking e histograma;
- overlay HTML para top presenca.

Fase 2 (pendente):
- script `eval_geometry_top_errors.py` com visao operacional e diagnostica.

Fase 3 (pendente):
- script `eval_label_shift_suspects.py` (cascata por quadrante e global).

Fase 4 (pendente):
- extensao para `p1/p2` cruzado/misto e score unificado.

Fase 5 (pendente):
- script agregador `build_annotation_review_queue.py` para gerar fila final de revisao.

Fase 6 (implementada):
- script dedicado `audit_axis_inversion.py` para caca de inversao de orientacao.

Fase 7 (implementada nesta iteracao):
- script `audit_morphology_suspects.py` com ranking morfologico e histogramas por indicador.

## 6) Decisoes tecnicas

- Usar `best.ckpt` da run para auditorias baseadas em predicao.
- Auditorias morfologicas podem rodar apenas sobre anotacoes (JSON), sem checkpoint.
- Manter avaliacao por split (`train/val/test/all`), com preferencia por:
  - `train` para "cacar anotacoes ruins";
  - `test` para medir generalizacao final.
- Relatorios devem registrar versao de modelo (quando houver), split e thresholds.

## 7) Riscos e mitigacao

Risco:
- confundir caso dificil real com anotacao ruim.

Mitigacao:
- usar multiplos sinais (presenca + geometria + cascata + morfologia);
- priorizar intersecao de sinais;
- usar thresholds robustos por dente/par (nao so globais);
- revisao humana final antes de corrigir rotulo.
