# HYDRA Two-Stage Crop Refinement Plan

Data: 2026-03-21
Status: Proposta tecnica para implementacao

## 1) Objetivo

Adicionar um pipeline de 2 redes para melhorar a precisao geometrica dos pontos:

1. Stage-1 (coarse): rede atual no frame completo da panoramica (`1x256x256`).
2. Stage-2 (refine): nova rede treinada em recorte dos dentes (bbox + margem), predizendo os mesmos 64 heatmaps e 32 presencas.

A saida final de pontos vem do Stage-2, convertida de coordenadas do crop para coordenadas da panoramica.

## 2) Hipotese e beneficio esperado

Hipotese: o Stage-2 recebe maior resolucao efetiva da regiao dentaria e reduz erro de localizacao fina.

Beneficio esperado:
- reduzir `point_error_median_px` e elevar `point_within_5px_rate` (criterios oficiais).
- manter ou melhorar robustez de presenca sem aumento relevante de falso ponto em ausentes.

## 3) Escopo e nao-escopo

Escopo desta fase:
- desenho de dados, treino, inferencia e avaliacao em 2 estagios.
- plano de testes e anti-regressao.
- estrategia de rollout gradual.

Nao-escopo nesta fase:
- trocar backbone base do Hydra atual.
- mudar ordem canonica de canais.
- alterar criterios oficiais de promocao.

## 4) Desenho de dados do Stage-2

## 4.1 Como obter o bbox dos dentes

Treino:
- usar bbox derivado dos pontos GT presentes (min/max de x,y).
- aplicar margem pequena configuravel: `crop_margin_px` (ex.: 16 a 32 px no frame original).

Inferencia:
- usar bbox derivado dos pontos preditos pelo Stage-1.
- margem pode ser maior que no treino para seguranca (ex.: +20%).

## 4.2 Regras de robustez do crop

Aplicar sempre:
- clamp do bbox para limites da imagem.
- largura/altura minimas (`crop_min_size_px`) para evitar zoom extremo.
- opcional: forcar aspecto padrao antes do resize para limitar distorcao.

## 4.3 Mapeamento de coordenadas

Para cada ponto GT/predito:
- converter pano -> crop para treino do Stage-2.
- converter crop -> pano para saida final.

Guardar transformacao afim simples por amostra:
- `offset_x`, `offset_y`, `scale_x`, `scale_y`.

Isso e obrigatorio para auditoria e debug de erros de coordenada.

## 5) Estrategia de treino (ponto critico)

## 5.1 Evitar mismatch treino vs inferencia

Risco principal:
- Stage-2 treinado com bbox perfeito (GT), mas inferido com bbox imperfeito (predito Stage-1).

Mitigacao obrigatoria:
- treino misto de origem de crop:
  - `p_gt_crop` (ex.: 0.5): bbox GT + margem.
  - `p_pred_or_noisy_crop` (ex.: 0.5): bbox simulado com ruido/jitter ou bbox de teacher Stage-1 precomputado.

Jitter sugerido no bbox:
- translacao aleatoria pequena (ex.: ate 5% do bbox).
- escala aleatoria (ex.: 0.9 a 1.1).
- perturbacao independente por lado pequena.

## 5.2 Targets e loss no Stage-2

Manter a mesma semantica multitask:
- `Y_heatmap`: 64 canais (ordem canonica identica ao Stage-1).
- `Y_presence`: 32.

Loss:
- mesmo `HydraMultiTaskLoss`, iniciando com parametros atuais.
- manter teste com `absent_heatmap_weight=0.0` como baseline preferencial (estado mais recente promissor).

## 5.3 Split e reproducibilidade

- manter split oficial fixo para comparabilidade.
- registrar seed, configuracoes de crop e proporcao GT/noisy por run.
- salvar artefatos de transformacao para debug amostral.

## 6) Pipeline de inferencia em producao

Fluxo:
1. Stage-1 prediz heatmaps/presenca no frame completo.
2. Gerar bbox de dentes via pontos Stage-1 com regra robusta (threshold, clamp, min-size).
3. Recortar imagem original usando bbox + margem.
4. Stage-2 prediz no crop.
5. Reprojetar pontos Stage-2 para coordenadas da panoramica.
6. Aplicar gating final por presenca (estrategia oficial Hydra).

Fallbacks (obrigatorios):
- se bbox invalido/degenerado: usar crop de fallback central amplo ou usar saida Stage-1 diretamente.
- logar taxa de fallback (metrica de saude de pipeline).

## 7) Plano de avaliacao

Comparar 3 modos no mesmo split de teste:

1. Baseline atual: Stage-1 apenas.
2. Oracle crop: Stage-2 com bbox GT (limite superior).
3. Real cascade: Stage-2 com bbox Stage-1 (modo real de producao).

Headline metrics (criterio oficial):
- `presence_f1_macro`
- `presence_auc_macro`
- `point_error_median_px` (present-only)
- `point_within_5px_rate` (present-only)
- `point_error_median_px_when_pred_presence_pos`
- `point_within_5px_rate_when_pred_presence_pos`
- `false_point_rate_gt_absent_global`

Leitura esperada:
- (2) > (3) >= (1) em localizacao.
- se (3) < (1), investigar erro de bbox/mapeamento antes de mexer em arquitetura.

## 8) Plano de testes (com foco em regressao)

## 8.1 Testes unitarios (geometria e coordenadas)

Adicionar testes para:
- funcao de bbox a partir de pontos (incluindo dentes ausentes).
- clamp/min-size/margem.
- conversao pano->crop->pano (round-trip com erro maximo tolerado <= 1 px).
- comportamento em casos extremos (bbox na borda, bbox muito pequeno, sem pontos validos).

## 8.2 Testes de integracao de dados

- DataLoader Stage-2 retorna shapes corretos.
- Ordem de canais e presenca permanecem identicas a especificacao.
- augmentacao geometrica consistente entre imagem e target no crop.

## 8.3 Testes de inferencia end-to-end

Com pequeno conjunto fixo:
- executar Stage-1 -> crop -> Stage-2 -> reprojecao.
- validar ausencia de excecoes e formato de saida.
- validar limites de coordenadas finais dentro da imagem.

## 8.4 Regressao de metricas (gate de promocao)

Antes de promover experimento:
- comparar contra baseline Stage-1 no mesmo `test`.
- bloquear promocao se:
  - queda relevante em `presence_f1_macro`, ou
  - piora de `point_error_median_px` no modo real cascade, ou
  - aumento relevante de `false_point_rate_gt_absent_global`.

Sugestao operacional:
- definir limiares minimos de delta aceitavel no config de avaliacao para decisao automatizavel.

## 8.5 Regressao visual

Gerar paineis lado a lado para amostras fixas:
- pano + pontos Stage-1.
- bbox gerado.
- crop + pontos Stage-2.
- pano final com pontos reprojetados.

Objetivo:
- detectar cedo bugs de transformacao de coordenadas que metricas agregadas podem mascarar.

## 9) Riscos principais e mitigacoes

1. Propagacao de erro do Stage-1 para Stage-2.
Mitigacao: treino misto GT/noisy e fallback.

2. Bug de transformacao de coordenadas.
Mitigacao: testes round-trip e overlays dedicados.

3. Overfit do Stage-2 em crop "limpo demais".
Mitigacao: jitter de bbox, augmentacao leve e validacao em modo real cascade.

4. Ganho apenas em oracle, sem ganho real em producao.
Mitigacao: decisao de promocao baseada exclusivamente no modo real cascade.

## 10) Plano de implementacao incremental

Fase 1: Infra de crop e geometria
- utilitarios de bbox/transformacao.
- testes unitarios de coordenadas.

Fase 2: Dataset Stage-2
- modo de treino com crop GT/noisy.
- auditoria visual do crop e targets.

Fase 3: Treino Stage-2 baseline
- primeira run controlada.
- avaliar oracle vs cascade.

Fase 4: Integracao de inferencia
- pipeline completo Stage-1+Stage-2 com fallback.
- teste end-to-end e artefatos de debug.

Fase 5: Hardening anti-regressao
- gates de metricas + regressao visual.
- checklist de promocao.

## 11) Checklist de promocao

Promover somente se, no split oficial de teste:
- melhora em localizacao present-only no modo real cascade.
- presenca nao degrada de forma relevante.
- falso ponto em ausentes nao piora de forma relevante.
- taxa de fallback baixa e estavel.
- sem falhas nos testes de round-trip e E2E.

## 12) Convencoes de configuracao sugeridas

Adicionar bloco no config de treino/eval:
- `cascade.enabled`
- `cascade.stage1_checkpoint`
- `cascade.crop_margin_px`
- `cascade.crop_min_size_px`
- `cascade.train_p_gt_crop`
- `cascade.train_bbox_jitter`
- `cascade.fallback_mode`

Manter defaults conservadores para rollout seguro.

## 13) Decisao recomendada agora

A abordagem e tecnicamente solida e alinhada ao estado atual do projeto.

Recomendacao:
1. Implementar primeiro a infraestrutura de bbox + transformacoes + testes de round-trip.
2. Treinar Stage-2 com mistura GT/noisy.
3. Aceitar a estrategia apenas se o modo real cascade superar Stage-1 nas metricas oficiais sem regressao relevante de presenca.

## 14) Consideracoes sobre geometria da panoramica e opcoes de recorte

Ponto observado: ao forcar toda panoramica em entrada quadrada (`256x256`), a dimensao horizontal (onde dentes adjacentes se diferenciam) sofre compressao relativa. Isso pode limitar precisao fina entre dentes lado a lado.

### 14.1 Opcao A: BBox dentario com expansao vertical (crop mais "quadrado")

Ideia:
- manter um unico crop dos dentes, mas expandir mais a altura que a largura para aproximar aspecto quadrado antes do resize final.

Vantagens:
- reduz compressao horizontal no recorte final.
- preserva contexto bilateral completo (todos os dentes juntos).
- mudanca menor no pipeline atual (continua 32 dentes / 64 canais).

Cuidados:
- expansao vertical excessiva pode reintroduzir muito fundo anatomico irrelevante.
- precisa controlar `target_aspect_ratio` + `max_extra_context`.

Parametros sugeridos:
- `cascade.crop_target_aspect_ratio` (ex.: 1.0).
- `cascade.crop_expand_vertical_factor` (ex.: 1.2 a 1.6).
- `cascade.crop_max_size_fraction` para evitar crop "grande demais".

### 14.2 Opcao B: Recorte por hemiface (16 dentes) com espelhamento controlado

Ideia:
- derivar bbox total dos pontos e separar em 2 ROIs (esquerda e direita), cada uma com pequena margem.
- treinar Stage-2 especializado em 16 dentes por lado.
- padronizar orientacao treinando no mesmo referencial (ex.: lado esquerdo canonico), usando o lado direito espelhado de forma deterministica com remapeamento de canais/dentes.

Vantagens:
- crop naturalmente mais quadrado e com maior resolucao relativa por dente.
- aumento efetivo de amostras (cada pano gera 2 exemplos: hemiface esquerda + hemiface direita transformada).
- especializacao pode reduzir ambiguidade local entre dentes vizinhos.

Cuidados criticos:
- espelhamento so e seguro aqui se houver remapeamento canonico rigoroso dos dentes e canais.
- nao pode violar semantica odontologica (ex.: 11..18 vs 21..28; 31..38 vs 41..48).
- inferencia precisa inverter transformacoes e remapear indices para voltar ao sistema global de 32 dentes.

Regras obrigatorias para essa opcao:
1. Definir tabela explicita de mapeamento de dentes/canais entre lado original e lado canonico.
2. Testar round-trip de indices (global -> hemiface -> global) sem perda.
3. Manter metrica final sempre no espaco global dos 32 dentes.

### 14.3 Impacto em augmentacao (regra de flip)

Regra vigente do projeto:
- "sem horizontal flip" para modelo global, por risco de quebrar lateralidade.

Refinamento da regra:
- para Stage-2 hemiface, horizontal flip como augmentacao continua proibido.
- no Stage-2 hemiface, o espelhamento do lado direito e permitido apenas como normalizacao deterministica de orientacao (aplicada sempre que entrada for hemiface direita), com remapeamento canonico validado por testes de indice.
- para Stage-1 global, a proibicao de flip permanece.

Distincao obrigatoria:
- augmentacao (aleatoria): proibida para flip.
- normalizacao de orientacao (deterministica): permitida para alinhar hemiface direita ao referencial canonico.

### 14.4 Regressao e comparacao justa entre opcoes

Comparar no mesmo split/teste oficial:
1. Baseline Stage-1 atual.
2. Cascade com Opcao A (bbox unico com expansao vertical).
3. Cascade com Opcao B (hemiface 16 dentes).

Critério de escolha:
- escolher a opcao com melhor ganho no modo real cascade, sem regressao relevante em presenca e sem piora de `false_point_rate_gt_absent_global`.

### 14.5 Recomendacao pratica de execucao

Ordem sugerida (menor risco -> maior mudanca):
1. Implementar e testar Opcao A primeiro (impacto menor, ganho potencial rapido).
2. Implementar Opcao B em paralelo como experimento estruturado, com foco forte em mapeamento canonico e testes de indice/coordenada.
3. Promover apenas a opcao vencedora em `test` real cascade.
