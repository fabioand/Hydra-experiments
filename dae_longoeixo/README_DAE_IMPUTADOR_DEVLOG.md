# DAE Imputador de Dentes - Histórico de Desenvolvimento

Este documento registra a evolução do autoencoder imputador de long-eixo dentário, desde a ideia inicial até o estado atual do pipeline.

## 1) Contexto e problema

Objetivo clínico/técnico que guiou o desenvolvimento:

- completar coordenadas de dentes ausentes a partir de um conjunto parcial de long-eixos;
- manter coerência anatômica global da arcada;
- aproveitar o padrão de operação já consolidado no projeto (treino reprodutível, TensorBoard, checkpoints, avaliação e inspeção visual).

Representação adotada desde o início:

- 32 dentes canônicos (`11..18, 21..28, 31..38, 41..48`);
- 2 pontos por dente (longo-eixo) = 64 pontos;
- vetor `128` (`x,y` para cada ponto), normalizado por imagem de origem (`x/(W-1)`, `y/(H-1)`).

## 2) Fase 1 - Primeiro DAE (somente dentição completa)

Primeira hipótese:

- treinar apenas com exames com os 32 dentes presentes;
- no treino, simular ausência via knockout de dentes (zerando os 2 pontos do dente para `(0,0)`);
- target (`Y`) permanece completo.

Decisões técnicas:

- arquitetura MLP para coordenadas (sem imagem), pois o problema é geométrico-relacional;
- corrupção on-the-fly no `Dataset`, sem gerar cópias estáticas;
- loss priorizando reconstrução de dentes nocauteados;
- split persistido em JSON para reprodutibilidade.

Resultado da fase:

- pipeline end-to-end funcional e estável;
- capacidade inicial de reconstrução plausível mesmo com poucas amostras completas.

## 3) Fase 2 - Operacionalização no padrão Hydra/U-Net

Para manter consistência com o resto do projeto, o pipeline ganhou:

- `train_dae.py` com `best.ckpt` e `last.ckpt`;
- `eval_dae.py` com resumo em JSON/CSV;
- TensorBoard com losses e métricas por época;
- scripts `run_*` para treino/eval/smoke;
- callbacks visuais para acompanhar imputação ao longo das épocas.

Arquivos principais consolidados:

- `dae_data.py`
- `dae_model.py`
- `train_dae.py`
- `eval_dae.py`
- `dae_visuals.py`

## 4) Fase 3 - Estratégia de dois autoencoders (arquitetura de produto)

Separação conceitual definida:

- AE-1: imputador de dentes ausentes (implementado);
- AE-2: avaliador de qualidade da predição da U-Net por erro de reconstrução (documentado, sem implementação nesta etapa).

Motivação:

- imputação e QA são problemas diferentes e pedem objetivos de treino diferentes;
- separar evita conflitar metas de reconstrução com metas de detecção de erro.

## 5) Fase 4 - Inspeção visual em exames com ausências reais

Foi criado fluxo de análise qualitativa para casos com dentes realmente faltantes:

- script de imputação sobre dataset real;
- geração de mosaico HTML;
- overlay de dentes anotados em verde e dentes imputados em vermelho.

Objetivo:

- facilitar validação clínica rápida e triagem de comportamento do modelo.

## 6) Fase 5 - Escalonamento para dataset incompleto (999 local e 16k EC2)

Problema identificado:

- treinar só com casos completos limita dados e generalização.

Solução implementada: supervisão parcial mascarada.

Decisões:

- incluir exames incompletos com `sample_filter=any_with_min_teeth`;
- dentes sem GT entram com placeholder `(0,0),(0,0)`;
- máscara explícita de GT disponível por dente (`gt_available_mask_32`);
- loss ignora dentes sem anotação válida (não supervisiona onde não há verdade-terreno);
- knockout só ocorre em dentes que têm GT disponível.

Impacto:

- modelo passa a aprender com toda a geometria observada do dataset incompleto;
- preserva rigor de supervisão sem inventar alvo para dentes ausentes no GT.

## 7) Fase 6 - Augmentação geométrica focada em ausência

Hipótese adicionada:

- pequenas migrações dentárias podem “disfarçar buracos” em ausências reais.

Técnica implementada:

- jitter horizontal leve durante treino;
- aplicado de forma coerente em entrada e target para não quebrar supervisão;
- reforço em vizinhanças relevantes aos dentes nocauteados.

Objetivo:

- melhorar robustez a variações anatômicas e compensações locais.

## 8) Fase 7 - Regularizações anatômicas suaves

Foram incluídas regularizações opcionais para melhorar plausibilidade:

- `w_arc_spacing`: suaviza regularidade de espaçamento entre centros de dentes adjacentes por quadrante;
- `w_anchor_rel`: ancora dente nocauteado aos vizinhos observados em GT, comparando relações vetoriais locais.

Princípios adotados:

- regularização fraca e opcional;
- evitar “engessar” anatomias atípicas;
- foco em estabilidade local sem impor forma rígida de arco.

Observação prática:

- pesos altos podem degradar casos difíceis;
- tuning deve ser conservador (especialmente no `anchor_rel`).

## 9) Fase 8 - Performance operacional do discovery

Gargalo encontrado em bases grandes:

- leitura completa de imagem no discovery só para obter dimensão;
- percepção de travamento por falta de feedback frequente.

Melhorias aplicadas:

- leitura apenas de dimensões (`H,W`) no discovery;
- logs progressivos por `scanned` com intervalo configurável (`discovery_progress_interval`).

Resultado:

- startup mais rápido e previsível;
- melhor observabilidade em EC2 durante varredura de milhares de exames.

## 10) Métricas e monitoramento consolidados

Métricas centrais usadas no ciclo de desenvolvimento:

- MSE/MAE global;
- MSE/MAE em dentes nocauteados;
- erro em pixels por ponto (média/mediana/P90);
- taxas por tolerância (`<=3px`, `<=5px`, `<=10px`);
- métricas por dente e ranking por amostra.

Monitoramento:

- TensorBoard por run;
- CSV por época;
- checkpoints `best`/`last`;
- mosaicos HTML para inspeção humana.

## 11) Decisões-chave que ficaram como padrão

- coordenadas normalizadas absolutas por imagem;
- placeholder `(0,0)` permitido, mas sempre acompanhado de máscara explícita;
- perda mascarada por disponibilidade de GT;
- corrupção e augmentação on-the-fly;
- separação entre imputação (AE-1) e avaliação de qualidade (AE-2);
- inspeção visual como etapa obrigatória de validação.

## 12) Estado atual

O AE imputador está funcional end-to-end, com:

- treino local e em EC2;
- suporte a dataset completo e incompleto;
- augmentações geométricas e regularizações opcionais;
- avaliação quantitativa e visual;
- pipeline reprodutível pronto para novos ciclos de tuning.

Próxima direção natural:

- ampliar treino em larga escala (16k) com tuning fino dos pesos de regularização;
- calibrar comportamento para casos extremos (edentulismo severo, terceiros molares atípicos);
- iniciar implementação do AE-2 (avaliador) com score mascarado por presença.

