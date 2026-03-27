# HYDRA On-the-Fly Data Approach (Escala)

Data: 2026-03-20
Status: aprovado para evolução do pipeline

## Objetivo

Viabilizar treino em escala (incluindo 16k casos) sem pré-armazenar `stack64` em resolução original.

## Problema observado

Pré-gerar `stack64` original para todos os exames causa crescimento de storage inviável (ordem de TB), além de aumentar custo de I/O e risco operacional.

## Abordagem oficial

1. Fonte canônica: `longoeixo/imgs` + `longoeixo/data_longoeixo`.
2. DataLoader gera `Y_heatmap` on-the-fly por amostra.
3. Heatmap provisório é produzido em memória e reduzido para `256x256` no fluxo do batch.
4. `Y_presence` é derivado no mesmo fluxo (JSON/canais).
5. Treino consome apenas `X(1,256,256)`, `Y_heatmap(64,256,256)` e `Y_presence(32)`.

## Paralelismo

A geração on-the-fly deve ser paralelizada via DataLoader:
- `num_workers > 0`
- prefetch de batches
- workers persistentes

Resultado esperado:
- CPU prepara próximos batches enquanto acelerador treina no batch atual.

## Auditoria visual (obrigatória, baixa frequência)

Mesmo com geração on-the-fly, manter amostragem visual durante o treino para validar qualidade do target:
- overlay `imagem + max(mask64)` para conferir coordenadas.
- comparação before/after augmentação.
- capturas de atenção por camada (quando habilitado).

Frequência recomendada:
- baixa (ex.: a cada 5 ou 10 épocas), para reduzir impacto de I/O.

Objetivo:
- detectar cedo desvios de geometria, clipping indevido, ruído exagerado ou problemas de alinhamento imagem/target.

## Checagens de coerência (canal/dente/presença)

Além da auditoria visual, validar periodicamente:
- ordem canônica dos 64 canais (`11..18, 21..28, 31..38, 41..48` com `p1/p2`).
- mapeamento `dente i <-> canais 2*i e 2*i+1`.
- consistência de `Y_presence (32)` com os canais ou com o JSON.

## Compatibilidade de plataforma

- Local (Apple Silicon): `device=auto` prioriza `mps` quando disponível.
- EC2 (NVIDIA): `device=auto` prioriza `cuda`.
- fallback: `cpu`.

## Papel do `stack64` em disco

`stack64` em disco permanece apenas como recurso opcional para:
- debug
- inspeção visual
- smoke test

Não é recomendado como estratégia principal para escala.
