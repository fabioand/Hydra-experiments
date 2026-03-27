# Estratégia com Dois Autoencoders (Sem Implementação)

Documento de arquitetura para usar dois autoencoders no contexto de long-eixo dentário:

1. **AE-1 (Imputador)**: completa coordenadas de dentes ausentes.
2. **AE-2 (Avaliador)**: estima confiabilidade da predição da U-Net via erro de reconstrução.

Este documento é apenas de planejamento. Não inclui implementação de código nesta etapa.

## 1) Objetivo Geral

Separar duas funções que têm naturezas diferentes:

- **Imputação anatômica**: reconstruir geometrias plausíveis quando dente está ausente.
- **Controle de qualidade (QA)**: detectar quando a predição da U-Net está anatomicamente incoerente.

## 2) AE-1 Imputador (Dentes Ausentes)

### 2.1 Entradas e saídas

- Entrada: coordenadas parciais (`128`) com dentes ausentes em `(0,0)`.
- Máscara de presença explícita (dente/ponto).
- Saída: coordenadas completas (`128`).

### 2.2 Regra de ausentes

- Ausentes podem ser representados como `(0,0)`.
- Obrigatório usar **máscara de presença** em conjunto.
- Loss deve priorizar coordenadas faltantes (ou usar ponderação faltantes vs presentes).

### 2.3 Papel no produto

- Fornecer eixo candidato para dentes ausentes.
- Geração de sugestões anatômicas iniciais (ex.: apoio para implantes, planejamento preliminar).

## 3) AE-2 Avaliador (QA da U-Net)

### 3.1 Ideia central

- Passar a predição da U-Net no AE-2.
- Medir diferença entre entrada e reconstrução.
- Diferença alta indica provável predição ruim/incoerente.

### 3.2 Entradas e saídas

- Entrada do AE-2:
  - coordenadas preditas pela U-Net (`128`),
  - máscara de presença (idealmente da própria U-Net),
  - opcional: probabilidades de presença/certeza por dente.
- Saída: reconstrução das coordenadas esperadas no manifold anatômico.

### 3.3 Score de qualidade

- Score primário: erro de reconstrução **mascarado por presença**.
- Não penalizar dentes ausentes (ou penalizar com peso muito baixo).
- Reportar:
  - score global da radiografia,
  - score por dente,
  - top dentes suspeitos.

## 4) Como Tratar Ausência (Ponto Crítico)

Para ambos os AEs:

- `(0,0)` para ausente é aceitável como placeholder.
- Sem máscara, o modelo aprende atalhos e mistura “ausente” com “erro”.
- Portanto, padrão obrigatório:
  - placeholder `(0,0)` + máscara explícita,
  - loss/score mascarados,
  - calibração com validação real.

## 5) Diferença de Treino entre AE-1 e AE-2

## 5.1 AE-1 Imputador

- Treino com knockout sintético de dentes.
- Target sempre completo.
- Foco em imputação de faltantes.

## 5.2 AE-2 Avaliador

- Treino para aprender manifold de predições plausíveis.
- Pode usar:
  - GT completo + perturbações controladas,
  - predições históricas da U-Net com rótulo de qualidade.
- Objetivo principal: separar predições boas vs ruins por score.

## 6) Integração com a U-Net (Fluxo Proposto)

1. U-Net produz long-eixos + presença.
2. AE-2 recebe saída da U-Net e gera score de confiabilidade.
3. Se score ruim:
   - marcar exame/dentes para revisão,
   - opcionalmente acionar AE-1 para proposta de ajuste/imputação em dentes críticos.
4. Se score bom:
   - seguir fluxo normal.

## 7) Calibração de Threshold do AE-2

- Não usar limiar fixo arbitrário.
- Calibrar em validação com erro real vs GT.
- Salvar thresholds por:
  - score global,
  - score por dente,
  - severidade (leve/moderado/grave).

## 8) Métricas Recomendadas

Para AE-1:

- MAE/MSE em dentes nocauteados.
- erro de ponto (px) em faltantes.
- taxas `<=3px`, `<=5px`, `<=10px`.

Para AE-2:

- AUC para detecção de predição ruim (global e por dente).
- Precision/Recall em diferentes limiares.
- correlação entre score do AE-2 e erro real da U-Net.

## 9) Riscos e Mitigações

- **Risco**: ausência confundir avaliação.
  - Mitigação: máscara explícita + score mascarado.
- **Risco**: AE-2 punir variações anatômicas reais raras.
  - Mitigação: dataset diverso + calibração por subgrupo.
- **Risco**: score difícil de interpretar clinicamente.
  - Mitigação: score por dente + explicação de contribuição local.

## 10) Roadmap (Somente Planejamento)

1. Definir protocolo de dataset para AE-2 (fontes e rótulos de qualidade).
2. Definir fórmula final do score mascarado.
3. Definir estratégia de calibração de limiar (global + por dente).
4. Definir contrato de integração no pipeline da U-Net.
5. Só depois iniciar implementação.

---

Status deste documento: **planejamento aprovado para fase de design** (sem implementação).
