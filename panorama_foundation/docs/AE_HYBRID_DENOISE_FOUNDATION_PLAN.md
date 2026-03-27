# Plano Oficial - Autoencoder Fundacional Denoiser/Inpainting (Compatível com Baseline)

Status: aprovado para implementação incremental mantendo a versão atual funcional.

## 1) Objetivo

Evoluir o pretreino fundacional de panorâmicas de:

- reconstrução integral (`x -> x`)

para uma versão mais semântica:

- denoising/inpainting (`x_corrompida -> x_limpa`)

preservando integralmente o pipeline baseline existente.

## 2) Diretriz de Compatibilidade

1. O comportamento atual permanece como padrão.
2. A nova versão entra por configuração (`pretext.mode`), sem quebrar scripts antigos.
3. Checkpoints, logs, TensorBoard e `train_visuals` mantêm o mesmo contrato de saída.

## 3) Arquitetura

## 3.1 Modelo

- manter `PanoramicResNetAutoencoder` atual (sem skip-connections).
- nenhuma alteração obrigatória no encoder/decoder para a v1.

## 3.2 Mudança principal

- alterar apenas o pretexto de entrada no dataset:
  - `target`: imagem limpa (`y_clean`)
  - `input`: imagem corrompida (`x_corrupted`)
- forward permanece: `recon = model(x_corrupted)`.

## 4) Modos de Pretexto

Adicionar no config:

1. `identity`
   - baseline atual (`x_in = clean`).
2. `denoise`
   - ruído aditivo/multiplicativo.
3. `inpaint`
   - buracos/blocos/patch dropout.
4. `hybrid` (recomendado)
   - mistura de denoise + inpaint.

## 5) Estratégia de Corrupção (v1)

## 5.1 Denoise

- gaussian noise leve/moderado
- poisson leve
- speckle leve

## 5.2 Inpainting

- blocos retangulares (`coarse dropout`)
- patch dropout com cobertura alvo entre `10%` e `30%`

## 5.3 Regras

- sem flips
- intensidade sorteada por amostra
- gerar `corruption_mask` por amostra para:
  - visualização
  - loss ponderada opcional

## 6) Loss

## 6.1 Padrão

- `L_total = 0.7*L1 + 0.3*MSE`

## 6.2 Opcional (recomendada na v1.1)

- foco em região corrompida:
  - `L = w_corrupted * L_corrupted + w_clean * L_clean`
  - com `w_corrupted > w_clean`

## 7) Alterações de Código Planejadas

1. `panorama_foundation/dataset.py`
   - suportar `pretext.mode`
   - retornar `corruption_mask`
2. `panorama_foundation/train_autoencoder.py`
   - suportar loss ponderada por máscara (feature flag)
3. `panorama_foundation/training_callbacks.py`
   - incluir painéis com máscara de corrupção
4. novos configs dedicados
   - sem alterar o baseline existente

## 8) Configuração (Contrato Proposto)

Bloco novo em config:

```json
{
  "pretext": {
    "mode": "hybrid",
    "corruption_prob": 1.0,
    "noise": {
      "gaussian_std_min": 0.01,
      "gaussian_std_max": 0.06,
      "poisson_strength": 0.03,
      "speckle_strength": 0.03
    },
    "inpaint": {
      "coverage_min": 0.10,
      "coverage_max": 0.30,
      "min_holes": 4,
      "max_holes": 20
    }
  },
  "loss": {
    "focus_corrupted_regions": false,
    "w_corrupted": 0.8,
    "w_clean": 0.2
  }
}
```

## 9) Plano Experimental

Rodar com mesmo split/seed para comparação justa:

1. E0: `identity` (baseline)
2. E1: `denoise`
3. E2: `inpaint`
4. E3: `hybrid` (principal)

## 10) Critérios de Sucesso

1. Treino estável da nova versão com o mesmo pipeline de artefatos.
2. Baseline antigo continua executando sem mudanças.
3. Encoder da versão híbrida melhora transferência downstream em pelo menos uma tarefa sem regressão severa nas demais.

## 11) Riscos e Mitigações

1. Corrupção excessiva -> instabilidade
   - mitigação: curriculum de severidade
2. Corrupção fraca -> comportamento próximo de identity
   - mitigação: elevar cobertura/intensidade gradualmente
3. Ganho apenas em reconstrução visual, sem ganho de transferência
   - mitigação: decisão guiada por benchmark downstream

## 12) Rollout

1. Implementar por feature flags com default `identity`.
2. Validar com smoke visual.
3. Executar full999 em `hybrid`.
4. Avaliar transferência (classificação/regressão/segmentação) com protocolo freeze -> unfreeze.

