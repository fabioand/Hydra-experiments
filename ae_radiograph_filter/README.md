# AE Radiograph Filter

Projeto para filtrar radiografias usando erro de reconstrução de um autoencoder.

## Estrutura

- `models/ae_identity_bestE21.ckpt`: checkpoint importado da EC2.
- `scripts/run_filter.py`: inferência + ranking + filtro.
- `scripts/run_sample_local.sh`: execução pronta em subset local.
- `outputs/`: resultados gerados.

## Saídas do filtro

Para cada execução, em `outputs/<run_name>/`:

- `scores_all.csv`: score por imagem (`mae`, `mse`), ordenado por pior erro.
- `scores_flagged.csv`: imagens acima do limiar (percentil configurado).
- `summary.json`: resumo da execução.
- `panels/top_errors/*.png`: painéis das piores reconstruções.
- `panels/low_errors/*.png`: painéis das melhores reconstruções.

## Rodar em algumas radiografias locais

```bash
cd /Users/fabioandrade/hydra
./ae_radiograph_filter/scripts/run_sample_local.sh
```

