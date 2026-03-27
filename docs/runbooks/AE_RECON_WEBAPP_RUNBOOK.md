# Runbook - AE Reconstruction Web App

Projeto:
- `/Users/fabioandrade/hydra/ae_recon_webapp`

## 1) Objetivo operacional

Disponibilizar reconstrução de imagem com autoencoder via web, com:
- upload do usuário,
- seleção de imagens de exemplo no servidor,
- sliders de ajuste em tempo real no cliente.

## 2) Checklist de pré-subida

1. Checkpoint existe:
   - `ae_radiograph_filter/models/ae_identity_bestE21.ckpt`
2. Virtualenv disponível (recomendado):
   - `/Users/fabioandrade/hydra/ae_recon_webapp/.venv`
3. Pasta de samples existe e tem imagens, se desejado:
   - `ae_recon_webapp/sample_images/`
4. Variáveis configuradas em `ae_recon_webapp/.env`.
5. Porta `8000` (API) e `5173`/`8080` (front) livres.

## 3) Subida local rápida

Backend:
```bash
cd /Users/fabioandrade/hydra
python3 -m venv ae_recon_webapp/.venv
./ae_recon_webapp/.venv/bin/python -m pip install -r ae_recon_webapp/backend/requirements.txt
bash ae_recon_webapp/scripts/run_backend.sh
```

Frontend:
```bash
cd /Users/fabioandrade/hydra
bash ae_recon_webapp/scripts/run_frontend.sh
```

## 4) Smoke test funcional

1. `GET /healthz` responde `status=ok`.
2. `GET /v1/samples` lista arquivos esperados.
3. Upload de uma imagem via UI gera:
   - canvas Original,
   - canvas Reconstrução,
   - canvas Enhanced.
4. Mudança de `fs`/`fa` altera imagem Enhanced sem nova chamada de inferência.

## 5) Deploy EC2 (Linux)

Opção A: processo direto
1. Subir backend por `uvicorn` sob systemd.
2. Build do frontend (`VITE_API_BASE=/api npm run build`) e servir estático por Nginx.
3. Usar templates:
   - `ae_recon_webapp/deploy/ae-recon-backend.service`
   - `ae_recon_webapp/deploy/nginx-ae-recon.conf`

Opção B: Docker
1. `docker compose up --build` em `ae_recon_webapp/`.
2. Expor portas ou colocar Nginx/ALB na frente.

## 6) Diagnóstico rápido

Falha comum: checkpoint não encontrado
- Sintoma: API não sobe ou `/healthz` falha na inicialização.
- Ação: corrigir `AE_RECON_CKPT`.

Falha comum: CORS no frontend
- Sintoma: browser bloqueia chamadas.
- Ação: ajustar `AE_RECON_ALLOWED_ORIGINS`.

Falha comum: sem samples
- Sintoma: combo de samples vazio.
- Ação: validar `AE_RECON_SAMPLE_IMAGES_DIR` e formatos aceitos.
