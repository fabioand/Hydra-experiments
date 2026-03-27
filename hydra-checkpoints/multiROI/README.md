# MultiROI Official Checkpoints (Local Only)

This directory defines the official local paths used by the MultiROI library defaults.

## Official local paths

- Center model:
  - `hydra-checkpoints/multiROI/center/center20shared_16k_stable3.ckpt`
- Lateral model:
  - `hydra-checkpoints/multiROI/lateral/lateral20_v1_fixedorient_nopres_absenthm1_16k_ft_stable_ep60_best.ckpt`

## Important

- `*.ckpt` files are intentionally ignored by Git in this repository.
- Keep only structure/docs/metadata in version control.
- Copy or download the checkpoint binaries locally before running MultiROI defaults.

## Typical workflow

1. Train large runs on EC2.
2. Select stable checkpoints for center/lateral.
3. Copy them to the paths above on local machine.
4. Run local eval/inference scripts that use MultiROI defaults.

## Optional integrity metadata

You can add a `checkpoints.lock.json` file under `hydra-checkpoints/multiROI/` with source path, SHA256 and run metadata.
