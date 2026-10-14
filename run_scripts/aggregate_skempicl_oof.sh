#!/usr/bin/env bash
set -euo pipefail

python scripts/ensemble_seed_predictions.py \
  outputs/SKEMPI_CL/seed20260911 \
  outputs/SKEMPI_CL/seed20260912 \
  outputs/SKEMPI_CL/seed20260913 \
  outputs/SKEMPI_CL/seed20260914 \
  outputs/SKEMPI_CL/seed20260915 \
  --output-dir outputs/SKEMPI_CL/five_seed_oof
