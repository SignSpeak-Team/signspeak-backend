#!/usr/bin/env bash
# =============================================================================
# scripts/deploy.sh
# Wrapper de despliegue completo: build → push → terraform plan → apply
# Uso: ./scripts/deploy.sh [IMAGE_TAG] [--auto-approve]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="$SCRIPT_DIR/.."
IMAGE_TAG="${1:-$(git rev-parse --short HEAD 2>/dev/null || echo "latest")}"
AUTO_APPROVE="${2:-}"

echo "======================================================"
echo "  🚀 SignSpeak Backend — Deploy"
echo "  Tag: $IMAGE_TAG"
echo "======================================================"

# 1. Build y push
echo ""
echo "── Paso 1/3: Build & Push de imágenes ──────────────"
bash "$SCRIPT_DIR/build_and_push.sh" "$IMAGE_TAG"

# 2. Terraform init (idempotente)
echo ""
echo "── Paso 2/3: Terraform Init ─────────────────────────"
cd "$TF_DIR"
terraform init -upgrade

# 3. Plan
echo ""
echo "── Paso 3/3: Terraform Apply ────────────────────────"
terraform plan \
  -var="image_tag=$IMAGE_TAG" \
  -out=tfplan

if [[ "$AUTO_APPROVE" == "--auto-approve" ]]; then
  terraform apply tfplan
else
  echo ""
  read -r -p "¿Aplicar el plan? (y/N): " CONFIRM
  if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    terraform apply tfplan
  else
    echo "Deploy cancelado."
    exit 0
  fi
fi

echo ""
echo "======================================================"
echo "  ✅ Deploy completado"
terraform output
echo "======================================================"
