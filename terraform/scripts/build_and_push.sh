#!/usr/bin/env bash
# =============================================================================
# scripts/build_and_push.sh
# Build de imágenes Docker y push a ECR para todos los microservicios
# Uso: ./scripts/build_and_push.sh [IMAGE_TAG]
# =============================================================================
set -euo pipefail

# ── Configuración ─────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE_TAG="${1:-$(git rev-parse --short HEAD 2>/dev/null || echo "latest")}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"   # raíz de signSpeak/

# Leer outputs de Terraform para obtener las URLs de ECR
TF_DIR="$SCRIPT_DIR/.."
pushd "$TF_DIR" > /dev/null

API_GATEWAY_REPO=$(terraform output -raw ecr_repository_urls 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['api-gateway'])" 2>/dev/null || echo "")
VISION_REPO=$(terraform output -raw ecr_repository_urls 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['vision-service'])" 2>/dev/null || echo "")
TRANSLATION_REPO=$(terraform output -raw ecr_repository_urls 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['translation-service'])" 2>/dev/null || echo "")

popd > /dev/null

# Si no hay outputs de Terraform aún, construye las URLs manualmente
PROJECT_NAME="${PROJECT_NAME:-signspeakbackend-dev}"
if [[ -z "$API_GATEWAY_REPO" ]]; then
  API_GATEWAY_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-api-gateway"
  VISION_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-vision-service"
  TRANSLATION_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-translation-service"
fi

echo "📦 Usando tag de imagen: $IMAGE_TAG"
echo "🏗️  AWS Account: $AWS_ACCOUNT_ID | Región: $AWS_REGION"

# ── Login a ECR ───────────────────────────────────────────────────────────────
echo ""
echo "🔐 Autenticando con ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# ── Función: build + tag + push ───────────────────────────────────────────────
build_push() {
  local SERVICE_NAME="$1"
  local DOCKERFILE_DIR="$2"
  local REPO_URL="$3"

  echo ""
  echo "🐳 [$SERVICE_NAME] Building..."
  docker build \
    --platform linux/amd64 \
    -t "${SERVICE_NAME}:${IMAGE_TAG}" \
    "$DOCKERFILE_DIR"

  echo "🏷️  [$SERVICE_NAME] Tagging → $REPO_URL:$IMAGE_TAG"
  docker tag "${SERVICE_NAME}:${IMAGE_TAG}" "${REPO_URL}:${IMAGE_TAG}"
  docker tag "${SERVICE_NAME}:${IMAGE_TAG}" "${REPO_URL}:latest"

  echo "🚀 [$SERVICE_NAME] Pushing..."
  docker push "${REPO_URL}:${IMAGE_TAG}"
  docker push "${REPO_URL}:latest"

  echo "✅ [$SERVICE_NAME] OK → ${REPO_URL}:${IMAGE_TAG}"
}

# ── Build y push de cada servicio ─────────────────────────────────────────────
build_push "api-gateway"        "$REPO_ROOT/services/api_gateway"        "$API_GATEWAY_REPO"
build_push "vision-service"     "$REPO_ROOT/services/vision_service"     "$VISION_REPO"
build_push "translation-service" "$REPO_ROOT/services/translation_service" "$TRANSLATION_REPO"

echo ""
echo "🎉 Todas las imágenes subidas con tag: $IMAGE_TAG"
echo ""
echo "Para desplegar, ejecuta:"
echo "  cd terraform && terraform apply -var='image_tag=$IMAGE_TAG'"
