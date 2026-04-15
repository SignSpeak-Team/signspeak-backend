################################################################################
# modules/ecr/main.tf
# Repositorios ECR privados para cada microservicio
################################################################################

locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

resource "aws_ecr_repository" "services" {
  for_each = toset(var.services)

  name                 = "${local.name_prefix}-${each.value}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true # Permite destruir aunque tenga imágenes

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Name    = "${local.name_prefix}-${each.value}"
    Service = each.value
  }
}

# ── Política de lifecycle: conservar sólo las últimas 10 imágenes ──────────────
resource "aws_ecr_lifecycle_policy" "cleanup" {
  for_each   = aws_ecr_repository.services
  repository = each.value.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Eliminar imágenes no-tagged más antiguas (conservar 10)"
        selection = {
          tagStatus   = "untagged"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Conservar sólo las últimas 20 imágenes tagged"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v", "sha-", "latest"]
          countType     = "imageCountMoreThan"
          countNumber   = 20
        }
        action = { type = "expire" }
      }
    ]
  })
}
