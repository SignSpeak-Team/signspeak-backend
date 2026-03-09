################################################################################
# modules/iam/main.tf
# Roles IAM para ECS: task execution role + task role
################################################################################

locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# ── Trust policy base para ECS tasks ─────────────────────────────────────────
data "aws_iam_policy_document" "ecs_task_assume_role" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

# ── Execution Role (para que ECS pueda pull imágenes ECR y escribir logs) ─────
resource "aws_iam_role" "task_execution" {
  name               = "${local.name_prefix}-ecs-execution-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json

  tags = {
    Name = "${local.name_prefix}-ecs-execution-role"
  }
}

resource "aws_iam_role_policy_attachment" "task_execution_policy" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Permiso extra: leer secretos de Secrets Manager / SSM (para env vars sensibles)
resource "aws_iam_policy" "task_execution_secrets" {
  name        = "${local.name_prefix}-ecs-execution-secrets"
  description = "Permite al ECS execution role leer secretos de SSM y Secrets Manager"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameters",
          "ssm:GetParameter",
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          "arn:aws:ssm:${var.aws_region}:${var.account_id}:parameter/${var.project_name}/*",
          "arn:aws:secretsmanager:${var.aws_region}:${var.account_id}:secret:${var.project_name}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "task_execution_secrets_attach" {
  role       = aws_iam_role.task_execution.name
  policy_arn = aws_iam_policy.task_execution_secrets.arn
}

# ── Task Role (permisos que el código de la aplicación necesita en runtime) ───
resource "aws_iam_role" "task_role" {
  name               = "${local.name_prefix}-ecs-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json

  tags = {
    Name = "${local.name_prefix}-ecs-task-role"
  }
}

# Permisos de aplicación: CloudWatch Logs + S3 (modelos ML, si aplica)
resource "aws_iam_policy" "task_app_permissions" {
  name        = "${local.name_prefix}-ecs-task-app-permissions"
  description = "Permisos de runtime para los contenedores de SignSpeak"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/ecs/${var.project_name}/*"
      },
      {
        Sid    = "S3ModelsBucket"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.project_name}-models",
          "arn:aws:s3:::${var.project_name}-models/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "task_app_permissions_attach" {
  role       = aws_iam_role.task_role.name
  policy_arn = aws_iam_policy.task_app_permissions.arn
}
