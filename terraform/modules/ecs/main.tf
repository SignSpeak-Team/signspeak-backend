################################################################################
# modules/ecs/main.tf
# ECS Cluster + Task Definitions + Services + Security Groups + CloudWatch Logs
################################################################################

locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# ── CloudWatch Log Groups ─────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "services" {
  for_each = toset(["api-gateway", "vision-service", "translation-service"])

  name              = "/ecs/${local.name_prefix}/${each.value}"
  retention_in_days = var.log_retention_days

  tags = {
    Service = each.value
  }
}

# ── ECS Cluster ───────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "${local.name_prefix}-cluster"
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name = aws_ecs_cluster.main.name

  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
    base              = 1
  }
}

# ── Security Group compartido para los servicios ECS ─────────────────────────
resource "aws_security_group" "ecs_services" {
  name        = "${local.name_prefix}-ecs-sg"
  description = "SG para los contenedores ECS. Solo acepta tráfico desde el ALB."
  vpc_id      = var.vpc_id

  # Permitir tráfico desde el ALB hacia api-gateway
  ingress {
    description     = "Desde ALB a api-gateway"
    from_port       = 7860
    to_port         = 7860
    protocol        = "tcp"
    security_groups = [var.alb_security_group_id]
  }

  # Comunicación interna entre servicios (todo el rango de puertos privados)
  ingress {
    description = "Tráfico interno entre microservicios"
    from_port   = 8000
    to_port     = 8002
    protocol    = "tcp"
    self        = true
  }

  egress {
    description = "Todo el tráfico saliente (ECR pull, internet, etc.)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${local.name_prefix}-ecs-sg"
  }
}

# ── Task Definition: api-gateway ──────────────────────────────────────────────
resource "aws_ecs_task_definition" "api_gateway" {
  family                   = "${local.name_prefix}-api-gateway"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.api_gateway_cpu
  memory                   = var.api_gateway_memory
  execution_role_arn       = var.task_execution_role_arn
  task_role_arn            = var.task_role_arn

  container_definitions = jsonencode([
    {
      name      = "api-gateway"
      image     = var.api_gateway_image
      essential = true

      portMappings = [
        { containerPort = 7860, protocol = "tcp" }
      ]

      environment = [
        { name = "TRANSLATION_SERVICE_URL", value = "http://localhost:8001" },
        { name = "VISION_SERVICE_URL", value = "http://localhost:8002" },
        { name = "PYTHONUNBUFFERED", value = "1" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.services["api-gateway"].name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:7860/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 30
      }
    }
  ])

  tags = {
    Name = "${local.name_prefix}-api-gateway-td"
  }
}

# ── Task Definition: vision-service ───────────────────────────────────────────
resource "aws_ecs_task_definition" "vision_service" {
  family                   = "${local.name_prefix}-vision-service"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.vision_service_cpu
  memory                   = var.vision_service_memory
  execution_role_arn       = var.task_execution_role_arn
  task_role_arn            = var.task_role_arn

  container_definitions = jsonencode([
    {
      name      = "vision-service"
      image     = var.vision_service_image
      essential = true

      portMappings = [
        { containerPort = 8000, protocol = "tcp" }
      ]

      environment = [
        { name = "PYTHONUNBUFFERED", value = "1" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.services["vision-service"].name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/api/v1/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60 # MediaPipe tarda en cargar
      }

      # Límites de recursos para el contenedor (ulimits para OpenCV/MediaPipe)
      ulimits = [
        { name = "nofile", softLimit = 65536, hardLimit = 65536 }
      ]
    }
  ])

  tags = {
    Name = "${local.name_prefix}-vision-service-td"
  }
}

# ── Task Definition: translation-service ─────────────────────────────────────
resource "aws_ecs_task_definition" "translation_service" {
  family                   = "${local.name_prefix}-translation-service"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.translation_cpu
  memory                   = var.translation_memory
  execution_role_arn       = var.task_execution_role_arn
  task_role_arn            = var.task_role_arn

  container_definitions = jsonencode([
    {
      name      = "translation-service"
      image     = var.translation_service_image
      essential = true

      portMappings = [
        { containerPort = 8001, protocol = "tcp" }
      ]

      environment = [
        { name = "VISION_SERVICE_URL", value = "http://localhost:8002" },
        { name = "PYTHONUNBUFFERED", value = "1" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.services["translation-service"].name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8001/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 30
      }
    }
  ])

  tags = {
    Name = "${local.name_prefix}-translation-service-td"
  }
}

# ── ECS Service: api-gateway ──────────────────────────────────────────────────
resource "aws_ecs_service" "api_gateway" {
  name            = "${local.name_prefix}-api-gateway"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api_gateway.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnets
    security_groups  = [aws_security_group.ecs_services.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = var.api_gateway_target_group_arn
    container_name   = "api-gateway"
    container_port   = 7860
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  deployment_controller {
    type = "ECS"
  }

  tags = {
    Name = "${local.name_prefix}-api-gateway-svc"
  }

  depends_on = [aws_ecs_cluster.main]
}

# ── ECS Service: vision-service ───────────────────────────────────────────────
resource "aws_ecs_service" "vision_service" {
  name            = "${local.name_prefix}-vision-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.vision_service.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnets
    security_groups  = [aws_security_group.ecs_services.id]
    assign_public_ip = false
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  tags = {
    Name = "${local.name_prefix}-vision-service-svc"
  }

  depends_on = [aws_ecs_cluster.main]
}

# ── ECS Service: translation-service ─────────────────────────────────────────
resource "aws_ecs_service" "translation_service" {
  name            = "${local.name_prefix}-translation-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.translation_service.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnets
    security_groups  = [aws_security_group.ecs_services.id]
    assign_public_ip = false
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  tags = {
    Name = "${local.name_prefix}-translation-service-svc"
  }

  depends_on = [aws_ecs_cluster.main]
}

# ── Auto Scaling para api-gateway ────────────────────────────────────────────
resource "aws_appautoscaling_target" "api_gateway" {
  max_capacity       = 4
  min_capacity       = 1
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api_gateway.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "api_gateway_cpu" {
  name               = "${local.name_prefix}-api-gw-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.api_gateway.resource_id
  scalable_dimension = aws_appautoscaling_target.api_gateway.scalable_dimension
  service_namespace  = aws_appautoscaling_target.api_gateway.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value       = 70.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60

    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}
