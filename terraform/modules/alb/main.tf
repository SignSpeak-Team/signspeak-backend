################################################################################
# modules/alb/main.tf
# Application Load Balancer público + security group + target groups
################################################################################

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  use_https   = var.certificate_arn != ""
}

# ── Security Group del ALB ────────────────────────────────────────────────────
resource "aws_security_group" "alb" {
  name        = "${local.name_prefix}-alb-sg"
  description = "Security group del ALB de SignSpeak. Permite HTTP(S) público."
  vpc_id      = var.vpc_id

  ingress {
    description = "HTTP público"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  dynamic "ingress" {
    for_each = local.use_https ? [1] : []
    content {
      description = "HTTPS público"
      from_port   = 443
      to_port     = 443
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }

  egress {
    description = "Todo el tráfico saliente"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${local.name_prefix}-alb-sg"
  }
}

# ── ALB ───────────────────────────────────────────────────────────────────────
resource "aws_lb" "main" {
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnet_ids

  enable_deletion_protection = false # Pon true en prod

  tags = {
    Name = "${local.name_prefix}-alb"
  }
}

# ── Target Group: api-gateway ─────────────────────────────────────────────────
resource "aws_lb_target_group" "api_gateway" {
  name        = "${local.name_prefix}-api-gw-tg"
  port        = 7860
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip" # Requerido para Fargate

  health_check {
    enabled             = true
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }

  tags = {
    Name = "${local.name_prefix}-api-gw-tg"
  }
}

# ── Listener HTTP (siempre activo) ────────────────────────────────────────────
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  # Si hay HTTPS, redirige. Si no, reenvía al target group.
  dynamic "default_action" {
    for_each = local.use_https ? [1] : []
    content {
      type = "redirect"
      redirect {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
  }

  dynamic "default_action" {
    for_each = local.use_https ? [] : [1]
    content {
      type             = "forward"
      target_group_arn = aws_lb_target_group.api_gateway.arn
    }
  }
}

# ── Listener HTTPS (sólo si hay certificado) ──────────────────────────────────
resource "aws_lb_listener" "https" {
  count = local.use_https ? 1 : 0

  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api_gateway.arn
  }
}
