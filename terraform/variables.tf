################################################################################
# variables.tf — Variables de entrada del root module
################################################################################

# ── General ──────────────────────────────────────────────────────────────────
variable "project_name" {
  description = "Nombre del proyecto. Se usa como prefijo en todos los recursos AWS."
  type        = string
  default     = "signspeakbackend"
}

variable "environment" {
  description = "Entorno de despliegue: dev | staging | prod"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "El entorno debe ser dev, staging o prod."
  }
}

variable "aws_region" {
  description = "Región de AWS donde se desplegará la infraestructura."
  type        = string
  default     = "us-east-1"
}

# ── Red / VPC ─────────────────────────────────────────────────────────────────
variable "vpc_cidr" {
  description = "Bloque CIDR para la VPC."
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Lista de Availability Zones a usar (mínimo 2 para alta disponibilidad)."
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

variable "public_subnets" {
  description = "CIDRs de subnets públicas (una por AZ). Usadas por el ALB."
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnets" {
  description = "CIDRs de subnets privadas (una por AZ). Usadas por ECS Fargate."
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24"]
}

# ── Imágenes ──────────────────────────────────────────────────────────────────
variable "image_tag" {
  description = "Tag de imagen Docker a desplegar. Normalmente el SHA del commit."
  type        = string
  default     = "latest"
}

# ── TLS / HTTPS ───────────────────────────────────────────────────────────────
variable "certificate_arn" {
  description = "ARN del certificado ACM para HTTPS. Dejar vacío ('') para solo HTTP."
  type        = string
  default     = ""
}

variable "domain_name" {
  description = "Dominio raíz en Route53 (ej: signspeakapp.com). Opcional."
  type        = string
  default     = ""
}

# ── ECS: api-gateway ──────────────────────────────────────────────────────────
variable "api_gateway_cpu" {
  description = "vCPU para el task de api-gateway (256 = 0.25 vCPU)."
  type        = number
  default     = 256
}

variable "api_gateway_memory" {
  description = "RAM en MB para el task de api-gateway."
  type        = number
  default     = 512
}

# ── ECS: vision-service ───────────────────────────────────────────────────────
variable "vision_service_cpu" {
  description = "vCPU para el task de vision-service (1024 = 1 vCPU)."
  type        = number
  default     = 1024
}

variable "vision_service_memory" {
  description = "RAM en MB para vision-service. Mínimo 2048 por OpenCV/MediaPipe."
  type        = number
  default     = 2048
}

# ── ECS: translation-service ──────────────────────────────────────────────────
variable "translation_cpu" {
  description = "vCPU para el task de translation-service."
  type        = number
  default     = 256
}

variable "translation_memory" {
  description = "RAM en MB para translation-service."
  type        = number
  default     = 512
}

# ── Logging ───────────────────────────────────────────────────────────────────
variable "log_retention_days" {
  description = "Días de retención de logs en CloudWatch."
  type        = number
  default     = 14
}
