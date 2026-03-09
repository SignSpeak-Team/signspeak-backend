variable "project_name"                 { type = string }
variable "environment"                  { type = string }
variable "aws_region"                   { type = string }
variable "vpc_id"                       { type = string }
variable "private_subnets"              { type = list(string) }
variable "public_subnets"               { type = list(string) }
variable "api_gateway_image"            { type = string }
variable "vision_service_image"         { type = string }
variable "translation_service_image"    { type = string }
variable "task_execution_role_arn"      { type = string }
variable "task_role_arn"                { type = string }
variable "api_gateway_target_group_arn" { type = string }
variable "alb_security_group_id"        { type = string }

variable "api_gateway_cpu" {
  type    = number
  default = 256
}

variable "api_gateway_memory" {
  type    = number
  default = 512
}

variable "vision_service_cpu" {
  type    = number
  default = 1024
}

variable "vision_service_memory" {
  type    = number
  default = 2048
}

variable "translation_cpu" {
  type    = number
  default = 256
}

variable "translation_memory" {
  type    = number
  default = 512
}

variable "log_retention_days" {
  type    = number
  default = 14
}
