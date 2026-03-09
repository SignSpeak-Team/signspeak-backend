################################################################################
# outputs.tf — Outputs útiles tras el apply
################################################################################

output "alb_dns_name" {
  description = "DNS del Application Load Balancer. Apunta tu dominio aquí con un CNAME."
  value       = module.alb.dns_name
}

output "alb_zone_id" {
  description = "Zone ID del ALB (útil para alias records en Route53)."
  value       = module.alb.zone_id
}

output "ecr_repository_urls" {
  description = "URLs de los repositorios ECR por servicio."
  value       = module.ecr.repository_urls
}

output "ecs_cluster_name" {
  description = "Nombre del cluster ECS."
  value       = module.ecs.cluster_name
}

output "ecs_cluster_arn" {
  description = "ARN del cluster ECS."
  value       = module.ecs.cluster_arn
}

output "vpc_id" {
  description = "ID de la VPC creada."
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "IDs de las subnets privadas (ECS)."
  value       = module.vpc.private_subnet_ids
}

output "public_subnet_ids" {
  description = "IDs de las subnets públicas (ALB)."
  value       = module.vpc.public_subnet_ids
}

output "task_execution_role_arn" {
  description = "ARN del rol de ejecución de ECS tasks."
  value       = module.iam.task_execution_role_arn
}
