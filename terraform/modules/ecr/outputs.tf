output "repository_urls" {
  description = "Mapa de nombre de servicio → URL del repositorio ECR"
  value       = { for k, v in aws_ecr_repository.services : k => v.repository_url }
}

output "repository_arns" {
  description = "Mapa de nombre de servicio → ARN del repositorio ECR"
  value       = { for k, v in aws_ecr_repository.services : k => v.arn }
}
