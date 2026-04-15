output "cluster_name" { value = aws_ecs_cluster.main.name }
output "cluster_arn"  { value = aws_ecs_cluster.main.arn }

output "api_gateway_service_name" {
  value = aws_ecs_service.api_gateway.name
}
output "vision_service_name" {
  value = aws_ecs_service.vision_service.name
}
output "translation_service_name" {
  value = aws_ecs_service.translation_service.name
}
output "ecs_security_group_id" {
  value = aws_security_group.ecs_services.id
}
