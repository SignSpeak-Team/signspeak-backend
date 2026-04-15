output "dns_name"                    { value = aws_lb.main.dns_name }
output "zone_id"                     { value = aws_lb.main.zone_id }
output "arn"                         { value = aws_lb.main.arn }
output "security_group_id"           { value = aws_security_group.alb.id }
output "api_gateway_target_group_arn" { value = aws_lb_target_group.api_gateway.arn }
