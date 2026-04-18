# SignSpeak — Infraestructura como Código (Terraform / AWS)

Este directorio contiene toda la IaC necesaria para desplegar el backend de SignSpeak
en AWS usando **ECS Fargate** + **ALB** + **ECR** + **VPC** con subnets públicas/privadas.

## Arquitectura

```
Internet
   │
   ▼
[ALB] (puerto 443 / 80→redirect)
   │
   ├──► ECS Service: api-gateway     (Fargate, puerto 7860)
   │         │
   │         ├──► [internal ALB / Service Connect]
   │         │
   │         ├──► ECS Service: vision-service      (Fargate, puerto 8000, 2 GB RAM)
   │         └──► ECS Service: translation-service (Fargate, puerto 8001)
   │
[Route 53] → dominio personalizado (opcional)
[ACM]       → certificado TLS
[ECR]       → repositorios de imágenes Docker
[VPC]       → subnets públicas (ALB) + privadas (ECS)
```

## Estructura de archivos

```
terraform/
├── README.md
├── main.tf              # Provider + backend remoto (S3)
├── variables.tf         # Todas las variables de entrada
├── outputs.tf           # Outputs útiles tras el apply
├── terraform.tfvars.example  # Ejemplo de valores (no commitear .tfvars real)
│
├── modules/
│   ├── vpc/             # VPC, subnets, IGW, NAT GW, route tables
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   │
│   ├── ecr/             # Repositorios ECR para cada servicio
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   │
│   ├── ecs/             # Cluster ECS, task definitions, services
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   │
│   ├── alb/             # Application Load Balancer + target groups
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   │
│   └── iam/             # Roles y políticas IAM para ECS
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
│
└── scripts/
    ├── build_and_push.sh   # Build + push de imágenes a ECR
    └── deploy.sh           # Wrapper: terraform plan + apply
```

## Requisitos previos

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.7
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) configurado (`aws configure`)
- Docker instalado y corriendo
- Bucket S3 para el backend remoto de Terraform (ver `main.tf`)

## Uso rápido

```bash
# 1. Copiar y editar variables
cp terraform.tfvars.example terraform.tfvars
# Edita terraform.tfvars con tus valores reales

# 2. Inicializar
terraform init

# 3. Planificar
terraform plan -out=tfplan

# 4. Aplicar
terraform apply tfplan

# 5. (Opcional) Build y push de imágenes
chmod +x scripts/build_and_push.sh
./scripts/build_and_push.sh
```

## Variables importantes

| Variable                | Descripción                                 |
| ----------------------- | ------------------------------------------- |
| `aws_region`            | Región AWS (ej: `us-east-1`)                |
| `project_name`          | Prefijo para todos los recursos             |
| `environment`           | `dev`, `staging`, `prod`                    |
| `vision_service_memory` | MB de RAM para vision-service (mínimo 2048) |
| `domain_name`           | Dominio Route53 (opcional)                  |

## Destruir infraestructura

```bash
terraform destroy
```

> ⚠️ Esto eliminará **todos** los recursos. Los repositorios ECR se destruyen con sus imágenes si `force_delete = true`.

Hecho con ❤️ por

Alan de los Santos Lopez Cetina — Matrícula: 2202116
Ángel Jonás Rosales Gonzales — Matrícula: 2202022
José Arturo González Flores — Matrícula: 2202012
Cesar Enrique Bernal Zurita— Matrícula: 2201100
Ángel David Quintana Pacheco — Matrícula: 2102165
Cristian Daniel Lázaro Acosta — Matrícula: 2202055
Ángel Adrián Yam Huchim — Matricula: 2202109
