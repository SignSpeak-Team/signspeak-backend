terraform {
  required_version = ">= 1.7.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
  }

  backend "s3" {
    bucket         = "signspeakterraformstate"
    key            = "signspeakbackend/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "signspeakterraformlock"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}


module "vpc" {
  source = "./modules/vpc"

  project_name       = var.project_name
  environment        = var.environment
  vpc_cidr           = var.vpc_cidr
  availability_zones = var.availability_zones
  public_subnets     = var.public_subnets
  private_subnets    = var.private_subnets
}


module "ecr" {
  source = "./modules/ecr"

  project_name = var.project_name
  environment  = var.environment
  services     = ["api-gateway", "vision-service", "translation-service"]
}


module "iam" {
  source = "./modules/iam"

  project_name = var.project_name
  environment  = var.environment
  aws_region   = var.aws_region
  account_id   = data.aws_caller_identity.current.account_id
}


module "alb" {
  source = "./modules/alb"

  project_name      = var.project_name
  environment       = var.environment
  vpc_id            = module.vpc.vpc_id
  public_subnet_ids = module.vpc.public_subnet_ids
  certificate_arn   = var.certificate_arn
}


module "ecs" {
  source = "./modules/ecs"

  project_name    = var.project_name
  environment     = var.environment
  aws_region      = var.aws_region
  vpc_id          = module.vpc.vpc_id
  private_subnets = module.vpc.private_subnet_ids
  public_subnets  = module.vpc.public_subnet_ids

  # Imágenes ECR
  api_gateway_image      = "${module.ecr.repository_urls["api-gateway"]}:${var.image_tag}"
  vision_service_image   = "${module.ecr.repository_urls["vision-service"]}:${var.image_tag}"
  translation_service_image = "${module.ecr.repository_urls["translation-service"]}:${var.image_tag}"

  # IAM
  task_execution_role_arn = module.iam.task_execution_role_arn
  task_role_arn           = module.iam.task_role_arn

  # ALB target groups
  api_gateway_target_group_arn = module.alb.api_gateway_target_group_arn
  alb_security_group_id        = module.alb.security_group_id

  # Capacidad de los servicios
  api_gateway_cpu        = var.api_gateway_cpu
  api_gateway_memory     = var.api_gateway_memory
  vision_service_cpu     = var.vision_service_cpu
  vision_service_memory  = var.vision_service_memory
  translation_cpu        = var.translation_cpu
  translation_memory     = var.translation_memory

  # Configuración
  log_retention_days = var.log_retention_days
}


data "aws_caller_identity" "current" {}
