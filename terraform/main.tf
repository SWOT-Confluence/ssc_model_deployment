terraform {
  backend "s3" {
    encrypt = true
  }
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  default_tags {
    tags = local.default_tags
  }
  region = var.aws_region
}

# Data sources
data "aws_caller_identity" "current" {}

data "aws_efs_file_system" "input" {
  creation_token = "${var.prefix}-input"
}

data "aws_efs_file_system" "flpe" {
  creation_token = "${var.prefix}-flpe"
}


data "aws_iam_role" "job" {
  name = "${var.prefix}-batch-job-role"
}

data "aws_iam_role" "exec" {
  name = "${var.prefix}-ecs-exe-task-role"
}

# Local variables
locals {
  account_id = data.aws_caller_identity.current.account_id
  default_tags = length(var.default_tags) == 0 ? {
    application : var.app_name,
    environment : lower(var.environment),
    version : var.app_version
  } : var.default_tags
}

module "confluence-ssc-input" {
  source      = "./modules/ssc_model_deployment"
  app_name    = var.app_name
  app_version = var.app_version
  aws_region  = var.aws_region
  efs_file_system_ids = {
    input = data.aws_efs_file_system.input.file_system_id,
    flpe = data.aws_efs_file_system.flpe.file_system_id
  }
  environment            = var.environment
  iam_execution_role_arn = data.aws_iam_role.exec.arn
  iam_job_role_arn       = data.aws_iam_role.job.arn
  prefix                 = var.prefix
}
