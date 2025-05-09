variable "app_name" {
  type        = string
  description = "Application name"
  default     = "confluence"
}

variable "app_version" {
  type        = string
  description = "The application version number"
}
variable "aws_region" {
  type        = string
  description = "AWS region to deploy to"
  default     = "us-west-2"
}

variable "efs_file_system_ids" {
  type        = map(string)
  description = "Map of EFS file system ids to pass to the container definition"
}

variable "environment" {
  type        = string
  description = "The environment in which to deploy to"
}

variable "iam_execution_role_arn" {
  type        = string
  description = "The IAM ARN of the execution role"
}

variable "iam_job_role_arn" {
  type        = string
  description = "The IAM ARN of the job role"
}

variable "image_tag" {
  type        = string
  description = "The container image tag to utilize"
  default     = "latest"
}

variable "prefix" {
  type        = string
  description = "Prefix to add to all AWS resources as a unique identifier"
}
