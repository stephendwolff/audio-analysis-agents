variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "Name of the S3 bucket for media uploads"
  type        = string
}

variable "environment" {
  description = "Environment name (e.g., production, staging)"
  type        = string
  default     = "production"
}

variable "allow_public_read" {
  description = "Allow public read access to uploaded files"
  type        = bool
  default     = true
}

variable "cors_allowed_origins" {
  description = "Origins allowed for CORS (e.g., your Railway app URL)"
  type        = list(string)
  default     = ["*"]
}
