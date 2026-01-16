terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# S3 Bucket for media uploads
resource "aws_s3_bucket" "media" {
  bucket = var.bucket_name

  tags = {
    Name        = "Audio Analysis Media"
    Environment = var.environment
  }
}

# Block public access (files accessed via signed URLs or made public via policy)
resource "aws_s3_bucket_public_access_block" "media" {
  bucket = aws_s3_bucket.media.id

  block_public_acls       = !var.allow_public_read
  block_public_policy     = !var.allow_public_read
  ignore_public_acls      = !var.allow_public_read
  restrict_public_buckets = !var.allow_public_read
}

# Bucket policy for public read (optional)
resource "aws_s3_bucket_policy" "media_public_read" {
  count  = var.allow_public_read ? 1 : 0
  bucket = aws_s3_bucket.media.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGetObject"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.media.arn}/media/*"
      }
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.media]
}

# CORS configuration for browser uploads
resource "aws_s3_bucket_cors_configuration" "media" {
  bucket = aws_s3_bucket.media.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT", "POST", "DELETE", "HEAD"]
    allowed_origins = var.cors_allowed_origins
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

# IAM user for the application
resource "aws_iam_user" "app" {
  name = "${var.bucket_name}-user"

  tags = {
    Name        = "Audio Analysis App User"
    Environment = var.environment
  }
}

# IAM policy for S3 access
resource "aws_iam_user_policy" "app_s3" {
  name = "${var.bucket_name}-s3-policy"
  user = aws_iam_user.app.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ListBucket"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = aws_s3_bucket.media.arn
      },
      {
        Sid    = "ReadWriteObjects"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:GetObjectAcl",
          "s3:PutObjectAcl"
        ]
        Resource = "${aws_s3_bucket.media.arn}/*"
      }
    ]
  })
}

# Access key for the IAM user
resource "aws_iam_access_key" "app" {
  user = aws_iam_user.app.name
}
