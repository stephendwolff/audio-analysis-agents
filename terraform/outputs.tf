output "bucket_name" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.media.id
}

output "bucket_region" {
  description = "Region of the S3 bucket"
  value       = aws_s3_bucket.media.region
}

output "aws_access_key_id" {
  description = "AWS Access Key ID for Railway"
  value       = aws_iam_access_key.app.id
}

output "aws_secret_access_key" {
  description = "AWS Secret Access Key for Railway"
  value       = aws_iam_access_key.app.secret
  sensitive   = true
}

# Output formatted for Railway environment variables
output "railway_env_vars" {
  description = "Environment variables to set in Railway"
  value       = <<-EOT

    Add these to Railway:

    AWS_STORAGE_BUCKET_NAME=${aws_s3_bucket.media.id}
    AWS_ACCESS_KEY_ID=${aws_iam_access_key.app.id}
    AWS_SECRET_ACCESS_KEY=<run: terraform output -raw aws_secret_access_key>
    AWS_S3_REGION_NAME=${aws_s3_bucket.media.region}
  EOT
}
