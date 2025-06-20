// infra/main.tf

resource "aws_s3_bucket" "models" {
  bucket = "mcaa-service-model-storage"

  tags = {
    Name        = "MCaaS Model Storage"
    Environment = "Production"
  }
}

// Block all public access (no ACLs, no public policies)
resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

// Enforce bucket-owner-only access (disables ACLs entirely)
resource "aws_s3_bucket_ownership_controls" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

// Separate versioning resource
resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

// Separate acl resource
resource "aws_s3_bucket_acl" "models" {
  depends_on = [aws_s3_bucket_ownership_controls.models]

  bucket = aws_s3_bucket.models.id
  acl    = "private"
}

// Separate server-side encryption resource
resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_cors_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT", "POST", "HEAD"]
    allowed_origins = ["*"]      # ← you can tighten this to your CloudFront URL
    expose_headers  = ["ETag"]   # so the browser can see the uploaded object’s ETag
    max_age_seconds = 3000
  }
}


// DynamoDB table unchanged
resource "aws_dynamodb_table" "metadata" {
  name         = "mcaa-service-metadata"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "ModelID"

  attribute {
    name = "ModelID"
    type = "S"
  }

  tags = {
    Name        = "MCaaS Model Metadata"
    Environment = "Production"
  }
}
