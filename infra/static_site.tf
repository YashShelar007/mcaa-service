// infra/static_site.tf

// S3 bucket to host the UI
resource "aws_s3_bucket" "ui" {
  bucket = "mcaa-service-ui"

  tags = {
    Name        = "MCaaS Static UI"
    Environment = "Production"
  }
}

// Block all public ACLs
resource "aws_s3_bucket_public_access_block" "ui" {
  bucket                  = aws_s3_bucket.ui.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

// Ownership controls (disable ACLs)
resource "aws_s3_bucket_ownership_controls" "ui" {
  bucket = aws_s3_bucket.ui.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

// Enable website hosting on the bucket
resource "aws_s3_bucket_website_configuration" "ui" {
  bucket = aws_s3_bucket.ui.id

  index_document {
    suffix = "index.html"
  }

  error_document {
    key = "error.html"
  }
}

// Origin Access Control (newer than OAI)
resource "aws_cloudfront_origin_access_control" "ui_oac" {
  name                              = "mcaa-service-ui-oac"
  description                       = "OAC for MCaaS Static UI bucket"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

// CloudFront distribution fronting the site
resource "aws_cloudfront_distribution" "ui" {
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"

  origin {
    domain_name              = aws_s3_bucket.ui.bucket_regional_domain_name
    origin_id                = "S3-mcaa-service-ui"
    origin_access_control_id = aws_cloudfront_origin_access_control.ui_oac.id
  }

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-mcaa-service-ui"

    viewer_protocol_policy = "redirect-to-https"
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
  }

  price_class = "PriceClass_100"

  viewer_certificate {
    cloudfront_default_certificate = true
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  tags = {
    Name        = "MCaaS UI CDN"
    Environment = "Production"
  }
}

// Bucket policy to allow only CloudFront (via OAC) to get objects
resource "aws_s3_bucket_policy" "ui" {
  bucket = aws_s3_bucket.ui.id
  policy = data.aws_iam_policy_document.ui_s3_policy.json
}

data "aws_iam_policy_document" "ui_s3_policy" {
  statement {
    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.ui.arn}/*"]
    condition {
      test     = "StringEquals"
      variable = "AWS:SourceArn"
      values   = [aws_cloudfront_distribution.ui.arn]
    }
  }
}

output "ui_distribution_id" {
  description = "CloudFront Distribution ID for the MCaaS UI"
  value       = aws_cloudfront_distribution.ui.id
}