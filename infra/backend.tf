terraform {
  backend "s3" {
    bucket         = "mcaa-service-tf-state"
    key            = "terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "mcaa-service-tf-locks"
    encrypt        = true
  }
}
