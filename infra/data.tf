// infra/data.tf

// 1) Grab your default VPC
data "aws_vpc" "default" {
  default = true
}

// 2) List its subnets
data "aws_subnet_ids" "default" {
  vpc_id = data.aws_vpc.default.id
}

// 3) Pull the default Security Group
data "aws_security_group" "default" {
  name   = "default"
  vpc_id = data.aws_vpc.default.id
}
