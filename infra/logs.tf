// infra/logs.tf

resource "aws_cloudwatch_log_group" "sfn" {
  name              = "/aws/vendedlogs/states/mcaa-service-pipeline"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "baseline" {
  name              = "/mcaa-service/baseline"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "prune" {
  name              = "/mcaa-service/prune"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "quantize" {
  name              = "/mcaa-service/quantize"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "distill" {
  name              = "/mcaa-service/distill"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "evaluator" {
  name              = "/mcaa-service/evaluator"
  retention_in_days = 14
}
