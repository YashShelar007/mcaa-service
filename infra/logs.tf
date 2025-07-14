// infra/logs.tf

resource "aws_cloudwatch_log_group" "sfn" {
  name              = "/aws/vendedlogs/states/mcaa-service-pipeline"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "baseline" {
  name              = "/mcaa-service/baseline"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "prune_structured" {
  name              = "/mcaa-service/prune_structured"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "quantize" {
  name              = "/mcaa-service/quantize"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "distill_kd" {
  name              = "/mcaa-service/distill_kd"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "evaluate" {
  name              = "/mcaa-service/evaluate"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "prune_search" {
  name              = "/mcaa-service/prune_search"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "prune_and_quantize" {
  name              = "/mcaa-service/prune_and_quantize"
  retention_in_days = 14
}
