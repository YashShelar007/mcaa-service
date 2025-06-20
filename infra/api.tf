// infra/api.tf

// IAM role for Lambda execution
data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_exec" {
  name               = "mcaa-service-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy" "lambda_logging" {
  name   = "mcaa-service-lambda-logging"
  role   = aws_iam_role.lambda_exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Effect   = "Allow"
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

// Replace your existing aws_iam_role_policy "lambda_sfn" with this:
resource "aws_iam_role_policy" "lambda_sfn" {
  name = "mcaa-service-lambda-sfn"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "states:StartExecution",
          "states:GetExecutionHistory",
          "states:DescribeExecution"
        ]
        Resource = [
          // the state machine itself:
          aws_sfn_state_machine.pipeline.arn,
          // all executions of that state machine
          // by swapping "stateMachine" → "execution" and appending :*
          "${replace(aws_sfn_state_machine.pipeline.arn, ":stateMachine:", ":execution:")}:*"
        ]
      }
    ]
  })
}


// Lambda function for API
resource "aws_lambda_function" "api" {
  function_name = "mcaa-service-api"
  role          = aws_iam_role.lambda_exec.arn
  handler       = "main.handler"
  runtime       = "python3.9"
  timeout       = 10

  filename         = "${path.module}/lambda/api.zip"  // we’ll build this package
  source_code_hash = filebase64sha256("${path.module}/lambda/api.zip")

  environment {
    variables = {
      STATE_MACHINE_ARN = aws_sfn_state_machine.pipeline.arn
      MODEL_BUCKET      = aws_s3_bucket.models.bucket
    }
  }
}

// HTTP API Gateway
resource "aws_apigatewayv2_api" "api" {
  name          = "mcaa-service-http-api"
  protocol_type = "HTTP"
  
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["OPTIONS", "GET", "POST"]
    allow_headers = ["Content-Type"]
  }
}

resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id           = aws_apigatewayv2_api.api.id
  integration_type = "AWS_PROXY"
  integration_uri  = aws_lambda_function.api.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "submit" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "POST /submit"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    logging_level            = "INFO"
    detailed_metrics_enabled = true
    throttling_burst_limit    = 100
    throttling_rate_limit     = 50
  }

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gw.arn
    format = jsonencode({
      requestId          = "$context.requestId",
      routeKey           = "$context.routeKey",
      status             = "$context.status",
      error              = "$context.error.messageString",
      integrationStatus  = "$context.integrationStatus",
      integrationLatency = "$context.integrationLatency",
      requestTime        = "$context.requestTime"          # supported: human‐readable
      # or use requestTimeEpoch = "$context.requestTimeEpoch"
    })
  }
}

// Grant API Gateway permission to invoke Lambda
resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}


# 2) Add a new route
resource "aws_apigatewayv2_route" "presign" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "GET /presign"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_iam_role_policy" "lambda_s3" {
  name = "mcaa-service-lambda-s3"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:PutObjectAcl",
          "s3:GetObject"
        ]
        Resource = "${aws_s3_bucket.models.arn}/*"
      }
    ]
  })
}

# 1) Add GET /status
resource "aws_apigatewayv2_route" "status" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "GET /status"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# infra/api.tf
resource "aws_apigatewayv2_route" "download" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "GET /download"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_cloudwatch_log_group" "api_gw" {
  name              = "/aws/http-api/mcaa-service"
  retention_in_days = 14
}

// Output the invoke URL
output "api_invoke_url" {
  description = "HTTP API endpoint"
  value       = aws_apigatewayv2_api.api.api_endpoint
}
