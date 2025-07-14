// ECS Task role
resource "aws_iam_role" "ecs_task" {
  name               = "mcaa-service-ecs-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
}

data "aws_iam_policy_document" "ecs_task_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "ecs_task_policy" {
  name   = "mcaa-service-ecs-task-policy"
  role   = aws_iam_role.ecs_task.id
  policy = data.aws_iam_policy_document.ecs_task_policy.json
}

data "aws_iam_policy_document" "ecs_task_policy" {
  statement {
    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:ListBucket"
    ]
    resources = [
      aws_s3_bucket.models.arn,
      "${aws_s3_bucket.models.arn}/*"
    ]
  }

  statement {
    actions = [
      "dynamodb:PutItem",
      "dynamodb:GetItem",
      "dynamodb:UpdateItem",
      "dynamodb:Query"
    ]
    resources = [
      aws_dynamodb_table.metadata.arn
    ]
  }

  statement {
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer"
    ]
    resources = ["*"]
  }

  # Allow containers to write logs
  statement {
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ]
    resources = [
      "arn:aws:logs:us-west-2:839918633714:log-group:/mcaa-service/*",
      "arn:aws:logs:us-west-2:839918633714:log-group:/mcaa-service/*:*"
    ]
  }
}

// Step Functions execution role
resource "aws_iam_role" "stepfunctions_exec" {
  name               = "mcaa-service-sfn-exec-role"
  assume_role_policy = data.aws_iam_policy_document.sfn_assume.json
}

data "aws_iam_policy_document" "sfn_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "stepfunctions_policy" {
  name   = "mcaa-service-sfn-policy"
  role   = aws_iam_role.stepfunctions_exec.id
  policy = data.aws_iam_policy_document.sfn_policy.json
}

data "aws_iam_policy_document" "sfn_policy" {
  statement {
    actions = [
      "ecs:RunTask",
      "iam:PassRole"
    ]
    resources = ["*"]
  }

  statement {
    actions = [
      "events:PutRule",
      "events:DescribeRule",
      "events:ListTargetsByRule",
      "events:PutTargets",
      "events:RemoveTargets",
      "events:DeleteRule"
    ]
    resources = ["*"]
  }

  statement {
    actions = [
      "logs:CreateLogDelivery",
      "logs:GetLogDelivery",
      "logs:UpdateLogDelivery",
      "logs:DeleteLogDelivery",
      "logs:ListLogDeliveries",
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ]
    resources = ["*"]
  }
}

# ----------------------------------------
# 1) IAM Role for the API‐Gateway → Lambda
# ----------------------------------------
resource "aws_iam_role" "api_lambda" {
  name = "mcaa-service-api-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# ----------------------------------------
# 2) Attach a policy that lets Lambda:
#    • write CloudWatch logs
#    • generate presigned S3 URLs (put/get/list)
#    • start & inspect Step Functions
#    • query DynamoDB metadata table
# ----------------------------------------
resource "aws_iam_role_policy" "api_lambda_policy" {
  name = "mcaa-service-api-lambda-policy"
  role = aws_iam_role.api_lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # CloudWatch Logs
      {
        Effect   = "Allow"
        Action   = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      # S3: presign upload & download
      {
        Effect   = "Allow"
        Action   = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.models.arn,
          "${aws_s3_bucket.models.arn}/*"
        ]
      },
      # Step Functions: start executions & read history
      {
        Effect = "Allow"
        Action = [
          "states:StartExecution",
          "states:GetExecutionHistory"
        ]
        Resource = aws_sfn_state_machine.pipeline.arn
      },
      # DynamoDB: query metadata table
      {
        Effect   = "Allow"
        Action   = [
          "dynamodb:Query",
          "dynamodb:GetItem"
        ]
        Resource = aws_dynamodb_table.metadata.arn
      }
    ]
  })
}
