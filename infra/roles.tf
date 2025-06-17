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
    resources = ["*"] // later tighten to specific ARNs
  }
  // you can add S3/DynamoDB actions here too if SFN writes directly
}
