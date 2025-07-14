// infra/ecs.tf

locals {
  workers = ["baseline", "prune_structured", "quantize", "distill_kd", "evaluate", "prune_search", "prune_and_quantize"]
}

// ─────────── ECR Repositories ───────────
resource "aws_ecr_repository" "worker" {
  for_each = toset(local.workers)
  name     = "mcaa-service-${each.key}"
}

// ─────────── ECS Cluster ───────────
resource "aws_ecs_cluster" "cluster" {
  name = "mcaa-service-cluster"
}

// ─────────── Task Definitions ───────────
resource "aws_ecs_task_definition" "worker" {
  for_each = toset(local.workers)

  family                   = "mcaa-service-${each.key}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "4096"
  memory                   = "8192"
  execution_role_arn       = aws_iam_role.ecs_task.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = each.key
      image     = "${aws_ecr_repository.worker[each.key].repository_url}:latest"
      essential = true
      cpu       = 4096
      memory    = 8192

      environment = [
        { name  = "PROFILE"       , value = "$.profile"                       },
        { name  = "MODEL_BUCKET"  , value = aws_s3_bucket.models.bucket        },
        { name  = "METADATA_TABLE", value = aws_dynamodb_table.metadata.name    }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/mcaa-service/${each.key}"
          "awslogs-region"        = "us-west-2"
          "awslogs-stream-prefix" = each.key
        }
      }
    }
  ])
}
