// infra/step_functions.tf

resource "aws_sfn_state_machine" "pipeline" {
  name     = "mcaa-service-pipeline"
  role_arn = aws_iam_role.stepfunctions_exec.arn

  definition = jsonencode({
    Comment = "MCaaS compression pipeline (prune → optional KD → quantize → eval)"
    StartAt = "Baseline"
    States  = {

      #################################################
      # 1) Download & copy into profile/baseline/…
      Baseline = {
        Type       = "Task"
        Resource   = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.cluster.arn
          LaunchType     = "FARGATE"
          TaskDefinition = aws_ecs_task_definition.worker["baseline"].arn
          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets        = data.aws_subnet_ids.default.ids
              SecurityGroups = [data.aws_security_group.default.id]
              AssignPublicIp = "ENABLED"
            }
          }
          Overrides = {
            ContainerOverrides = [
              {
                Name = "baseline"
                Environment = [
                  { Name = "MODEL_BUCKET", Value = aws_s3_bucket.models.bucket },
                  { Name = "MODEL_S3_KEY",  "Value.$" = "$.model_s3_key" },
                  { Name = "USER_ID",       "Value.$" = "$.user_id" },
                  { Name = "PROFILE",       "Value.$" = "$.profile" }
                ]
              }
            ]
          }
        }
        ResultPath = "$.baseline_raw_output" # Store the raw output of the task
        Next = "ProcessBaseline"
      },

      # Compute the new S3 key for baseline under the profile prefix
      ProcessBaseline = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"       = "$.user_id",
          "profile.$"       = "$.profile",
          "baseline_key.$": "States.Format('users/{}/{}/baseline/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.model_s3_key, '/'), States.MathAdd(States.ArrayLength(States.StringSplit($.model_s3_key, '/')), -1)))"
        }
        ResultPath = "$.baseline_info"
        Next       = "Prune"
      },

      #################################################
      # 2) Prune → writes users/.../pruned/…
      Prune = {
        Type       = "Task"
        Resource   = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.cluster.arn
          LaunchType     = "FARGATE"
          TaskDefinition = aws_ecs_task_definition.worker["prune"].arn
          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets        = data.aws_subnet_ids.default.ids
              SecurityGroups = [data.aws_security_group.default.id]
              AssignPublicIp = "ENABLED"
            }
          }
          Overrides = {
            ContainerOverrides = [
              {
                Name = "prune"
                Environment = [
                  { Name = "MODEL_BUCKET", Value = aws_s3_bucket.models.bucket },
                  { Name = "MODEL_S3_KEY",  "Value.$" = "$.baseline_info.baseline_key" },
                  { Name = "USER_ID",       "Value.$" = "$.user_id" },
                  { Name = "PROFILE",       "Value.$" = "$.profile" }
                ]
              }
            ]
          }
        }
        ResultPath = "$.prune_raw_output" # Store the raw output of the task
        Next = "ProcessPrune"
      },

      # Compute the pruned key
      ProcessPrune = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"      = "$.user_id",
          "profile.$"      = "$.profile",
          "pruned_key.$": "States.Format('users/{}/{}/pruned/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.baseline_info.baseline_key, '/'), States.MathAdd(States.ArrayLength(States.StringSplit($.baseline_info.baseline_key, '/')), -1)))"
        }
        ResultPath = "$.prune_info"
        Next       = "BranchOnProfile"
      },

      #################################################
      # 3) Choose KD vs straight-to-quantize
      BranchOnProfile = {
        Type     = "Choice"
        Choices  = [
          {
            Variable    = "$.profile",
            StringEquals = "balanced",
            Next         = "Distill"
          }
        ]
        Default = "QuantizePruned"
      },

      #################################################
      # 4a) Balanced → Distill → writes …/distilled/…
      Distill = {
        Type       = "Task"
        Resource   = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.cluster.arn
          LaunchType     = "FARGATE"
          TaskDefinition = aws_ecs_task_definition.worker["distill"].arn
          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets        = data.aws_subnet_ids.default.ids
              SecurityGroups = [data.aws_security_group.default.id]
              AssignPublicIp = "ENABLED"
            }
          }
          Overrides = {
            ContainerOverrides = [
              {
                Name = "distill"
                Environment = [
                  { Name = "MODEL_BUCKET", Value = aws_s3_bucket.models.bucket },
                  { Name = "BASELINE_S3_KEY" , "Value.$" = "$.baseline_info.baseline_key" },
                  { Name = "PRUNED_S3_KEY",  "Value.$" = "$.prune_info.pruned_key" },
                  { Name = "USER_ID",       "Value.$" = "$.user_id" },
                  { Name = "PROFILE",       "Value.$" = "$.profile" }
                ]
              }
            ]
          }
        }
        ResultPath = "$.distill_raw_output" # Store the raw output of the task
        Next = "ProcessDistill"
      },

      # Compute the distilled key
      ProcessDistill = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"       = "$.user_id",
          "profile.$"       = "$.profile",
          "distilled_key.$": "States.Format('users/{}/{}/distilled/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.prune_info.pruned_key, '/'), States.MathAdd(States.ArrayLength(States.StringSplit($.prune_info.pruned_key, '/')), -1)))"
        }
        ResultPath = "$.distill_info"
        Next       = "QuantizeDistilled"
      },

      #################################################
      # 4b) Quantize branches
      QuantizePruned = {
        Type       = "Task"
        Resource   = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.cluster.arn
          LaunchType     = "FARGATE"
          TaskDefinition = aws_ecs_task_definition.worker["quantize"].arn
          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets        = data.aws_subnet_ids.default.ids
              SecurityGroups = [data.aws_security_group.default.id]
              AssignPublicIp = "ENABLED"
            }
          }
          Overrides = {
            ContainerOverrides = [
              {
                Name = "quantize"
                Environment = [
                  { Name = "MODEL_BUCKET", Value = aws_s3_bucket.models.bucket },
                  { Name = "MODEL_S3_KEY",  "Value.$" = "$.prune_info.pruned_key" },
                  { Name = "USER_ID",       "Value.$" = "$.user_id" },
                  { Name = "PROFILE",       "Value.$" = "$.profile" }
                ]
              }
            ]
          }
        }
        ResultPath = "$.quantize_raw_output" # Store the raw output of the task
        Next = "ProcessQuantizePruned"
      },

      # Compute the final quantized key
      ProcessQuantizePruned = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"        = "$.user_id",
          "profile.$"        = "$.profile",
          "quantized_key.$": "States.Format('users/{}/{}/quantized/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.prune_info.pruned_key, '/'), States.MathAdd(States.ArrayLength(States.StringSplit($.prune_info.pruned_key, '/')), -1)))"
        }
        ResultPath = "$.quantize_info"
        Next       = "Evaluate"
      },

      QuantizeDistilled = {
        Type       = "Task"
        Resource   = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.cluster.arn
          LaunchType     = "FARGATE"
          TaskDefinition = aws_ecs_task_definition.worker["quantize"].arn
          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets        = data.aws_subnet_ids.default.ids
              SecurityGroups = [data.aws_security_group.default.id]
              AssignPublicIp = "ENABLED"
            }
          }
          Overrides = {
            ContainerOverrides = [
              {
                Name = "quantize"
                Environment = [
                  { Name = "MODEL_BUCKET", Value = aws_s3_bucket.models.bucket },
                  { Name = "MODEL_S3_KEY",  "Value.$" = "$.distill_info.distilled_key" },
                  { Name = "USER_ID",       "Value.$" = "$.user_id" },
                  { Name = "PROFILE",       "Value.$" = "$.profile" }
                ]
              }
            ]
          }
        }
        ResultPath = "$.quantize_raw_output" # Store the raw output of the task
        Next = "ProcessQuantizeDistilled"
      },

      # Compute the final quantized key
      ProcessQuantizeDistilled = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"        = "$.user_id",
          "profile.$"        = "$.profile",
          "quantized_key.$": "States.Format('users/{}/{}/quantized/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.distill_info.distilled_key, '/'), States.MathAdd(States.ArrayLength(States.StringSplit($.distill_info.distilled_key, '/')), -1)))"
        }
        ResultPath = "$.quantize_info"
        Next       = "Evaluate"
      },

      #################################################
      # 5) Final evaluation on the INT8 model
      Evaluate = {
        Type       = "Task"
        Resource   = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.cluster.arn
          LaunchType     = "FARGATE"
          TaskDefinition = aws_ecs_task_definition.worker["evaluator"].arn
          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets        = data.aws_subnet_ids.default.ids
              SecurityGroups = [data.aws_security_group.default.id]
              AssignPublicIp = "ENABLED"
            }
          }
          Overrides = {
            ContainerOverrides = [
              {
                Name = "evaluator"
                Command = ["python","measure_inference_time.py"]
                Environment = [
                  { Name = "MODEL_BUCKET", Value = aws_s3_bucket.models.bucket },
                  { Name = "MODEL_S3_KEY",  "Value.$" = "$.quantize_info.quantized_key" },
                  { Name = "USER_ID",       "Value.$" = "$.user_id" },
                  { Name = "PROFILE",       "Value.$" = "$.profile" }
                ]
              }
            ]
          }
        }
        End = true
      }
    }
  })
}
