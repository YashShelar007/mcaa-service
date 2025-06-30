resource "aws_sfn_state_machine" "pipeline" {
  name     = "mcaa-service-pipeline"
  role_arn = aws_iam_role.stepfunctions_exec.arn

  definition = jsonencode({
    Comment = "MCaaS compression pipeline (prune → optional KD → quantize → eval, with custom‐constraints support)"
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
        ResultPath = "$.baseline_raw_output"
        Next       = "ProcessBaseline"
      },

      #################################################
      # 2) Compute the profile/baseline S3 path
      ProcessBaseline = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"      = "$.user_id",
          "profile.$"      = "$.profile",
          "baseline_key.$" = "States.Format('users/{}/{}/baseline/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.model_s3_key,'/'), States.MathAdd(States.ArrayLength(States.StringSplit($.model_s3_key,'/')),-1)))"
        }
        ResultPath = "$.baseline_info"
        Next       = "Prune"
      },

      #################################################
      # 3) One‐shot prune → writes users/.../pruned/…
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
        ResultPath = "$.prune_raw_output"
        Next       = "ProcessPrune"
      },

      #################################################
      # 4) Compute the pruned key
      ProcessPrune = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"     = "$.user_id",
          "profile.$"     = "$.profile",
          "pruned_key.$"  = "States.Format('users/{}/{}/pruned/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.baseline_info.baseline_key,'/'), States.MathAdd(States.ArrayLength(States.StringSplit($.baseline_info.baseline_key,'/')),-1)))"
        }
        ResultPath = "$.prune_info"
        Next       = "BranchOnConstraints"
      },

      #################################################
      # 5) CUSTOM‐WORKFLOW BRANCHING (new)
      BranchOnConstraints = {
        Type    = "Choice"
        Choices = [
          # Quantize only if only bitwidth is specified
          {
            And = [
              { Variable = "$.bitwidth", IsPresent = true },
              { Variable = "$.acc_tol",  IsPresent = false },
              { Variable = "$.max_size", IsPresent = false }
            ]
            Next = "QuantizePruned"
          },
          # Prune‐only loop if only accuracy‐tolerance or max‐size specified
          {
            And = [
              { Variable = "$.bitwidth", IsPresent = false },
              { Or = [
                  { Variable = "$.acc_tol",  IsPresent = true },
                  { Variable = "$.max_size", IsPresent = true }
                ]
              }
            ]
            Next = "PruneLoop"
          },
          # Prune→Quant pipeline if any mix of constraints
          {
            And = [
              { Variable = "$.bitwidth", IsPresent = true },
              { Or = [
                  { Variable = "$.acc_tol",  IsPresent = true },
                  { Variable = "$.max_size", IsPresent = true }
                ]
              }
            ]
            Next = "PruneThenQuant"
          }
        ]
        Default = "BranchOnProfile"
      },

      #################################################
      # 6) ORIGINAL PROFILE BRANCHING (fallback)
      BranchOnProfile = {
        Type     = "Choice"
        Choices  = [
          {
            Variable     = "$.profile",
            StringEquals = "balanced",
            Next         = "Distill"
          }
        ]
        Default = "QuantizePruned"
      },

      #################################################
      # 7a) KD step (balanced profile)
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
                  { Name = "MODEL_BUCKET",    Value = aws_s3_bucket.models.bucket },
                  { Name = "BASELINE_S3_KEY", "Value.$" = "$.baseline_info.baseline_key" },
                  { Name = "PRUNED_S3_KEY",    "Value.$" = "$.prune_info.pruned_key" },
                  { Name = "USER_ID",          "Value.$" = "$.user_id" },
                  { Name = "PROFILE",          "Value.$" = "$.profile" }
                ]
              }
            ]
          }
        }
        ResultPath = "$.distill_raw_output"
        Next       = "ProcessDistill"
      },

      #################################################
      # 7b) Compute distilled key
      ProcessDistill = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"        = "$.user_id",
          "profile.$"        = "$.profile",
          "distilled_key.$"  = "States.Format('users/{}/{}/distilled/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.prune_info.pruned_key,'/'), States.MathAdd(States.ArrayLength(States.StringSplit($.prune_info.pruned_key,'/')),-1)))"
        }
        ResultPath = "$.distill_info"
        Next       = "QuantizeDistilled"
      },

      #################################################
      # 8a) Quantize pruned
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
                  { Name = "MODEL_BUCKET",    Value = aws_s3_bucket.models.bucket },
                  { Name = "MODEL_S3_KEY",     "Value.$" = "$.prune_info.pruned_key" },
                  { Name = "USER_ID",          "Value.$" = "$.user_id" },
                  { Name = "PROFILE",          "Value.$" = "$.profile" }
                ]
              }
            ]
          }
        }
        ResultPath = "$.quantize_raw_output"
        Next       = "ProcessQuantizePruned"
      },

      #################################################
      # 8b) Compute quantized key (pruned→quant)
      ProcessQuantizePruned = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"        = "$.user_id",
          "profile.$"        = "$.profile",
          "quantized_key.$"  = "States.Format('users/{}/{}/quantized/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.prune_info.pruned_key,'/'), States.MathAdd(States.ArrayLength(States.StringSplit($.prune_info.pruned_key,'/')),-1)))"
        }
        ResultPath = "$.quantize_info"
        Next       = "Evaluate"
      },

      #################################################
      # 8c) Quantize distilled
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
                  { Name = "MODEL_BUCKET",    Value = aws_s3_bucket.models.bucket },
                  { Name = "MODEL_S3_KEY",     "Value.$" = "$.distill_info.distilled_key" },
                  { Name = "USER_ID",          "Value.$" = "$.user_id" },
                  { Name = "PROFILE",          "Value.$" = "$.profile" }
                ]
              }
            ]
          }
        }
        ResultPath = "$.quantize_raw_output"
        Next       = "ProcessQuantizeDistilled"
      },

      #################################################
      # 9) Compute quantized key (distilled→quant)
      ProcessQuantizeDistilled = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"        = "$.user_id",
          "profile.$"        = "$.profile",
          "quantized_key.$"  = "States.Format('users/{}/{}/quantized/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.distill_info.distilled_key,'/'), States.MathAdd(States.ArrayLength(States.StringSplit($.distill_info.distilled_key,'/')),-1)))"
        }
        ResultPath = "$.quantize_info"
        Next       = "Evaluate"
      },

      #################################################
      # 10) Prune‐only loop (custom workflow)
      PruneLoop = {
        Type       = "Task"
        Resource   = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.cluster.arn
          LaunchType     = "FARGATE"
          TaskDefinition = aws_ecs_task_definition.worker["prune_loop"].arn
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
                Name = "prune_loop"
                Environment = [
                  { Name = "MODEL_BUCKET", Value = aws_s3_bucket.models.bucket },
                  { Name = "PRUNED_S3_KEY", "Value.$" = "$.prune_info.pruned_key" },
                  { Name = "USER_ID",       "Value.$" = "$.user_id" },
                  { Name = "PROFILE",       "Value.$" = "$.profile" },
                  { Name = "ACC_TOL",       "Value.$" = "$.acc_tol" },
                  { Name = "MAX_SIZE",      "Value.$" = "$.max_size" }
                ]
              }
            ]
          }
        }
        ResultPath = "$.prune_info"
        Next       = "Evaluate"
      },

      #################################################
      # 11) Prune→Quant combined pipeline (custom)
      PruneThenQuant = {
        Type       = "Task"
        Resource   = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.cluster.arn
          LaunchType     = "FARGATE"
          TaskDefinition = aws_ecs_task_definition.worker["prune_then_quant"].arn
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
                Name = "prune_then_quant"
                Environment = [
                  { Name = "MODEL_BUCKET", Value = aws_s3_bucket.models.bucket },
                  { Name = "PRUNED_S3_KEY", "Value.$" = "$.prune_info.pruned_key" },
                  { Name = "USER_ID",       "Value.$" = "$.user_id" },
                  { Name = "PROFILE",       "Value.$" = "$.profile" },
                  { Name = "ACC_TOL",       "Value.$" = "$.acc_tol" },
                  { Name = "MAX_SIZE",      "Value.$" = "$.max_size" },
                  { Name = "BITWIDTH",      "Value.$" = "$.bitwidth" }
                ]
              }
            ]
          }
        }
        ResultPath = "$.quantize_info"
        Next       = "Evaluate"
      },

      #################################################
      # 12) Final evaluation on the INT8 model
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
                Name    = "evaluator"
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
