resource "aws_sfn_state_machine" "pipeline" {
  name     = "mcaa-service-pipeline"
  role_arn = aws_iam_role.stepfunctions_exec.arn

  definition = jsonencode({
    Comment = "MCaaS compression pipeline (baseline → custom‐constraints → preset profiles)"
    StartAt = "Baseline"
    States = {

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
                  { Name = "MODEL_BUCKET", "Value" = aws_s3_bucket.models.bucket             },
                  { Name = "MODEL_S3_KEY",  "Value.$" = "$.model_s3_key"                      },
                  { Name = "USER_ID",       "Value.$" = "$.user_id"                           },
                  { Name = "PROFILE",       "Value.$" = "$.profile"                           }
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
          "user_id.$"        = "$.user_id",
          "profile.$"        = "$.profile",
          "baseline_key.$"   = "States.Format('users/{}/{}/baseline/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.model_s3_key,'/'),States.MathAdd(States.ArrayLength(States.StringSplit($.model_s3_key,'/')),-1)))"
        }
        ResultPath = "$.baseline_info"
        Next       = "BranchOnConstraints"
      },

      #################################################
      # 3) CUSTOM‐WORKFLOW BRANCHING (before any one‐shot prune)
      BranchOnConstraints = {
        Type    = "Choice"
        Choices = [
          # a) Quantize only: only bitwidth provided
          {
            And = [
              { Variable = "$.bitwidth", IsPresent = true },
              { Variable = "$.acc_tol",  IsPresent = false },
              { Variable = "$.size_limit_mb", IsPresent = false }
            ]
            Next = "QuantizeBaseline"
          },
          # b) Prune‐only loop: only acc_tol or size_limit provided
          {
            And = [
              { Variable = "$.bitwidth",       IsPresent = false },
              { Or = [
                  { Variable = "$.acc_tol",        IsPresent = true },
                  { Variable = "$.size_limit_mb",  IsPresent = true }
                ]
              }
            ]
            Next = "PruneSearch"
          },
          # c) Prune→Quant combined: any mix of constraints
          {
            And = [
              { Variable = "$.bitwidth",       IsPresent = true },
              { Or = [
                  { Variable = "$.acc_tol",        IsPresent = true },
                  { Variable = "$.size_limit_mb",  IsPresent = true }
                ]
              }
            ]
            Next = "PruneThenQuant"
          }
        ]
        Default = "BranchOnProfile"
      },

      #################################################
      # 4) Quantize directly from baseline
      QuantizeBaseline = {
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
                  { Name = "MODEL_BUCKET",  "Value" = aws_s3_bucket.models.bucket            },
                  { Name = "MODEL_S3_KEY",   "Value.$" = "$.baseline_info.baseline_key"     },
                  { Name = "USER_ID",        "Value.$" = "$.user_id"                        },
                  { Name = "PROFILE",        "Value.$" = "$.profile"                        },
                  { Name = "BITWIDTH",       "Value" = "8"                       }
                ]
              }
            ]
          }
        }
        ResultPath = "$.quantize_raw_output"
        Next       = "ProcessQuantizeBaseline"
      },

      ProcessQuantizeBaseline = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"       = "$.user_id",
          "profile.$"       = "$.profile",
          "quantized_key.$" = "States.Format('users/{}/{}/quantized/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.baseline_info.baseline_key,'/'),States.MathAdd(States.ArrayLength(States.StringSplit($.baseline_info.baseline_key,'/')),-1)))"
          "evaluate_key.$"  = "States.Format('users/{}/{}/quantized/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.baseline_info.baseline_key,'/'),States.MathAdd(States.ArrayLength(States.StringSplit($.baseline_info.baseline_key,'/')),-1)))"
        }
        ResultPath = "$.quantize_info"
        Next       = "Evaluate"
      },

      #################################################
      # 5) Preset‐profile branching (fallback)
      BranchOnProfile = {
        Type     = "Choice"
        Choices  = [
          {
            Variable     = "$.profile",
            StringEquals = "balanced",
            Next         = "PrunePreset"
          },
          { Variable     = "$.profile", 
            StringEquals = "high_accuracy", 
            Next         = "DistillBaseline"
          }
        ]
        Default = "QuantizeBaseline"
      },

      #################################################
      # 6b) Knowledge‐distillation (max accuracy)
      DistillBaseline = {
        Type       = "Task"
        Resource   = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.cluster.arn
          LaunchType     = "FARGATE"
          TaskDefinition = aws_ecs_task_definition.worker["distill_kd"].arn
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
                Name = "distill_kd"
                Environment = [
                  { Name = "MODEL_BUCKET",     "Value" = aws_s3_bucket.models.bucket            },
                  { Name = "BASELINE_S3_KEY",  "Value.$" = "$.baseline_info.baseline_key"       },
                  { Name = "PRUNED_S3_KEY",    "Value.$" = "$.baseline_info.baseline_key"       },
                  { Name = "USER_ID",          "Value.$" = "$.user_id"                          },
                  { Name = "PROFILE",          "Value.$" = "$.profile"                          }
                ]
              }
            ]
          }
        }
        ResultPath = "$.distill_raw_output"
        Next       = "ProcessDistillBaseline"
      },

      ProcessDistillBaseline = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"       = "$.user_id",
          "profile.$"       = "$.profile",
          "distilled_key.$" = "States.Format('users/{}/{}/distilled/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.prune_info.pruned_key,'/'),States.MathAdd(States.ArrayLength(States.StringSplit($.prune_info.pruned_key,'/')),-1)))"
          "evaluate_key.$"  = "States.Format('users/{}/{}/distilled/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.prune_info.pruned_key,'/'),States.MathAdd(States.ArrayLength(States.StringSplit($.prune_info.pruned_key,'/')),-1)))"
        }
        ResultPath = "$.distill_info"
        Next       = "Evaluate"
      },

      #################################################
      # 6a) One‐shot 50% prune for preset profiles
      PrunePreset = {
        Type       = "Task"
        Resource   = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.cluster.arn
          LaunchType     = "FARGATE"
          TaskDefinition = aws_ecs_task_definition.worker["prune_structured"].arn
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
                Name = "prune_structured"
                Environment = [
                  { Name = "MODEL_BUCKET",  "Value" = aws_s3_bucket.models.bucket           },
                  { Name = "MODEL_S3_KEY",   "Value.$" = "$.baseline_info.baseline_key"    },
                  { Name = "USER_ID",        "Value.$" = "$.user_id"                      },
                  { Name = "PROFILE",        "Value.$" = "$.profile"                      }
                ]
              }
            ]
          }
        }
        ResultPath = "$.prune_raw_output"
        Next       = "ProcessPrunePreset"
      },

      ProcessPrunePreset = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"    = "$.user_id",
          "profile.$"    = "$.profile",
          "pruned_key.$" = "States.Format('users/{}/{}/pruned/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.baseline_info.baseline_key,'/'),States.MathAdd(States.ArrayLength(States.StringSplit($.baseline_info.baseline_key,'/')),-1)))"
        }
        ResultPath = "$.prune_info"
        Next       = "Distill"
      },

      #################################################
      # 6b) Knowledge‐distillation (balanced)
      Distill = {
        Type       = "Task"
        Resource   = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.cluster.arn
          LaunchType     = "FARGATE"
          TaskDefinition = aws_ecs_task_definition.worker["distill_kd"].arn
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
                Name = "distill_kd"
                Environment = [
                  { Name = "MODEL_BUCKET",     "Value" = aws_s3_bucket.models.bucket           },
                  { Name = "BASELINE_S3_KEY",  "Value.$" = "$.baseline_info.baseline_key"       },
                  { Name = "PRUNED_S3_KEY",    "Value.$" = "$.prune_info.pruned_key"            },
                  { Name = "USER_ID",          "Value.$" = "$.user_id"                          },
                  { Name = "PROFILE",          "Value.$" = "$.profile"                          }
                ]
              }
            ]
          }
        }
        ResultPath = "$.distill_raw_output"
        Next       = "ProcessDistill"
      },

      ProcessDistill = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"       = "$.user_id",
          "profile.$"       = "$.profile",
          "distilled_key.$" = "States.Format('users/{}/{}/distilled/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.prune_info.pruned_key,'/'),States.MathAdd(States.ArrayLength(States.StringSplit($.prune_info.pruned_key,'/')),-1)))"
        }
        ResultPath = "$.distill_info"
        Next       = "QuantizeDistilled"
      },

      #################################################
      # 6c) Quantize distilled
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
                  { Name = "MODEL_BUCKET",  "Value" = aws_s3_bucket.models.bucket           },
                  { Name = "MODEL_S3_KEY",   "Value.$" = "$.distill_info.distilled_key"     },
                  { Name = "USER_ID",        "Value.$" = "$.user_id"                        },
                  { Name = "PROFILE",        "Value.$" = "$.profile"                        },
                  { Name = "BITWIDTH",       "Value" = "8"                      }
                ]
              }
            ]
          }
        }
        ResultPath = "$.quantize_raw_output"
        Next       = "ProcessQuantizeDistilled"
      },

      ProcessQuantizeDistilled = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"        = "$.user_id",
          "profile.$"        = "$.profile",
          "quantized_key.$"  = "States.Format('users/{}/{}/quantized/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.distill_info.distilled_key,'/'),States.MathAdd(States.ArrayLength(States.StringSplit($.distill_info.distilled_key,'/')),-1)))"
          "evaluate_key.$"   = "States.Format('users/{}/{}/quantized/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.distill_info.distilled_key,'/'),States.MathAdd(States.ArrayLength(States.StringSplit($.distill_info.distilled_key,'/')),-1)))"
        }
        ResultPath = "$.quantize_info"
        Next       = "Evaluate"
      },

      #################################################
      # 7) Prune‐only loop (custom workflow)
      PruneSearch = {
        Type       = "Task"
        Resource   = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.cluster.arn
          LaunchType     = "FARGATE"
          TaskDefinition = aws_ecs_task_definition.worker["prune_search"].arn
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
                Name = "prune_search"
                Environment = [
                  { Name = "MODEL_BUCKET",    "Value" = aws_s3_bucket.models.bucket       },
                  { Name = "PRUNED_S3_KEY",    "Value.$" = "$.baseline_info.baseline_key"      },
                  { Name = "USER_ID",          "Value.$" = "$.user_id"                     },
                  { Name = "PROFILE",          "Value.$" = "$.profile"                     },
                  { Name = "ACC_TOL",          "Value.$" = "$.acc_tol"                     },
                  { Name = "SIZE_LIMIT_MB",    "Value.$" = "$.size_limit_mb"               }
                ]
              }
            ]
          }
        }
        ResultPath = "$.prune_info"
        Next       = "ProcessPruneSearch"
      },

      ProcessPruneSearch = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"    = "$.user_id",
          "profile.$"    = "$.profile",
          "pruned_key.$" = "States.Format('users/{}/{}/pruned/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.baseline_info.baseline_key,'/'),States.MathAdd(States.ArrayLength(States.StringSplit($.baseline_info.baseline_key,'/')),-1)))"
          "evaluate_key.$" = "States.Format('users/{}/{}/pruned/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.baseline_info.baseline_key,'/'),States.MathAdd(States.ArrayLength(States.StringSplit($.baseline_info.baseline_key,'/')),-1)))"
        }
        ResultPath = "$.prune_info"
        Next       = "Evaluate"
      },

      #################################################
      # 8) Prune→Quant combined pipeline (custom)
      PruneThenQuant = {
        Type       = "Task"
        Resource   = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.cluster.arn
          LaunchType     = "FARGATE"
          TaskDefinition = aws_ecs_task_definition.worker["prune_and_quantize"].arn
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
                Name = "prune_and_quantize"
                Environment = [
                  { Name = "MODEL_BUCKET",    "Value" = aws_s3_bucket.models.bucket       },
                  { Name = "PRUNED_S3_KEY",    "Value.$" = "$.baseline_info.baseline_key"      },
                  { Name = "USER_ID",          "Value.$" = "$.user_id"                   },
                  { Name = "PROFILE",          "Value.$" = "$.profile"                   },
                  { Name = "ACC_TOL",          "Value.$" = "$.acc_tol"                   },
                  { Name = "SIZE_LIMIT_MB",    "Value.$" = "$.size_limit_mb"             },
                  { Name = "BITWIDTH",         "Value.$" = "$.bitwidth"                  }
                ]
              }
            ]
          }
        }
        ResultPath = "$.quantize_info"
        Next       = "ProcessPruneThenQuant"
      },

      ProcessPruneThenQuant = {
        Type       = "Pass"
        Parameters = {
          "user_id.$"        = "$.user_id",
          "profile.$"        = "$.profile",
          "quantized_key.$"  = "States.Format('users/{}/{}/quantized/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.distill_info.distilled_key,'/'),States.MathAdd(States.ArrayLength(States.StringSplit($.distill_info.distilled_key,'/')),-1)))"
          "evaluate_key.$"   = "States.Format('users/{}/{}/quantized/{}', $.user_id, $.profile, States.ArrayGetItem(States.StringSplit($.distill_info.distilled_key,'/'),States.MathAdd(States.ArrayLength(States.StringSplit($.distill_info.distilled_key,'/')),-1)))"
        }
        ResultPath = "$.quantize_info"
        Next       = "Evaluate"
      },

      #################################################
      # 9) Final evaluation on the INT8 model
      Evaluate = {
        Type       = "Task"
        Resource   = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.cluster.arn
          LaunchType     = "FARGATE"
          TaskDefinition = aws_ecs_task_definition.worker["evaluate"].arn
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
                Name    = "evaluate"
                Command = ["python","measure_inference_time.py"]
                Environment = [
                  { Name = "MODEL_BUCKET", "Value" = aws_s3_bucket.models.bucket            },
                  { Name = "MODEL_S3_KEY",  "Value.$" = "$.quantize_info.evaluate_key"  },
                  { Name = "USER_ID",       "Value.$" = "$.user_id"                     },
                  { Name = "PROFILE",       "Value.$" = "$.profile"                     }
                ]
              }
            ]
          }
        }
        End = true
      }

    } # end States
  })   # end jsonencode
}
