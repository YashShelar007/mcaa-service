import os
import json
import boto3

# Initialize the Step Functions client
sfn = boto3.client("stepfunctions")
# ARN of your state machine, injected via Terraform env var
STATE_MACHINE_ARN = os.environ["STATE_MACHINE_ARN"]

def handler(event, context):
    """
    Expects a JSON POST body:
      {
        "user_id":    "<string>",
        "model_s3_key":"<s3-key>",
        "profile":    "<high_accuracy|balanced|max_compression>"
      }
    Kicks off a Step Functions execution and returns its ARN.
    """
    # 1. Parse and validate input
    try:
        payload = json.loads(event.get("body") or "{}")
        user_id    = payload["user_id"]
        model_key  = payload["model_s3_key"]
        profile    = payload["profile"]
    except (json.JSONDecodeError, KeyError) as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"Invalid input: {str(e)}"})
        }

    # 2. Start the state machine
    try:
        resp = sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            input=json.dumps({
                "user_id":       user_id,
                "model_s3_key":  model_key,
                "profile":       profile
            })
        )
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to start execution: {str(e)}"})
        }

    # 3. Return the execution ARN
    return {
        "statusCode": 200,
        "body": json.dumps({
            "executionArn": resp["executionArn"]
        })
    }
