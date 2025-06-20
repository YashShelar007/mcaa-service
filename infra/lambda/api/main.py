import os
import json
import boto3
import urllib.parse

s3  = boto3.client("s3")
sfn = boto3.client("stepfunctions")

BUCKET   = os.environ["MODEL_BUCKET"]
SM_ARN   = os.environ["STATE_MACHINE_ARN"]

CORS = {
    "Access-Control-Allow-Origin" : "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "OPTIONS,GET,POST"
}

def handler(event, context):
    # 1) Always handle CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": CORS, "body": ""}

    # 2) Look at the raw path
    path = event.get("rawPath", "")

    if path.endswith("/presign"):
        return presign(event)
    if path.endswith("/submit"):
        return submit(event)
    if path.endswith("/status"):
        return status(event)
    if path.endswith("/download"):
        return download(event)

    # 3) Favicon or anything else
    return {"statusCode": 404, "headers": CORS, "body": "Not Found"}

def presign(event):
    params   = event.get("queryStringParameters") or {}
    user_id  = params.get("user_id")
    filename = params.get("filename")
    if not user_id or not filename:
        return _bad("user_id and filename required")

    safe_fn = urllib.parse.quote(filename, safe="")
    key     = f"users/{user_id}/baseline/{safe_fn}"

    post = s3.generate_presigned_post(Bucket=BUCKET, Key=key, ExpiresIn=300)
    return {
        "statusCode": 200,
        "headers":     CORS,
        "body":        json.dumps({"url": post["url"], "fields": post["fields"], "key": key})
    }

def submit(event):
    try:
        payload   = json.loads(event.get("body") or "{}")
        user_id   = payload["user_id"]
        model_key = payload["model_s3_key"]
        profile   = payload["profile"]
    except Exception as e:
        return _bad(f"Invalid input: {e}")

    try:
        resp = sfn.start_execution(
            stateMachineArn=SM_ARN,
            input=json.dumps({
                "user_id":       user_id,
                "model_s3_key":  model_key,
                "profile":       profile
            })
        )
    except Exception as e:
        return _error(f"Failed to start execution: {e}")

    return {
        "statusCode": 200,
        "headers":     CORS,
        "body":        json.dumps({"executionArn": resp["executionArn"]})
    }

def status(event):
    params = event.get("queryStringParameters") or {}
    arn    = params.get("executionArn")
    if not arn:
        return _bad("executionArn required")

    try:
        hist = sfn.get_execution_history(
            executionArn=arn,
            maxResults=100,
            reverseOrder=True
        )
    except Exception as e:
        return _error(f"Failed to fetch history: {e}")

    events = []
    for e in hist.get("events", []):
        t = e.get("type", "")
        if not t.endswith("StateEntered"):
            continue

        # find the one key ending in "EventDetails"
        detail_key = next(
            (k for k in e.keys() if k.endswith("EventDetails")),
            None
        )
        if not detail_key:
            # nothing to extract here
            print(f"Skipping malformed entry: type={t}, keys={list(e.keys())}")
            continue

        detail = e[detail_key] or {}
        name   = detail.get("name")
        ts     = e.get("timestamp")
        if name and ts:
            events.append({
                "state":     name,
                "timestamp": ts.isoformat()
            })
        else:
            print(f"Skipping incomplete detail: {detail_key}, detail={detail}")

    return {
        "statusCode": 200,
        "headers":     CORS,
        "body":        json.dumps(events),
    }


def download(event):
    params = event.get("queryStringParameters") or {}
    key    = params.get("modelKey")
    if not key:
        return _bad("modelKey required")

    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": key},
            ExpiresIn=300
        )
    except Exception as e:
        return _error(f"Failed to get presigned url: {e}")

    return {
        "statusCode": 200,
        "headers":     CORS,
        "body":        json.dumps({"url": url})
    }

def _bad(msg):
    return {"statusCode": 400, "headers": CORS, "body": json.dumps({"error": msg})}

def _error(msg):
    return {"statusCode": 500, "headers": CORS, "body": json.dumps({"error": msg})}
