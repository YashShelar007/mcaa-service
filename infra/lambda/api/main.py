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
    # 1) CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": CORS, "body": ""}

    path = event.get("rawPath", "")
    if path.endswith("/presign"):
        return presign(event)
    if path.endswith("/submit"):
        return submit(event)
    if path.endswith("/status"):
        return status(event)
    if path.endswith("/download"):
        return download(event)
    return {"statusCode": 404, "headers": CORS, "body": "Not Found"}


def presign(event):
    params   = event.get("queryStringParameters") or {}
    user_id  = params.get("user_id")
    filename = params.get("filename")
    if not user_id or not filename:
        return _bad("user_id and filename required")

    safe_fn = urllib.parse.quote(filename, safe="")
    key     = f"users/{user_id}/baseline/{safe_fn}"
    post    = s3.generate_presigned_post(Bucket=BUCKET, Key=key, ExpiresIn=300)

    return {
        "statusCode": 200,
        "headers":     CORS,
        "body":        json.dumps({"url": post["url"], "fields": post["fields"], "key": key})
    }


def submit(event):
    # parse JSON
    try:
        payload   = json.loads(event.get("body") or "{}")
        user_id   = payload["user_id"]
        model_key = payload["model_s3_key"]
        # ← change here: default to "custom" when profile is missing/null
        profile   = payload.get("profile") or "custom"
    except Exception as e:
        return _bad(f"Invalid input: {e}")

    # helper to parse numeric fields
    def parse_field(name, cast, minval=None, maxval=None):
        v = payload.get(name)
        if v is None or v == "":
            return None
        try:
            v = cast(v)
        except:
            raise ValueError(f"{name} must be a {cast.__name__}")
        if minval is not None and v < minval:
            raise ValueError(f"{name} must be ≥ {minval}")
        if maxval is not None and v > maxval:
            raise ValueError(f"{name} must be ≤ {maxval}")
        return v

    try:
        acc_tol   = parse_field("acc_tol", float, 0.0, 100.0)
        size_lim  = parse_field("size_limit", float, 0.0, None)
        size_unit = payload.get("size_unit")  # “MB” or “GB”
        bitwidth  = parse_field("bitwidth", int, 1, None)

        # validate size_unit if provided
        if size_lim is not None and size_unit not in ("MB", "GB"):
            raise ValueError("size_unit must be 'MB' or 'GB'")
    except ValueError as ve:
        return _bad(str(ve))

    # build Step Functions input
    exec_input = {
        "user_id":      user_id,
        "model_s3_key": model_key,
        "profile":      profile,    # now always a non-null string
    }
    if acc_tol  is not None: exec_input["acc_tol"]  = acc_tol
    if bitwidth is not None: exec_input["bitwidth"] = bitwidth

    # normalize and include size_limit as MB
    if size_lim is not None:
        mb = size_lim * (1024.0 if size_unit == "GB" else 1.0)
        exec_input["size_limit_mb"] = mb

    try:
        resp = sfn.start_execution(
            stateMachineArn=SM_ARN,
            input=json.dumps(exec_input),
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
            reverseOrder=True,
            includeExecutionData=True
        )
    except Exception as e:
        return _error(f"Failed to fetch history: {e}")

    events = []

    # 1) Inject overall failure if the execution failed
    for e in hist.get("events", []):
        if e.get("type") == "ExecutionFailed":
            events.insert(0, {
                "state":     "ExecutionFailed",
                "timestamp": e["timestamp"].isoformat()
            })
            break

    # 2) Capture each task’s exit output for metrics
    for ev in hist.get("events", []):
        if ev.get("type") != "TaskStateExited":
            continue

        details = ev.get("stateExitedEventDetails", {})
        name    = details.get("name")
        ts      = ev.get("timestamp")
        out_str = details.get("output", "{}")

        try:
            out = json.loads(out_str)
        except:
            out = {}

        entry = {
            "state":     name,
            "timestamp": ts.isoformat()
        }
        if "accuracy"   in out:
            entry["accuracy"]   = out["accuracy"]
        if "size_bytes" in out:
            entry["size_bytes"] = out["size_bytes"]

        events.append(entry)

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
