# infra/lambda/build.sh
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/api"
rm -f ../api.zip
zip -r ../api.zip . -x '*.pyc'
