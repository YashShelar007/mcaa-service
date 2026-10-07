# Contributing

This project is not actively developed, so changes are unlikely to be merged. If you fork it:

- Python files compile with `python -m py_compile modules/compression/*.py infra/lambda/api/main.py`.
- Check the infrastructure with `cd infra && terraform init -backend=false && terraform validate`.
- There is no test suite yet; `tests/` is empty.
- Open a PR against `main` with a short note on what changed and how you checked it.
