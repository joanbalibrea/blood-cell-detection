#!/bin/zsh
# Works from any current directory; uses this project's existing environment.
set -eu
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  echo "Python environment missing. Follow the CellScope setup in README.md."
  exit 1
fi
echo 'CellScope is starting. Open http://127.0.0.1:8000'
echo 'Keep this terminal open. Press Control+C to stop.'
exec .venv/bin/python -m uvicorn cellscope_app.main:app --host 127.0.0.1 --port 8000
