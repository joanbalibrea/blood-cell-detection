#!/bin/zsh
# Keep this Terminal window open while viewing the dashboard in Safari.
cd "$(dirname "$0")" || exit 1
CLASSIFIER_PY="../.venv/bin/python"
if [[ ! -x "$CLASSIFIER_PY" ]]; then
  CLASSIFIER_PY=".venv/bin/python"
fi
if [[ ! -x "$CLASSIFIER_PY" ]]; then
  echo "No se encuentra el entorno Python. Consulta README.md."
  read -r "?Pulsa Intro para cerrar."
  exit 1
fi
echo "Resultados: http://127.0.0.1:8766/dashboard/teacher_baseline.html"
echo "Entrenamiento: http://127.0.0.1:8766/dashboard/"
echo "Mantén esta ventana abierta. Ctrl+C detiene solo el servidor local."
exec "$CLASSIFIER_PY" manage.py serve --port 8766
