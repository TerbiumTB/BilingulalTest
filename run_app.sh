set -e
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "→ создаю виртуальное окружение $VENV ..."
  python3 -m venv "$VENV"
  "$PY" -m pip install -q --upgrade pip
fi

if ! "$PY" -c "import uvicorn" 2>/dev/null; then
  echo "→ ставлю зависимости ..."
  "$PY" -m pip install -q -r requirements.txt
fi

echo "→ сервер на http://127.0.0.1:8000  (Ctrl+C для остановки)"
exec "$PY" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
