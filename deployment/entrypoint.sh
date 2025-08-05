#!/bin/sh
set -e

echo "[entrypoint] restoring data from Filebase…"
# Run data restoration in background to avoid blocking startup
python /app/deployment/fetch_filebase.py &
FETCH_PID=$!

# Wait a bit for critical data, but don't block indefinitely
sleep 10

# Check if fetch is still running
if kill -0 $FETCH_PID 2>/dev/null; then
    echo "[entrypoint] Data fetch still running in background (PID: $FETCH_PID)"
else
    echo "[entrypoint] Data fetch completed"
fi

echo "[entrypoint] launching services…"
exec supervisord -c /etc/supervisord.conf