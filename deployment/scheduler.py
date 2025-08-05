import os
import time
import subprocess
import sys
import threading
import asyncio
from dotenv import load_dotenv
import httpx
import os

# -----------------------------------------------------------------------------
# LOCATE DATA-PIPELINE SCRIPT
# -----------------------------------------------------------------------------
if os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "main.py"))):
    PIPELINE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "main.py"))
    PIPELINE_DIR = os.path.dirname(PIPELINE_PATH)
else:
    raise FileNotFoundError("src/main.py not found!")

# -----------------------------------------------------------------------------
# CONFIGURATION (via ENV)
# -----------------------------------------------------------------------------
# URL to ping every N seconds (default 300s = 5min)
load_dotenv()
TRIGGER_HEALTH_URL = os.getenv(
    "TRIGGER_HEALTH_URL",
    "https://advisor-trigger-ki3t.onrender.com/health, https://advisorai-data-1ew2.onrender.com/health"
)
PING_INTERVAL = int(os.getenv("TRIGGER_PING_INTERVAL", "300"))
# Pipeline interval is fixed at 1800s = 30min
PIPELINE_INTERVAL = int(1800*1.5)

# -----------------------------------------------------------------------------
# ASYNC PINGER WITH EXPONENTIAL BACKOFF
# -----------------------------------------------------------------------------
async def ping_remote():
    """
    Continuously GET each URL in TRIGGER_HEALTH_URL (comma-separated) every PING_INTERVAL seconds,
    backing off on failure (up to 2.5 minutes).
    """
    urls = [u.strip() for u in TRIGGER_HEALTH_URL.split(",") if u.strip()]
    backoff = min(PING_INTERVAL, 5)
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            all_success = True
            for url in urls:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    print(f"[Pinger] {url} -> {resp.status_code}")
                except Exception as e:
                    print(f"[Pinger] error pinging {url}: {e}")
                    all_success = False
            if all_success:
                backoff = PING_INTERVAL
                await asyncio.sleep(PING_INTERVAL)
            else:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 150)

def start_async_ping():
    """
    Spin up a dedicated asyncio loop in a daemon thread
    to run ping_remote() forever.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(ping_remote())
    loop.run_forever()

# launch the ping loop in the background
threading.Thread(target=start_async_ping, daemon=True).start()
print("[Scheduler] Started background ping thread")

# -----------------------------------------------------------------------------
# MAIN PIPELINE LOOP (runs every 30 minutes)
# -----------------------------------------------------------------------------
import traceback

while True:
    from datetime import datetime
    last_run = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[Scheduler] Running pipeline... Last run: {last_run}")
    # Write last_run to file for API access
    try:
        with open(os.path.join(os.path.dirname(__file__), 'last_run.txt'), 'w') as f:
            f.write(last_run)
    except Exception as e:
        print(f"[Scheduler] Failed to write last_run.txt: {e}")
    try:
        # Set working directory to project root (parent of deployment)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        print(f"[Scheduler] Project root: {project_root}")
        print(f"[Scheduler] Pipeline path: {PIPELINE_PATH}")

        result = subprocess.run(
            [sys.executable, PIPELINE_PATH],
            cwd=project_root,
            capture_output=True,
            text=True,
            env=os.environ.copy()
        )
        print(f"[Scheduler] Pipeline finished with code {result.returncode}")

        if result.stdout:
            print("[Scheduler] STDOUT:\n", result.stdout)
        if result.stderr:
            print("[Scheduler] STDERR:\n", result.stderr)

        # Raise an exception if the return code is non-zero
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"[Scheduler] Pipeline execution failed with return code {e.returncode}")
        print(f"[Scheduler] STDOUT:\n{e.stdout}")
        print(f"[Scheduler] STDERR:\n{e.stderr}")
    except Exception as e:
        print(f"[Scheduler] Exception running pipeline: {e}")
        print(traceback.format_exc())

    print(f"[Scheduler] Sleeping for {PIPELINE_INTERVAL // 60} minutes...")
    time.sleep(PIPELINE_INTERVAL)
