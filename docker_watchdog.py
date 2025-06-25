import subprocess
import time
import logging

CONTAINER_NAME = "omniparser"
CHECK_INTERVAL = 60  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

def get_container_health():
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", CONTAINER_NAME],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "not_found"

def get_container_status():
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", CONTAINER_NAME],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "not_found"

def restart_container():
    logging.warning(f"Restarting container: {CONTAINER_NAME}")
    subprocess.run(["docker", "restart", CONTAINER_NAME])

def main():
    while True:
        health = get_container_health()
        status = get_container_status()
        logging.info(f"Container health: {health}, status: {status}")

        if health in ("unhealthy", "not_found") or status != "running":
            restart_container()
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main() 