"""Exercise the real HTTP endpoint with the synthetic request shipped in Git."""
import argparse
import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    for attempt in range(30):
        try:
            with urlopen(base + "/health", timeout=2) as response:
                health = json.load(response)
            assert health["status"] == "ok"
            break
        except (URLError, TimeoutError):
            if attempt == 29:
                raise
            time.sleep(1)
    payload = (Path(__file__).resolve().parents[1] / "examples/request.json").read_bytes()
    request = Request(base + "/predict", data=payload, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        result = json.load(response)
    assert 0 <= result["failure_risk"] <= 1
    assert result["horizon_hours"] == 168 and result["threshold"] == .5
    assert result["alert"] == (result["failure_risk"] >= .5)
    assert result["model_version"] == health["model_version"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
