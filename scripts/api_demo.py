"""Demonstrate the Module 4 scoring API end to end.

    python scripts/api_demo.py --spawn                 # start the API itself, run the demo, stop it
    python scripts/api_demo.py --url http://127.0.0.1:8000   # use a server that is already running

Steps: health, model info, the four worked complaints, one what-if (change one field), an unseen company, a rejected
request (invalid month), and a check that the response never contains a decision or a deadline. Output is printed and
saved to reports/m5/api_demo_output.json.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "app" / "assets" / "dashboard_data.json").read_text())


def show(title, r):
    print(f"\n=== {title}  [HTTP {r.status_code}]")
    body = r.json()
    if r.status_code == 200 and "relief_probability" in body:
        flag = "FLAG for review" if body["flag_for_review"] else "not flagged"
        print(f"relief {body['relief_probability']:.1%} (line {body['review_threshold']:.1%}) -> {flag}; late response {body['untimely_probability']:.1%}")
        for x in body["reasons"]:
            print(f"   - {x['feature']}: {x['value']} ({x['effect']:+.2f})")
        if body["unseen_values"]:
            print("   unseen:", body["unseen_values"])
    else:
        print(json.dumps(body, indent=2)[:600])
    return body


def run(base):
    out, c = [], httpx.Client(base_url=base, timeout=60)
    out.append(("health", show("GET /health", c.get("/health"))))
    out.append(("model-info", show("GET /model-info", c.get("/model-info"))))
    for e in DATA["examples"]:
        out.append((f"example {e['id']}", show(f"POST /predict: example {e['id']} ({e['title']})", c.post("/predict", json=e["inputs"]))))
    base_case = dict(DATA["examples"][0]["inputs"])
    what_if = {**base_case, "sub_issue": "Deposits and withdrawals"}
    a = show("What-if, original (example A)", c.post("/predict", json=base_case))
    b = show("What-if, sub-issue changed to 'Deposits and withdrawals'", c.post("/predict", json=what_if))
    print(f"\nEffect of the change: {a['relief_probability']:.1%} -> {b['relief_probability']:.1%}; flag {a['flag_for_review']} -> {b['flag_for_review']}")
    out += [("what-if original", a), ("what-if changed", b)]
    out.append(("unseen company", show("Unseen company", c.post("/predict", json={**base_case, "company": "A BRAND NEW FINTECH LTD"}))))
    bad = c.post("/predict", json={**base_case, "month": 13})
    out.append(("rejected request", show("Invalid month (expect 422)", bad)))
    assert bad.status_code == 422
    forbidden = {"decision", "deadline", "approve", "deny", "verdict"}
    for name, body in out:
        assert not (forbidden & {k.lower() for k in body}), f"{name}: response contains a decision field"
    print("\nChecked: no response contains a decision or a deadline field (Charter F4).")
    dest = ROOT / "reports" / "m5"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "api_demo_output.json").write_text(json.dumps(dict(out), indent=2))
    print(f"Saved {dest / 'api_demo_output.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--spawn", action="store_true", help="start uvicorn on a free port for the duration of the demo")
    a = ap.parse_args()
    proc = None
    base = a.url
    if a.spawn:
        port = 8077
        base = f"http://127.0.0.1:{port}"
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(port), "--log-level", "warning"], cwd=ROOT)
        for _ in range(60):
            try:
                if httpx.get(base + "/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(1)
    try:
        run(base)
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    main()
