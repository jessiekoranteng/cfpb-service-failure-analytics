"""Create the GitHub repository, labels, milestones, issues and Projects board from tickets.csv.

Usage: python setup_github.py --repo OWNER/NAME [--dry-run] [--public]
Requires the GitHub CLI (gh), authenticated with the 'project' scope. Nothing runs without --repo.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SPRINTS = {
    "0": "Sprint 0 - Foundation",
    "1": "Sprint 1 - Data pipeline",
    "2": "Sprint 2 - Models, XAI and fairness",
    "3": "Sprint 3 - Dashboard and stakeholders",
    "4": "Sprint 4 - Integration and deployment",
}


def run(cmd, dry, capture=False):
    print("+", " ".join(cmd))
    if dry:
        return ""
    res = subprocess.run(cmd, check=True, text=True, capture_output=capture)
    return res.stdout if capture else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()
    owner, name = a.repo.split("/")
    d = a.dry_run
    with open(HERE / "tickets.csv") as fh:
        rows = list(csv.DictReader(fh))

    repo_root = str(HERE.parent.parent)
    if not (Path(repo_root) / ".git").exists():
        run(["git", "-C", repo_root, "init", "-b", "main"], d)
        run(["git", "-C", repo_root, "add", "-A"], d)
        run(["git", "-C", repo_root, "commit", "-m", "chore: initialise repository from Module 1 structure"], d)
    run(["gh", "repo", "create", a.repo, "--public" if a.public else "--private", "--source", repo_root, "--push"], d)
    for k in SPRINTS:
        run(["gh", "label", "create", f"sprint-{k}", "--repo", a.repo, "--force"], d)
    for g in ("G1", "G2", "G3", "G4", "G5"):
        run(["gh", "label", "create", g, "--repo", a.repo, "--force"], d)
    for k, title in SPRINTS.items():
        run(["gh", "api", f"repos/{a.repo}/milestones", "-f", f"title={title}", "-f", f"description=Two-week sprint {k}"], d)
    project_url = run(["gh", "project", "create", "--owner", owner, "--title", f"{name} board", "--format", "json"], d, capture=True)
    number = json.loads(project_url)["number"] if project_url else "<PROJECT_NUMBER>"
    for r in rows:
        labels = [f"sprint-{r['sprint']}"] + ([r["goal"]] if r["goal"] != "-" else [])
        body = f"Path: `{r['path']}`\nStory points: {r['points']}\nModule: {r['module']}"
        cmd = ["gh", "issue", "create", "--repo", a.repo, "--title", f"{r['id']} {r['title']}", "--body", body,
               "--milestone", SPRINTS[r["sprint"]]]
        for lab in labels:
            cmd += ["--label", lab]
        out = run(cmd, d, capture=True)
        url = out.strip() if out else "<ISSUE_URL>"
        run(["gh", "project", "item-add", str(number), "--owner", owner, "--url", url], d)
        if r["status"] == "Done":
            run(["gh", "issue", "close", url, "--reason", "completed"], d)
    print("Done. Open the board, choose Group by: Milestone, and take the screenshot for the submission.")


if __name__ == "__main__":
    sys.exit(main())
