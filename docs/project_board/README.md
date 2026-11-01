# Project board

* `tickets.csv` is the single source of truth for 24 tickets across five two-week sprints.
* `board_preview.png` is a rendering of the plan. The graded screenshot must come from the live GitHub Projects board.

## Publish to GitHub
```bash
gh auth login && gh auth refresh -s project
python docs/project_board/setup_github.py --repo YOUR_USER/cfpb-service-failure-analytics --dry-run   # review
python docs/project_board/setup_github.py --repo YOUR_USER/cfpb-service-failure-analytics             # create
```
The script creates the repository (private), labels (sprint-0..4, G1..G5), five milestones (one per sprint),
24 issues, closes the Done tickets, and creates a Projects board with all issues added. In the board UI choose
**Group by: Milestone** to show the sprints, then take the screenshot.
