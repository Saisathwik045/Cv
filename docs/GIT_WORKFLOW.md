# Git workflow

Use one branch and PR per phase: phase/00-foundation, phase/01-data, etc.
After a phase is reviewed and merged, fetch origin, switch main, fast-forward from
origin/main, and create the next phase branch. Never base the next phase on an
unmerged/stale phase branch. Inspect status before switching; preserve user work.

Required before merging: lightweight tests, package installation, compile check,
configuration/plan checks and environment report on Python 3.11 and 3.12; review
research safeguards and confirm no data/checkpoints/secrets are staged. Later phases
add synthetic data, metric and interruption/resume checks appropriate to their scope.
GPU runs remain deliberate jobs, separate from pull-request CI.

Implemented now: `.github/workflows/checks.yml` runs read-only, CPU-only jobs on PRs
and pushes to main/phase branches; it requests only contents:read. No account secrets
are needed for this workflow. No paid runner or training trigger is configured.

Account settings still required: enable GitHub Actions if disabled, then configure a
main ruleset requiring PRs and both `foundation (3.11)` and `foundation (3.12)` status
checks after they appear, block force pushes/deletions, and require review according
to available collaborators. The workflow does not enforce branch protection itself.
Do not make a sole maintainer require an unavailable second reviewer.

Phase 0 must be committed locally and opened as a PR when authentication permits;
never merge it automatically. Shell Git/gh access and connector access are independent.
If shell transport fails but connector writes work, publish the tested tree through
GitHub's Git data API and verify its tree against the local commit.
