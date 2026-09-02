# dev-contribution-quality-analysis — Learnings

## Patterns That Work

## Mistakes to Avoid

- [2026-09-01] A corrupt .git husk makes extract-commits.sh die mid-scan (set -e, git exit 128), silently truncating raw-commits.csv; mismatched repo counts vs mr-acceptances.csv are the tell. Check exit code, not just output.
- [2026-07-11] The 1.75x logic-error stat is CodeRabbit's (Dec 2025, 470 PRs), not Veracode's; only the 2.74x XSS stat is Veracode's.
- [2026-07-11] The '54% bugs / 242.7% incidents' figures cited alongside DORA 2025 are from Faros AI's 2026 telemetry report, not DORA's own survey — verify before citing as DORA.
## Domain Knowledge

- [2026-07-11] By mid-2026, fully agent-authored PRs (Devin, Codex cloud, Claude Code delegated sessions) merged under a human identity are common; score them as a distinct provenance class, not blended human craft.
## Open Questions

## Consolidated Principles

