# Agent Guidance

## PR Review Workflow

When helping with a pull request review, separate context gathering from review comments. Use the repo-local `review-nvbench` skill when available.

The default mode is context gathering: use `ci/util/pr_review_context.sh`, explain the issue, PR intent, implementation strategy, and suggested file review order, but do not produce review findings until the user explicitly asks for feedback. If no issue or PR context is discoverable, explicitly flag that the review is missing required context before proceeding.

Follow `docs/pr_review.md` for the full review workflow and NVBench-specific review focus.
