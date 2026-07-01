---
name: review-nvbench
description: Use this skill when the user invokes /review-nvbench to review an NVBench pull request, PR branch, commit range, or diff. Supports context, feedback, and adversarial modes.
---

# Review NVBench Pull Request

Read `AGENTS.md` and `docs/pr_review.md` before acting.

Use one of these modes:

- `context`: Gather PR context and explain the change without review findings.
- `feedback`: Produce normal review findings after the human reviewer has inspected the PR.
- `adversarial`: Challenge assumptions, design direction, compatibility risks, and hidden failure modes after the PR context is understood.

If no mode is provided, default to `context`.

## Context Mode

Run or inspect `ci/util/pr_review_context.sh` when reviewing a checked-out PR branch. If the user provided a PR number, pass `--pr <number>`. If the user provided an issue number, pass `--issue <number>`.

Summarize:

- The linked issue or PR context.
- The problem the PR is trying to solve.
- How the PR attempts to solve it.
- The important changed files and a sensible human review order.
- Any missing context, such as no discoverable linked issue.

Do not produce review findings in context mode.

## Feedback Mode

Before producing findings, make sure the context pass has happened. If it has not, gather and summarize the PR context first.

Review for correctness, benchmark validity, performance, API compatibility, build and packaging impact, and consistency with existing code. Let the changed files determine which risks matter, while paying attention to relevant NVBench contracts such as CUDA stream ordering, timing boundaries, cold/batch/CPU-only measurement behavior, manual timer semantics, axis generation, output schemas, optional CUPTI/NVML behavior, CMake options, Python bindings, and test coverage.

Report findings first, ordered by severity, with file and line references. Do not post comments to GitHub; the human reviewer decides which findings to use.

## Adversarial Mode

Before producing adversarial feedback, make sure the PR intent and implementation strategy are understood.

Focus on whether the approach is sound:

- Hidden assumptions that could fail.
- Simpler or safer alternative designs.
- Compatibility, layering, maintenance, or rollout risks.
- Cases where the implementation solves the local issue but weakens a broader NVBench contract.

Keep this distinct from normal feedback. It is acceptable for adversarial mode to produce questions or risks rather than concrete blocking findings.
