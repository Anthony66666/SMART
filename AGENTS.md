# SMART Workflow

Use codebase-memory-mcp as the default discovery layer for this repository.

## Discovery Order

1. Use `get_architecture` to understand the project layout before reading files.
2. Use `search_graph` to find functions, classes, routes, modules, and related symbols.
3. Use `trace_path` to follow inbound and outbound call paths.
4. Use `get_code_snippet` to read only the specific symbols that matter.
5. Fall back to `rg` or direct file reads only for string literals, YAML values, shell scripts, generated artifacts, or when graph results are insufficient.

## Project Memory

Keep durable project state in `docs/`.

- `docs/spec.md`: current goal, scope, constraints, acceptance criteria, and active plan
- `docs/progress.md`: append one concise entry after each meaningful completed task
- `docs/decisions.md`: record durable technical decisions and rationale
- `docs/next.md`: keep the live next actions and blockers current

## Update Rules

- Read existing `docs/*.md` files before substantial work.
- Update `docs/spec.md` only when the current plan or scope changes.
- Append to `docs/progress.md` after meaningful completed work that changed code, tooling, docs, or operating procedure.
- Append to `docs/decisions.md` only when the decision will matter in future sessions.
- Refresh `docs/next.md` before ending the task if project state changed.
- Keep entries concise and factual. Do not paste shell transcripts.

## Project Notes

- This repository is the main implementation repo for SMART baseline and SMARTJEPA experiments.
- Keep SMART rollout and inference interfaces stable while adding JEPA training-time changes.
- Prefer touching the smallest number of files needed for each task.
