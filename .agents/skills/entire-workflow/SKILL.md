---
name: entire-workflow
description: >-
  Use this skill when interacting with Entire.io CLI (`entire`), inspecting checkpoints,
  searching past agent sessions or commits, generating recaps/dispatches, tracking trails,
  or checking why specific code lines exist in the repository (`gh/Pabl057/turku-sinceai`).
---

# Entire.io Agent Workflow Skill

This repository uses **Entire** (`entire` CLI) with Antigravity lifecycle hooks (`.agents/hooks.json`) and Git ref checkpoints (`.entire/settings.json`) to track agent sessions, checkpoints, and team collaboration across the 4-developer team.

## Core Principles

1. **Source of Truth**: Always use `entire agent-help` or `entire agent-help <command>` to check exact, currently-installed flags rather than guessing flags.
2. **Auto-Detected Repository**: Entire automatically detects the current repository (`gh/Pabl057/turku-sinceai`) from the `git` origin remote. Do **not** pass `--repo` unless explicitly targeting a different repository.
3. **Structured Output**: Append `--json` when querying `entire` programmatically.

## Key Commands for Agents

### 1. Inspect Status & Recent Team Activity
* **Check Entire status**:
  ```bash
  entire status
  ```
* **Summarize recent checkpoint activity**:
  ```bash
  entire recap
  ```
* **View overall activity**:
  ```bash
  entire activity
  ```

### 2. Search Checkpoints, Commits & Sessions
* **Semantic & keyword search across checkpoints, commits, and sessions**:
  ```bash
  entire search "<query>"
  ```
* **Inspect checkpoints**:
  ```bash
  entire checkpoint --help
  entire agent-help checkpoint
  ```
* **Explain why a line of code exists**:
  ```bash
  entire why <file>
  ```

### 3. Session & Trail Management
* **Inspect sessions**:
  ```bash
  entire agent-help session
  ```
* **Manage and view trails across branches**:
  * Note: Trail listing requires `--project` or `--mine`.
  ```bash
  entire agent-help trail
  entire trail list --mine
  ```
* **Generate a dispatch summarizing recent agent work for teammates**:
  ```bash
  entire dispatch
  ```

## Safety Boundaries
* **Safe Read-Only Commands**: `status`, `search`, `why`, `activity`, `recap`, `version`, `agent-help`, and read-only subcommands of `checkpoint`, `session`, and `trail`.
* **User-Managed Commands (Suggest, Do Not Run Unprompted)**: `auth`, `login`, `logout`, `enable`, `disable`, `clean`, `configure`, `agent`, `plugin`, `org`, `project`, `repo`, `cluster`, `service-account`.
