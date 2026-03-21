# CLAUDE.md

## Shared Skills

Before starting any work, read the relevant skills from `~/skills/skills/`.

### Always read
- ~/skills/skills/project-scaffold/SKILL.md
- ~/skills/skills/env-setup/SKILL.md
- ~/skills/skills/git-workflow/SKILL.md

### Read for data work
- ~/skills/skills/market-data/SKILL.md
- ~/skills/skills/edav/SKILL.md
- ~/skills/skills/experiment-logging/SKILL.md

### Read for ML/model work
- ~/skills/skills/keras3-pytorch/SKILL.md
- ~/skills/skills/notebook-workflow/SKILL.md
- ~/skills/skills/experiment-workflow/SKILL.md
- ~/skills/skills/paper-replication/SKILL.md

### Read for quant/finance work
- ~/skills/skills/vol-and-curves/SKILL.md
- ~/skills/skills/pricing/SKILL.md
- ~/skills/skills/quant-patterns/SKILL.md
- ~/skills/skills/backtesting/SKILL.md

### Read for testing
- ~/skills/skills/testing-conventions/SKILL.md

### Read for CI/CD (optional — add when project is mature)
- ~/skills/skills/ci-cd/SKILL.md

## Commands

### "init" — New Project

**Do NOT start scaffolding immediately.** First, have a conversation:

1. **Ask the user:**
   - What does this project do? (one sentence)
   - What type of work is it? (ML model, data pipeline, paper replication, quant tool, app, etc.)
   - What data sources will it use?
   - Are there any reference papers or existing projects to base this on?
   - Any specific packages or approaches already in mind?

2. **Based on the answers, update this CLAUDE.md:**
   - Rename this file to `CLAUDE_<directory_name>.md` (e.g. `CLAUDE_vol_pipeline.md`)
   - Fill in **Project Rules** with the decisions made
   - Fill in **Architecture** with the planned structure
   - Fill in **Current State** with the starting point
   - Add any constraints to **Do NOT**

3. **Confirm the plan with the user** — show them the updated CLAUDE.md
   and ask "Does this look right? Ready to go?"

4. **Only then start scaffolding** — read project-scaffold and env-setup skills,
   check for existing Poetry environment, scaffold the directory structure,
   and work on `main`.

### "upgrade" — Existing Project

**Do NOT start coding immediately.** First, understand the objective:

1. **Ask the user:**
   - What needs to change? (e.g. "migrate from CSV to DB", "add Heston model", "refactor training loop")
   - Why? (e.g. "CSV is too slow", "need stochastic vol", "code is messy")
   - Are there any reference materials? (papers, other projects, docs)
   - What should NOT break? (existing notebooks, trained models, API interfaces)
   - What does success look like? (e.g. "same results, faster loading", "new model calibrates with RMSE < 0.01")

2. **Read the existing CLAUDE.md** (or create one if missing) and understand
   the current project state.

3. **Based on the answers, update the CLAUDE.md:**
   - Add the upgrade objective to **Current State**
   - Add any new rules to **Project Rules**
   - Add constraints to **Do NOT** (e.g. "do not require model retraining")
   - Update **Architecture** if the structure will change

4. **Present the plan** — show the user:
   - What branch will be created
   - What steps will be taken (in order)
   - What will be tested after each step
   - What the merge criteria are

5. **Confirm with the user** — "Does this plan look right? Ready to proceed?"

6. **Only then start working** — read upgrade-repo and git-workflow skills,
   create the branch, and execute step by step.

## Keeping this file and PROGRESS.md up to date

**CLAUDE.md** is the source of truth for project configuration.
**PROGRESS.md** is the source of truth for what's been done and what's next.

When the user asks to change the project, update both files:

**CLAUDE.md updates:**
- Add new project rules under **Project Rules** when decisions are made
- Add constraints under **Do NOT** when the user says to avoid something
- Update **Architecture** when the project structure changes
- Update **Current State** when branches, active work, or known issues change

**PROGRESS.md updates:**
- Move completed work from "In Progress" to "Done"
- Add new work to "In Progress" when starting
- Add planned work to "Next" when discussed
- Record key decisions under "Decisions" with date and reasoning

The user should never need to edit these files by hand. If they want to change
something, they tell Claude and Claude updates the code and both files.

## Project Rules

<!-- Rules will be added here during init or upgrade conversation -->

## Architecture

<!-- Will be filled in after the init/upgrade conversation -->

## Current State

<!-- Will be updated as work progresses -->

## Do NOT

<!-- Constraints will be added here during init/upgrade conversation -->

---

# PROGRESS.md Template

Create `PROGRESS.md` in the project root during init. Keep it updated
as work progresses. Format:

```markdown
# Progress

## Done
<!-- Completed milestones — add date and one-line summary -->

## In Progress
<!-- What's currently being worked on -->

## Next
<!-- Planned but not started — in priority order -->

## Decisions
<!-- Key decisions made, with date and reasoning -->
<!-- e.g. 2026-03-21: Chose Optuna over Keras Tuner — better pruning support -->

## References
<!-- Papers, links, other projects informing this work -->
```
