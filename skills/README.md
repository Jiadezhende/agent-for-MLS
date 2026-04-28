# Measurement Strategy Documents (Skills)

Each `.md` file in this directory is a **strategy document** that describes
how to measure a particular type of GPU hardware property or analyze a
specific kind of workload.

The agent loads these documents on demand via the `read_skill` tool.
Files starting with `_` are excluded from `list_skills` (they are templates
or internal references).

## Discoverable skills

Keep the discoverable skill set small and grouped by measurement workflow.
The worker should normally see:

- `gpu_profiling_overview.md`
- `memory_hierarchy.md`
- `throughput_resources.md`
- `clock_environment.md`

Older single-metric notes should either be merged into one of these files or
renamed with a leading `_` so `list_skills` hides them.

## Adding a new skill

1. Copy `_template.md` to a new file.
2. Prefer a workflow-level name over a metric-level name.
3. Fill in all sections.
4. Test by running the agent with a target that should trigger this skill.
