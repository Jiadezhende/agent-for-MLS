# Measurement Strategy Documents (Skills)

Each `.md` file in this directory is a **strategy document** that describes
how to measure a particular type of GPU hardware property or analyze a
specific kind of workload.

The agent loads these documents on demand via the `read_skill` tool.
Files starting with `_` are excluded from `list_skills` (they are templates
or internal references).

## File naming convention

Use lowercase snake_case, e.g.:

- `memory_latency.md`
- `memory_bandwidth.md`
- `cache_capacity.md`
- `clock_measurement.md`
- `bank_conflict.md`
- `operator_analysis.md`

## Adding a new skill

1. Copy `_template.md` to a new file.
2. Fill in all sections.
3. Test by running the agent with a target that should trigger this skill.
