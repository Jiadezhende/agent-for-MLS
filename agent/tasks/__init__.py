"""
agent/tasks — Task plugin package.

Importing this package triggers registration of all built-in task types.
Add new task types by creating a subpackage and importing it here.
"""
import agent.tasks.hardware_probe  # noqa: F401  registers 'hardware_probe'
