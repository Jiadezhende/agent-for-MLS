"""
agents — Multi-agent pipeline package.

Importing this package triggers registration of all built-in agent types.
Add new agent types by creating a subpackage and importing it here.
"""
import agents.agents.hardware_probe_agent    # noqa: F401  registers 'hardware_probe'
import agents.agents.op_profiler_agent       # noqa: F401  registers 'op_profiler'
import agents.agents.bottleneck_analyst_agent  # noqa: F401  registers 'bottleneck_analyst'
import agents.agents.kernel_optimizer_agent  # noqa: F401  registers 'kernel_optimizer'
