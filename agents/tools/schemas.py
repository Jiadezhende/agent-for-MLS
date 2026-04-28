"""
agents/tools/schemas.py — OpenAI function-calling JSON schemas for all 9 tools.

Keeping schemas here (separate from implementations in tools/ and cuda_executor.py)
means llm/client.py can feed them to the API, while registry.py holds
the callables. They are linked only by the tool name string.
"""

TOOL_SCHEMAS: list[dict] = [
    # ------------------------------------------------------------------
    # Knowledge tools
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": (
                "List all available measurement strategy documents (skills). "
                "Call this first to discover what strategies exist, then use "
                "read_skill to load the one most relevant to your target metric."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": (
                "Read the full content of a measurement strategy document. "
                "Each skill describes when to use a particular approach, what "
                "CUDA kernel pattern to apply, how to interpret the output, "
                "and what anomalies to watch for."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Skill name (filename without .md extension). "
                            "Use list_skills to get valid names."
                        ),
                    }
                },
                "required": ["name"],
            },
        },
    },

    # ------------------------------------------------------------------
    # Execution tools (route through Executor)
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "run_cuda_probe",
            "description": (
                "Compile and run a CUDA kernel source file. Use this when the "
                "kernel measures hardware properties via self-timing (clock64) "
                "and you need the stdout output as your primary measurement. "
                "The Executor handles compilation, sandboxing, and stdout capture. "
                "Do NOT use this for profiling — use profile_with_ncu instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "Complete CUDA C++ source code (.cu content). "
                            "MUST be actual code starting with #include or __global__. "
                            "NEVER pass a skill name, filename, or description here."
                        ),
                    },
                    "probe_name": {
                        "type": "string",
                        "description": (
                            "Short identifier for this probe (used in cache key "
                            "and logs). E.g. 'pointer_chase_dram', 'bandwidth_global'."
                        ),
                    },
                    "compile_flags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extra nvcc flags (e.g. [\"-O3\", \"-arch=sm_86\"]).",
                        "default": [],
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command-line arguments passed to the compiled binary.",
                        "default": [],
                    },
                    "timeout_s": {
                        "type": "integer",
                        "description": "Execution timeout in seconds.",
                        "default": 60,
                    },
                },
                "required": ["source", "probe_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "profile_with_ncu",
            "description": (
                "Run a kernel under NVIDIA Nsight Compute to collect hardware performance "
                "counters. Compiles with -lineinfo for source-line correlation and saves a "
                ".ncu-rep report (openable in Nsight Compute GUI). "
                "Specify what to measure via section_set (recommended for broad analysis), "
                "sections (targeted deep-dive), or metrics (exact counter names). "
                "source_type='cuda_source' compiles and profiles directly; "
                "source_type='binary' profiles an existing workspace binary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_type": {
                        "type": "string",
                        "enum": ["cuda_source", "binary"],
                        "description": (
                            "'cuda_source': provide CUDA C source — compiled with -lineinfo "
                            "internally, kernel is NOT pre-run before profiling. "
                            "'binary': workspace-relative path to an already-compiled binary."
                        ),
                    },
                    "source_or_path": {
                        "type": "string",
                        "description": (
                            "If source_type='cuda_source': full CUDA C source code. "
                            "If source_type='binary': workspace-relative binary path "
                            "(e.g. from a prior run_cuda_probe result's binary_path field)."
                        ),
                    },
                    "kernel_name": {
                        "type": "string",
                        "description": (
                            "Kernel function name or regex passed to ncu --kernel-name. "
                            "ncu will only instrument matching kernels."
                        ),
                    },
                    "section_set": {
                        "type": "string",
                        "description": (
                            "Predefined ncu section set (--set). Recommended starting point. "
                            "'default' covers Speed-of-Light, memory, compute utilisation. "
                            "'full' adds all sections (slow). 'roofline' adds roofline model. "
                            "Leave empty if using sections or metrics instead."
                        ),
                        "default": "",
                    },
                    "sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Targeted ncu sections (--section). Use for focused analysis. "
                            "Examples: ['SpeedOfLight', 'MemoryWorkloadAnalysis', "
                            "'ComputeWorkloadAnalysis', 'Occupancy', 'SchedulerStats']. "
                            "Can be combined with metrics."
                        ),
                        "default": [],
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Explicit ncu metric names (--metrics). Use when you need "
                            "precise counters not covered by sections. "
                            "Examples: ['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum', "
                            "'sm__cycles_elapsed.avg.per_second']. "
                            "At least one of metrics / sections / section_set is required."
                        ),
                        "default": [],
                    },
                    "compile_flags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Extra nvcc flags. -lineinfo is always injected automatically."
                        ),
                        "default": [],
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "timeout_s": {
                        "type": "integer",
                        "default": 600,
                        "description": "Total timeout for the ncu run in seconds.",
                    },
                },
                "required": ["source_type", "source_or_path", "kernel_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "profile_with_nsys",
            "description": (
                "Run a program under NVIDIA Nsight Systems to capture the "
                "CPU-GPU timeline. Use this to understand kernel launch overhead, "
                "CUDA stream execution order, and host-device synchronization patterns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_type": {
                        "type": "string",
                        "enum": ["cuda_source", "python_script", "binary"],
                        "description": (
                            "'cuda_source': CUDA source code (Executor compiles and profiles). "
                            "'python_script': Python code run under nsys. "
                            "'binary': existing workspace binary."
                        ),
                    },
                    "source_or_path": {
                        "type": "string",
                        "description": "Source code string or workspace-relative path.",
                    },
                    "compile_flags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "duration_s": {
                        "type": "integer",
                        "default": 10,
                        "description": "Approximate profiling window in seconds.",
                    },
                    "timeout_s": {
                        "type": "integer",
                        "default": 300,
                    },
                },
                "required": ["source_type", "source_or_path"],
            },
        },
    },
    # ------------------------------------------------------------------
    # Recording tools (need AgentContext injection)
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "record_measurement",
            "description": (
                "Record a confirmed hardware measurement in the results. "
                "Every call MUST include at least one evidence string — "
                "a direct quote or path from a tool output earlier in this "
                "conversation. Do not call this with invented values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "The metric name, matching the target_spec key.",
                    },
                    "value": {
                        "description": "The measured value (number, string, or object).",
                        "oneOf": [
                            {"type": "number"},
                            {"type": "string"},
                            {"type": "object"},
                        ],
                    },
                    "unit": {
                        "type": ["string", "null"],
                        "description": "Unit of measurement (e.g. 'cycles', 'GB/s', 'MHz').",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence in this measurement (0–1).",
                    },
                    "method": {
                        "type": "string",
                        "description": (
                            "Brief description of how this was measured "
                            "(e.g. 'pointer-chasing kernel with 256MB array > L2')."
                        ),
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": (
                            "At least one string from a previous tool output that "
                            "supports this measurement."
                        ),
                    },
                },
                "required": [
                    "metric", "value", "unit", "confidence", "method", "evidence"
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_event",
            "description": (
                "Record an anomaly, decision, or observation for the engineering "
                "reasoning log. Use this when you detect a non-standard environment "
                "(e.g. clock throttling, SM masking, API spoofing) or make a "
                "significant methodological decision."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": (
                            "Event category. Examples: 'clock_locked', 'sm_masked', "
                            "'api_spoofed', 'retry_measurement', 'strategy_switch'."
                        ),
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["info", "warn", "error"],
                    },
                    "detail": {
                        "type": "string",
                        "description": "Human-readable explanation for the LLM-as-Judge.",
                    },
                },
                "required": ["type", "severity", "detail"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_results",
            "description": (
                "Finalize and submit all measured results. Call this exactly once "
                "when you have recorded all target metrics with acceptable confidence. "
                "After this call the agent loop terminates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "2–5 sentence summary of methodology, anomalies found, "
                            "and overall confidence. This is read by the LLM-as-Judge."
                        ),
                    }
                },
                "required": ["summary"],
            },
        },
    },
    # ------------------------------------------------------------------
    # Environment tools
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "probe_environment",
            "description": (
                "Scan the filesystem for nvcc, ncu, and nsys binaries without executing them. "
                "Call this when error_class='infrastructure' and error='binary_not_found'. "
                "If binaries are found, the Executor is reconfigured automatically so "
                "subsequent tool calls use the discovered paths. "
                "Does NOT run any subprocess — only filesystem stat checks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "force_rescan": {
                        "type": "boolean",
                        "description": (
                            "If true, re-scans even if a binary appears to be on PATH. "
                            "Use only if you suspect PATH-based resolution is wrong. "
                            "Default: false."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
]
