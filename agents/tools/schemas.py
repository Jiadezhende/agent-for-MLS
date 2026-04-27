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
                "Run a kernel under NVIDIA Nsight Compute to collect hardware "
                "performance counters. Use this to cross-verify measurements, "
                "detect clock throttling, check cache efficiency, or measure "
                "memory bandwidth from the hardware counter side. "
                "source_type='cuda_source' compiles internally; "
                "source_type='binary' runs an existing workspace binary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_type": {
                        "type": "string",
                        "enum": ["cuda_source", "binary"],
                        "description": (
                            "'cuda_source' to provide CUDA source code (Executor compiles it); "
                            "'binary' to run a binary already in the workspace."
                        ),
                    },
                    "source_or_path": {
                        "type": "string",
                        "description": (
                            "If source_type='cuda_source': the full CUDA source code. "
                            "If source_type='binary': workspace-relative path to the binary."
                        ),
                    },
                    "kernel_name": {
                        "type": "string",
                        "description": (
                            "CUDA kernel function name to profile. "
                            "ncu will only instrument this kernel."
                        ),
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of ncu metric names to collect. "
                            "Examples: ['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum', "
                            "'sm__cycles_elapsed.avg.per_second', "
                            "'l2__throughput.avg.pct_of_peak_sustained_elapsed']"
                        ),
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
                    "timeout_s": {
                        "type": "integer",
                        "default": 600,
                        "description": "Total timeout for ncu run in seconds.",
                    },
                },
                "required": ["source_type", "source_or_path", "kernel_name", "metrics"],
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
    {
        "type": "function",
        "function": {
            "name": "profile_with_torch",
            "description": (
                "Run Python code under PyTorch Profiler to capture operator-level "
                "statistics. Use this to identify hotspot operators in a PyTorch "
                "model or function, measure GPU time per operator, and see memory "
                "allocation patterns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "python_code": {
                        "type": "string",
                        "description": (
                            "Python code that defines and calls the operation to profile. "
                            "The Executor wraps it with torch.profiler automatically. "
                            "Must import torch and define the operation inline."
                        ),
                    },
                    "op_name": {
                        "type": "string",
                        "description": (
                            "Human-readable name for this operation "
                            "(used in logs and cache keys)."
                        ),
                    },
                    "num_iters": {
                        "type": "integer",
                        "default": 100,
                        "description": "Number of iterations to run (for warmup + profiling).",
                    },
                    "timeout_s": {
                        "type": "integer",
                        "default": 300,
                    },
                },
                "required": ["python_code", "op_name"],
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
            "name": "find_binary",
            "description": (
                "Search the filesystem for a required binary (nvcc, ncu, nsys) that "
                "was not found in PATH. Updates the executor config for this session "
                "if found. Call this when you receive a binary_not_found infrastructure error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "binary_name": {
                        "type": "string",
                        "enum": ["nvcc", "ncu", "nsys"],
                        "description": "Name of the binary to locate.",
                    },
                },
                "required": ["binary_name"],
            },
        },
    },
]
