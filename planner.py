from dataclasses import dataclass
from math import ceil

BYTES_PER_PARAMETER = {
    "FP32": 4.0,
    "FP16/BF16": 2.0,
    "INT8": 1.0,
    "INT4": 0.5,
}


@dataclass(frozen=True)
class DeploymentEstimate:
    base_model_memory_gb: float
    estimated_runtime_memory_gb: float
    minimum_gpu_count: int
    deployment_pattern: str
    rationale: list[str]
    validation_steps: list[str]


def estimate_base_model_memory_gb(parameters_billions: float, precision: str) -> float:
    if parameters_billions <= 0:
        raise ValueError("Model parameter count must be greater than zero.")
    if precision not in BYTES_PER_PARAMETER:
        raise ValueError(f"Unsupported precision: {precision}")

    bytes_total = parameters_billions * 1_000_000_000 * BYTES_PER_PARAMETER[precision]
    return bytes_total / (1024 ** 3)


def estimate_runtime_memory_gb(
    parameters_billions: float,
    precision: str,
    overhead_percent: float = 25.0,
) -> float:
    if overhead_percent < 0:
        raise ValueError("Overhead percentage cannot be negative.")

    base = estimate_base_model_memory_gb(parameters_billions, precision)
    return base * (1 + overhead_percent / 100.0)


def minimum_gpu_count(required_memory_gb: float, gpu_memory_gb: float) -> int:
    if required_memory_gb <= 0 or gpu_memory_gb <= 0:
        raise ValueError("Memory values must be greater than zero.")
    return max(1, ceil(required_memory_gb / gpu_memory_gb))


def recommend_deployment(
    parameters_billions: float,
    precision: str,
    gpu_memory_gb: float,
    overhead_percent: float,
    expected_concurrent_users: int,
    availability_required: bool,
) -> DeploymentEstimate:
    if expected_concurrent_users < 1:
        raise ValueError("Expected concurrent users must be at least 1.")

    base = estimate_base_model_memory_gb(parameters_billions, precision)
    runtime = estimate_runtime_memory_gb(
        parameters_billions, precision, overhead_percent
    )
    gpus = minimum_gpu_count(runtime, gpu_memory_gb)

    rationale = []
    validation_steps = [
        "Benchmark the selected model on the target GPU before production.",
        "Measure latency, throughput, GPU utilization, and peak VRAM usage.",
        "Load-test expected prompt sizes, context lengths, and concurrency.",
        "Validate authentication, network exposure, logging, and failure behavior.",
    ]

    if gpus == 1:
        if expected_concurrent_users <= 5 and not availability_required:
            pattern = "Single GPU inference node"
            rationale.append(
                "The planning estimate fits on one GPU and the workload is small."
            )
        else:
            pattern = "Replicated single-GPU inference nodes behind a load balancer"
            rationale.append(
                "The model fits on one GPU, while concurrency or availability favors replicas."
            )
    else:
        if expected_concurrent_users <= 5 and not availability_required:
            pattern = "Multi-GPU inference node with model parallelism"
            rationale.append(
                "The estimated model footprint exceeds one GPU's memory capacity."
            )
        else:
            pattern = "Replicated multi-GPU inference nodes behind a load balancer"
            rationale.append(
                "The model requires multiple GPUs and the workload also benefits from replicas."
            )

    rationale.append(
        f"Estimated runtime memory is {runtime:.1f} GiB with "
        f"{overhead_percent:.0f}% planning headroom."
    )
    rationale.append(
        f"At least {gpus} GPU(s) of {gpu_memory_gb:.0f} GiB each are required "
        "for the memory-fit estimate."
    )

    if availability_required:
        rationale.append(
            "High availability was requested, so the recommendation avoids a single inference node."
        )

    if expected_concurrent_users >= 20:
        rationale.append(
            "Higher concurrency makes horizontal scaling and load testing especially important."
        )

    return DeploymentEstimate(
        base_model_memory_gb=base,
        estimated_runtime_memory_gb=runtime,
        minimum_gpu_count=gpus,
        deployment_pattern=pattern,
        rationale=rationale,
        validation_steps=validation_steps,
    )


def format_customer_summary(
    model_name: str,
    parameters_billions: float,
    precision: str,
    gpu_memory_gb: float,
    expected_concurrent_users: int,
    estimate: DeploymentEstimate,
) -> str:
    rationale = "\n".join(f"- {item}" for item in estimate.rationale)
    validation = "\n".join(f"- {item}" for item in estimate.validation_steps)

    return f"""# LLM Deployment Recommendation

## Workload
- Model: {model_name}
- Parameters: {parameters_billions:.1f}B
- Precision: {precision}
- GPU memory per device: {gpu_memory_gb:.0f} GiB
- Expected concurrent users: {expected_concurrent_users}

## Planning Estimate
- Model-weight memory: {estimate.base_model_memory_gb:.1f} GiB
- Estimated runtime memory: {estimate.estimated_runtime_memory_gb:.1f} GiB
- Minimum GPU count for memory fit: {estimate.minimum_gpu_count}

## Recommended Pattern
**{estimate.deployment_pattern}**

## Why
{rationale}

## Production Validation
{validation}

## Important Limitation
This tool is a capacity-planning aid, not a performance benchmark. Actual GPU
requirements depend on model architecture, framework, context length, KV cache,
batching, kernels, quantization implementation, and workload behavior.
"""
