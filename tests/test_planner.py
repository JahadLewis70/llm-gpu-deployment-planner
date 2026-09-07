import pytest
from planner import (
    estimate_base_model_memory_gb,
    estimate_runtime_memory_gb,
    minimum_gpu_count,
    recommend_deployment,
)

def test_fp16_memory_estimate_for_7b_model():
    result = estimate_base_model_memory_gb(7, "FP16/BF16")
    assert 13.0 < result < 13.1

def test_int4_uses_less_memory_than_fp16():
    fp16 = estimate_base_model_memory_gb(7, "FP16/BF16")
    int4 = estimate_base_model_memory_gb(7, "INT4")
    assert int4 < fp16

def test_runtime_headroom_increases_estimate():
    base = estimate_base_model_memory_gb(7, "FP16/BF16")
    runtime = estimate_runtime_memory_gb(7, "FP16/BF16", 25)
    assert runtime == pytest.approx(base * 1.25)

def test_minimum_gpu_count_rounds_up():
    assert minimum_gpu_count(50, 24) == 3

def test_high_availability_recommends_replicas_when_model_fits():
    estimate = recommend_deployment(
        parameters_billions=7,
        precision="INT8",
        gpu_memory_gb=24,
        overhead_percent=25,
        expected_concurrent_users=3,
        availability_required=True,
    )
    assert "Replicated" in estimate.deployment_pattern
