import streamlit as st
from planner import BYTES_PER_PARAMETER, format_customer_summary, recommend_deployment

st.set_page_config(
    page_title="GPU-Aware LLM Deployment Planner",
    page_icon="🧠",
    layout="wide",
)

st.title("GPU-Aware LLM Deployment Planner")
st.caption(
    "A lightweight capacity-planning and architecture tool for early-stage LLM deployment discussions."
)

with st.sidebar:
    st.header("Workload Inputs")
    model_name = st.text_input("Model name", value="Example 7B Model")
    parameters_billions = st.number_input(
        "Model parameters (billions)", min_value=0.1, value=7.0, step=0.1
    )
    precision = st.selectbox("Precision / quantization", list(BYTES_PER_PARAMETER))
    gpu_memory_gb = st.number_input(
        "GPU memory per device (GiB)", min_value=1.0, value=24.0, step=1.0
    )
    overhead_percent = st.slider(
        "Planning headroom (%)", min_value=0, max_value=100, value=25, step=5
    )
    expected_concurrent_users = st.number_input(
        "Expected concurrent users", min_value=1, value=5, step=1
    )
    availability_required = st.checkbox("High availability required", value=False)

estimate = recommend_deployment(
    parameters_billions=parameters_billions,
    precision=precision,
    gpu_memory_gb=gpu_memory_gb,
    overhead_percent=overhead_percent,
    expected_concurrent_users=expected_concurrent_users,
    availability_required=availability_required,
)

c1, c2, c3 = st.columns(3)
c1.metric("Model-weight memory", f"{estimate.base_model_memory_gb:.1f} GiB")
c2.metric("Runtime planning estimate", f"{estimate.estimated_runtime_memory_gb:.1f} GiB")
c3.metric("Minimum GPUs for memory fit", estimate.minimum_gpu_count)

st.subheader("Recommended Deployment Pattern")
st.success(estimate.deployment_pattern)

st.subheader("Architecture Reasoning")
for item in estimate.rationale:
    st.write(f"- {item}")

st.subheader("Production Validation Checklist")
for item in estimate.validation_steps:
    st.checkbox(item, value=False)

st.info(
    "This is a planning estimate, not a performance benchmark. Real inference "
    "requirements depend on model architecture, runtime, context length, batching, "
    "KV cache, kernels, and workload."
)

report = format_customer_summary(
    model_name=model_name,
    parameters_billions=parameters_billions,
    precision=precision,
    gpu_memory_gb=gpu_memory_gb,
    expected_concurrent_users=expected_concurrent_users,
    estimate=estimate,
)

st.download_button(
    "Download customer-facing recommendation",
    data=report,
    file_name="deployment_recommendation.md",
    mime="text/markdown",
)
