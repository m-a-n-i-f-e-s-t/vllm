# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example demonstrating inference with PowerCoder, a retention-based model.

PowerCoder uses power retention mechanism instead of standard attention,
enabling efficient long-context modeling with constant-size state.

Run with:
    python retention_example.py

Note: When using chunked prefill with retention, the retention layer's chunk_size
must be a multiple of max_num_batched_tokens. The model will validate this
constraint at initialization.
"""

from vllm import LLM, SamplingParams

# Create prompts
prompts = [
    "def fibonacci(n):",
    "# Sort a list using quicksort\ndef quicksort(arr):",
    "class BinaryTree:",
]

# Create sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=64,
)

# Initialize PowerCoder model
# Note: Retention supports chunked prefill as long as retention's chunk_size
# is a multiple of max_num_batched_tokens
llm = LLM(
    model="manifestai/powercoder-3b",
    trust_remote_code=True,
    max_model_len=4096,
    # enforce_eager=True,
    enable_prefix_caching=False,
    # compilation_config=dict(level=3, ),
)

# Generate completions
print("=" * 80)
print("Generating with PowerCoder (Retention-based model)")
print("=" * 80)

outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\nPrompt: {prompt!r}")
    print(f"Generated: {generated_text!r}")
    print("-" * 80)

