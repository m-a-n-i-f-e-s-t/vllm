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
import os

# Create prompts
prompts = [
    "def fibonacci(n):",
    "# Sort a list using quicksort\ndef quicksort(arr):",
    "class BinaryTree:",
]

# Create prompts
def generate_prompts(batch_size: int, prompt_length: int) -> list[str]:
    """Generate a list of random prompts.
    
    Args:
        batch_size: Number of prompts to generate
        prompt_length: Target length of each prompt in characters
        
    Returns:
        List of randomly generated prompts
    """
    prompts = []
    for i in range(batch_size):
        # Generate a random function name and args
        func_name = f"function_{i}"
        args = f"arg_{i}"
        
        # Create a prompt with target length by padding with comments
        prompt = f"def {func_name}({args}):"
        padding_length = prompt_length - len(prompt)
        if padding_length > 0:
            prompt = "# " + "x" * padding_length + "\n" + prompt
            
        prompts.append(prompt)
    return prompts

# Uncomment to programatically generate prompts
# prompts = generate_prompts(batch_size=16, prompt_length=8192)

# Create sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=128,
)


# Initialize PowerCoder model
# Note: Retention supports chunked prefill as long as retention's chunk_size
# is a multiple of max_num_batched_tokens
llm = LLM(
    model="manifestai/powercoder-3b",
    trust_remote_code=True,
    enable_prefix_caching=False,
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

