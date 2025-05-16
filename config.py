MODEL_API_MAP = {
    'open_router': {
        'llama-4-scout': 'meta-llama/llama-4-scout',
        'llama-4-maverick': 'meta-llama/llama-4-maverick',
        
        'gemma-3-4b-it': 'google/gemma-3-4b-it',
        'gemma-3-12b-it': 'google/gemma-3-12b-it',
        'gemma-3-27b-it': 'google/gemma-3-27b-it',
        
        'grok-2-vision-1212': 'x-ai/grok-2-vision-1212',
        
        'gpt-4o': 'openai/gpt-4o',
        'gpt-4o-mini': 'openai/gpt-4o-mini',
        
        'gemini-2.5-flash-preview': 'google/gemini-2.5-flash-preview',
        
        'claude-3.7-sonnet': 'anthropic/claude-3.7-sonnet',
        
        'pixtral-12b': 'mistralai/pixtral-12b',
        
        'phi-4-multimodal-instruct': 'microsoft/phi-4-multimodal-instruct',
        
        'qwen-2.5-vl-7b-instruct': 'qwen/qwen-2.5-vl-7b-instruct',
        'qwen2.5-vl-72b-instruct': 'qwen/qwen2.5-vl-72b-instruct',
    },

    'vllm': {
        'idefics3-8b-llama3': '/workspace/models/Idefics3-8B-Llama3',
        'llava-onevision-7b': '/workspace/models/llava-onevision-qwen2-7b-ov-hf',
        'smolvlm-2b': '/workspace/models/SmolVLM2-2.2B-Instruct',
    }
}


DETECT_PROMPT = '''
### Instructions:
1. Compare the **answer** and the **ground truth**.
2. Ignore any extra information in the answer beyond the option letter.
3. Only consider the **option letter** (e.g., 'A', 'B', 'C', 'D') when determining if the answer matches the ground truth.

### Input:
Answer: [answer]  
Ground truth: [ground_truth]  

### Task:  
Does the option letter in the answer match the ground truth?  

- Respond **only** with 'true' if the option letter matches.  
- Respond **only** with 'false' if the option letter does not match.  

### Examples:  
1. Answer: "A: The capital of France is Paris."  
    Ground truth: "A"  
    Response: "true"  

2. Answer: "C: Incorrect option."  
    Ground truth: "A"  
    Response: "false"

Now process the following input:
Answer: {}
Ground truth: {}
'''