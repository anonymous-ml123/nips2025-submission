# Anonymous Benchmark for Evaluating MLLMs in Real-World Reasoning

Welcome to the official codebase for our benchmark: a dataset designed to evaluate Multi-modal Large Language Models (MLLMs) in real-world physical reasoning tasks.

## About the Benchmark

This benchmark provides a comprehensive evaluation of MLLMs across the following categories:
- Spatial Reasoning
- Commonsense Knowledge
- Environment Interaction

The dataset consists of newly created diagrams with image-question pairs, carefully curated through a standardized annotation and filtering pipeline.

## Quick Start

### Setup

Most models are evaluated using the [OpenRouter API](https://openrouter.ai/) for efficiency. For models not supported by OpenRouter, we use [VLLM](https://github.com/vllm-project/vllm) with default inference hyperparameters.

To install required dependencies:

```bash
pip install -r requirements.txt
```

### Evaluation

To access the dataset, please clone the following temporary anonymous repository:

```shell
cd ./data
git clone https://huggingface.co/datasets/anonymous-ml123/nips2025-data
```

To evaluate different models on the benchmark:

```shell
# test by openrouter api, generate the result in a JSON file. 
python eval.py --model all # test all the models
python eval.py --model gpt-4o # test one model
```

To calculate accuracy from result files:

```python
python calc_acc.py --jsons gpt-4o_open_router_check.json # gpt-4o accuracy
```