The folder contains some benchmark scripts to benchmark the inference speed of dInfer and Fast-dLLM:
* benchmark_dataset.py: takes the benchmark dataset as input and benchmark the average inference speed of dInfer with LLaDA-MoE on the benchmark dataset.
* benchmark_ep.py: benchmark the inference speed of dInfer with LLaDA-MoE on a task collected from gsm8k.
* benchmark.py: similar to benchmark_ep.py. It benchmark dInfer with LLaDA, instead of LLaDA-MoE.
* benchmark_dataset_fastdllm.py: reproduce Fast-dLLM results on LLaDA-MoE.
