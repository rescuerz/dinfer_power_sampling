"""
Debug script for BlockMCMC inference.
Enables all debug flags to trace the confidence tracking issue.

Usage:
    python debug_mcmc_inference.py
"""

import os
import torch
import time
import logging
from transformers import AutoTokenizer, AutoConfig
from vllm import distributed
from vllm.config import ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config

from dinfer.model import LLaDAMoeModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import MCMCThresholdParallelDecoder, BlockMCMCDiffusionLLM

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 启用调试开关
# ============================================================================
def enable_debug_flags():
    """Enable all debug flags in MCMC components"""
    from dinfer.decoding.parallel_strategy import MCMCThresholdParallelDecoder
    from dinfer.decoding.generate_uniform import (
        MCMCDiffusionIteration, 
        MCMCBlockRunner, 
        BlockMCMCDiffusionLLM
    )
    
    # 启用调试
    MCMCThresholdParallelDecoder.DEBUG_MCMC_DECODER = True
    MCMCDiffusionIteration.DEBUG_MCMC_ITERATION = True
    MCMCBlockRunner.DEBUG_MCMC_BLOCK_RUNNER = True
    BlockMCMCDiffusionLLM.DEBUG_MCMC_GENERATE = True
    
    print("✅ All debug flags enabled")


def setup_distributed():
    """Initialize distributed environment"""
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12347'  # 使用不同端口避免冲突
    
    distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
    distributed.initialize_model_parallel(1, backend='nccl')


def main():
    print("=" * 60)
    print("Debug BlockMCMC Inference")
    print("=" * 60)
    
    # 启用调试
    enable_debug_flags()
    
    # 设置
    model_path = "/home/zhounan/models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct-fused"
    device = torch.device('cuda:0')
    
    # 初始化分布式
    print("\n🔧 Setting up distributed environment...")
    setup_distributed()
    
    # 加载模型
    print("\n📦 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    parallel_config = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = LLaDAMoeModelLM(config=model_config).eval()
        model.load_weights(model_path, torch_dtype=torch.bfloat16)
        model = model.to(device)
    
    # 创建解码器
    mask_id = 156895
    eos_id = 156892
    
    decoder = MCMCThresholdParallelDecoder(
        temperature=0.9,
        threshold=0.9,
        mask_id=mask_id,
        eos_id=eos_id
    )
    
    # 创建 DLLM（使用较小的参数以便调试）
    print("\n🏗️ Creating BlockMCMCDiffusionLLM...")
    dllm = BlockMCMCDiffusionLLM(
        model=model,
        decoder=decoder,
        iterator_factory=BlockIteratorFactory(True),
        cache_factory=None,  # 不使用 KV cache 简化调试
        enable_mcmc=True,
        n_mcmc_steps=1,  # 只做 1 步 MCMC 以便调试
        mcmc_alpha=4.0,
        mcmc_temperature=0.9,
        tokenizer=tokenizer,
        verbose=False  # 关闭 verbose，使用我们自己的调试输出
    )
    
    # 准备输入（使用较长的 prompt 来测试更多块）
    prompt = "The vending machine sells drinks for 80 cents each. However, it gives you a 20-cent refund for each empty bottle you return. James has 2 dollars (200 cents). Assuming he can buy a drink, drink it, and immediately return the bottle for the refund (and repeat), how many drinks can he drink in total?"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(formatted_prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    print(f"\n📝 Input prompt: {prompt[:50]}...")
    print(f"Input tokens: {input_ids.shape[1]}")
    
    # 生成（使用较大的长度来测试多个块）
    print("\n🚀 Generating (with debug output)...")
    print("=" * 60)
    
    gen_length = 256  # 较长的生成长度，测试多个块
    block_length = 64  # 标准块长度
    
    start_time = time.time()
    output = dllm.generate(input_ids, gen_length=gen_length, block_length=block_length)
    end_time = time.time()
    
    print("=" * 60)
    print("\n📊 Results:")
    print(f"Generation time: {end_time - start_time:.2f}s")
    print(f"Output shape: {output.shape}")
    print(f"Total forwards: {dllm.num_forwards}")
    
    # 解码输出
    generated_text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n📄 Generated text:\n{generated_text}")
    
    print("\n✅ Done!")


def cleanup_distributed():
    """Clean up distributed environment"""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
        print("🧹 Distributed process group destroyed")


if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup_distributed()
