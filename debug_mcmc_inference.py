"""
Debug script for BlockMCMC inference.
Enables all debug flags to trace the confidence tracking and KV Cache behavior.

================================================================================
参数说明
================================================================================
- mcmc_alpha: 目标分布的 power 参数，用于计算置信度 log p^α(x)，影响 MH 接受率
- proposal_alpha: 提议分布的 power 参数，用于 token 选择时的 logits scaling
  - proposal_alpha=1.0: 标准解码（与 Phase 1 相同）
  - proposal_alpha>1.0: power-scaled 解码，提议更集中于高概率 token
- mcmc_temperature: 提议分布温度（默认 0.9）
- use_shift: 是否使用 shift 解码（仅在 enable_mcmc=False 时生效）

================================================================================
常用命令示例
================================================================================

# 1. 基本调试（不使用 KV Cache）
python debug_mcmc_inference.py

# 2. 调试 KV Cache（prefix 模式）
python debug_mcmc_inference.py --use_kv_cache --kv_cache_type prefix

# 3. 调试 KV Cache（dual 模式）
python debug_mcmc_inference.py --use_kv_cache --kv_cache_type dual

# 4. 调试 MCMC 提议生成的 KV Cache 加速
python debug_mcmc_inference.py --use_kv_cache --kv_cache_type dual --mcmc_use_kv_cache

# 5. 调试 MCMC 提议生成（不使用 KV Cache，即使主解码使用）
python debug_mcmc_inference.py --use_kv_cache --kv_cache_type dual --no_mcmc_kv_cache

# 6. 调试单步 MCMC（最小化调试）
python debug_mcmc_inference.py --n_mcmc_steps 1 --gen_length 64 --block_length 32

# 7. 调试多步 MCMC
python debug_mcmc_inference.py --n_mcmc_steps 5 --gen_length 128 --block_length 32

# 8. 禁用 MCMC 调试（仅调试扩散解码，退化为 BlockWiseDiffusionLLM）
python debug_mcmc_inference.py --disable_mcmc


# 10. 调试 power-scaled 提议分布（proposal_alpha=4.0）
python debug_mcmc_inference.py --proposal_alpha 4.0 --n_mcmc_steps 2


# 12. 完整调试配置
python debug_mcmc_inference.py \\
    --use_kv_cache --kv_cache_type dual \\
    --mcmc_use_kv_cache \\
    --n_mcmc_steps 2 \\
    --mcmc_alpha 4.0 --proposal_alpha 1.0 \\
    --gen_length 128 --block_length 32

================================================================================
"""

import os
import torch
import time
import logging
import argparse
from transformers import AutoTokenizer, AutoConfig
from vllm import distributed
from vllm.config import ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config

from dinfer.model import LLaDAMoeModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import MCMCThresholdParallelDecoder, BlockMCMCDiffusionLLM

__version__ = "1.1.0"

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Debug BlockMCMC Diffusion LLM Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Version
    parser.add_argument('--version', action='store_true', help='Print version and configuration info')
    
    # Model settings
    parser.add_argument('--model_path', type=str, 
                        default="/home/zhounan/models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct-fused",
                        help='Path to the model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    
    # Generation settings
    parser.add_argument('--gen_length', type=int, default=256, help='Generation length')
    parser.add_argument('--block_length', type=int, default=64, help='Block length')
    parser.add_argument('--temperature', type=float, default=0.9, help='Sampling temperature')
    parser.add_argument('--threshold', type=float, default=0.9, help='Confidence threshold')
    
    # MCMC settings
    parser.add_argument('--enable_mcmc', action='store_true', default=True, help='Enable MCMC refinement')
    parser.add_argument('--disable_mcmc', action='store_true', help='Disable MCMC refinement')
    parser.add_argument('--n_mcmc_steps', type=int, default=1, help='Number of MCMC steps per block (default: 1 for debug)')
    parser.add_argument('--mcmc_alpha', type=float, default=4.0, help='MCMC alpha (power parameter for target distribution)')
    parser.add_argument('--mcmc_temperature', type=float, default=0.9, help='MCMC temperature (default: 0.9)')
    
    # KV Cache settings
    parser.add_argument('--use_kv_cache', action='store_true', help='Enable KV cache for main decoding')
    parser.add_argument('--kv_cache_type', type=str, default='dual', choices=['prefix', 'dual'],
                        help='KV cache type: prefix or dual')
    
    # MCMC KV Cache settings
    parser.add_argument('--mcmc_use_kv_cache', action='store_true', default=False,
                        help='Enable KV cache acceleration in MCMC proposal generation')
    parser.add_argument('--no_mcmc_kv_cache', action='store_true',
                        help='Disable KV cache in MCMC proposal generation')
    
    # Proposal alpha settings
    parser.add_argument('--proposal_alpha', type=float, default=4.0,
                        help='Power parameter for proposal distribution in MCMC (default: 1.0). '
                             '1.0 = standard decoding, >1.0 = power-scaled decoding.')
    
    # Shift decoding (only effective when enable_mcmc=False)
    parser.add_argument('--use_shift', action='store_true', default=False,
                        help='Use shift decoding (only effective when MCMC is disabled)')
    
    # Debug settings
    parser.add_argument('--disable_debug', action='store_true', help='Disable debug output')
    parser.add_argument('--prompt', type=str, default=None, help='Custom prompt')
    
    args = parser.parse_args()
    
    # Handle enable/disable mcmc
    if args.disable_mcmc:
        args.enable_mcmc = False
    
    # Handle MCMC KV cache settings
    if args.no_mcmc_kv_cache:
        args.mcmc_use_kv_cache = False
    elif args.use_kv_cache and not args.no_mcmc_kv_cache:
        if not args.mcmc_use_kv_cache:
            args.mcmc_use_kv_cache = True
    
    return args


def print_version_info():
    """Print version and configuration information"""
    print("=" * 60)
    print(f"Debug BlockMCMC Inference Script v{__version__}")
    print("=" * 60)
    print("\n📦 Dependencies:")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n🔧 Debug flags available:")
    print("  - MCMCThresholdParallelDecoder.DEBUG_MCMC_DECODER")
    print("  - MCMCDiffusionIteration.DEBUG_MCMC_ITERATION")
    print("  - MCMCBlockRunner.DEBUG_MCMC_BLOCK_RUNNER")
    print("  - BlockMCMCDiffusionLLM.DEBUG_MCMC_GENERATE")
    
    print("\n📖 For full help, run: python debug_mcmc_inference.py --help")
    print("=" * 60)


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


def cleanup_distributed():
    """Clean up distributed environment"""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
        print("🧹 Distributed process group destroyed")


def main():
    args = parse_args()
    
    # Handle --version flag
    if args.version:
        print_version_info()
        return
    
    print("=" * 60)
    print(f"Debug BlockMCMC Inference v{__version__}")
    print("=" * 60)
    
    # 启用调试（除非显式禁用）
    if not args.disable_debug:
        enable_debug_flags()
    
    # Print configuration
    print("\n📋 Debug Configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  Device: {args.device}")
    print(f"  Generation length: {args.gen_length}")
    print(f"  Block length: {args.block_length}")
    
    print(f"\n🎯 MCMC Settings:")
    print(f"  MCMC enabled: {args.enable_mcmc}")
    if args.enable_mcmc:
        print(f"  MCMC steps: {args.n_mcmc_steps}")
        print(f"  MCMC alpha (target): {args.mcmc_alpha}")
        print(f"  MCMC temperature: {args.mcmc_temperature}")
        print(f"  Proposal alpha: {args.proposal_alpha}")
    else:
        print(f"  Use shift: {args.use_shift}")
    
    print(f"\n💾 KV Cache Settings:")
    print(f"  Main KV cache: {args.use_kv_cache}")
    if args.use_kv_cache:
        print(f"  KV cache type: {args.kv_cache_type}")
    print(f"  MCMC KV cache: {args.mcmc_use_kv_cache}")
    
    # 设置
    device = torch.device(args.device)
    
    # 初始化分布式
    print("\n🔧 Setting up distributed environment...")
    setup_distributed()
    
    # 加载模型
    print("\n📦 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    parallel_config = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        model = LLaDAMoeModelLM(config=model_config).eval()
        model.load_weights(args.model_path, torch_dtype=torch.bfloat16)
        model = model.to(device)
    
    # 创建解码器
    mask_id = tokenizer.convert_tokens_to_ids('[MASK]') if '[MASK]' in tokenizer.get_vocab() else 156895
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 156892
    
    print(f"Using mask_id={mask_id}, eos_id={eos_id}")
    
    decoder = MCMCThresholdParallelDecoder(
        temperature=args.temperature,
        threshold=args.threshold,
        mask_id=mask_id,
        eos_id=eos_id
    )
    
    # 创建 KV Cache 工厂
    cache_factory = None
    if args.use_kv_cache:
        cache_factory = KVCacheFactory(args.kv_cache_type)
        print(f"Using KV cache type: {args.kv_cache_type}")
    
    # 创建 DLLM
    print("\n🏗️ Creating BlockMCMCDiffusionLLM...")
    dllm = BlockMCMCDiffusionLLM(
        model=model,
        decoder=decoder,
        iterator_factory=BlockIteratorFactory(True),
        cache_factory=cache_factory,
        enable_mcmc=args.enable_mcmc,
        n_mcmc_steps=args.n_mcmc_steps,
        mcmc_alpha=args.mcmc_alpha,
        mcmc_temperature=args.mcmc_temperature,
        mcmc_use_kv_cache=args.mcmc_use_kv_cache,  # MCMC 提议生成是否使用 KV Cache
        proposal_alpha=args.proposal_alpha,  # 提议序列的 power scaling 参数
        use_shift=args.use_shift,  # 是否使用 shift 解码 (仅在 enable_mcmc=False 时生效)
        tokenizer=tokenizer,
        verbose=False  # 关闭 verbose，使用我们自己的调试输出
    )
    
    # 准备输入
    if args.prompt is None:
        prompt = "The vending machine sells drinks for 80 cents each. However, it gives you a 20-cent refund for each empty bottle you return. James has 2 dollars (200 cents). Assuming he can buy a drink, drink it, and immediately return the bottle for the refund (and repeat), how many drinks can he drink in total?"
    else:
        prompt = args.prompt
    
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(formatted_prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    print(f"\n📝 Input prompt: {prompt[:50]}...")
    print(f"Input tokens: {input_ids.shape[1]}")
    
    # 生成
    print("\n🚀 Generating (with debug output)...")
    print("=" * 60)
    
    start_time = time.time()
    output = dllm.generate(input_ids, gen_length=args.gen_length, block_length=args.block_length)
    end_time = time.time()
    
    print("=" * 60)
    print("\n📊 Results:")
    print(f"Generation time: {end_time - start_time:.2f}s")
    print(f"Output shape: {output.shape}")
    print(f"Total forwards: {dllm.num_forwards}")
    
    if args.enable_mcmc and dllm.proposal_generator is not None:
        print(f"  - Diffusion forwards: {dllm.diff_iteration.num_forwards}")
        print(f"  - Proposal forwards: {dllm.proposal_generator.num_forwards}")
    
    # 解码输出
    generated_text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n📄 Generated text:\n{generated_text}")
    
    # 额外调试信息
    print("\n🔍 Debug info:")
    generated_part = output[0, input_ids.shape[1]:]
    mask_count = (generated_part == mask_id).sum().item()
    eos_count = (generated_part == eos_id).sum().item()
    unique_tokens = torch.unique(generated_part).shape[0]
    
    print(f"  Mask tokens remaining: {mask_count}")
    print(f"  EOS tokens: {eos_count}")
    print(f"  Unique tokens: {unique_tokens}")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup_distributed()