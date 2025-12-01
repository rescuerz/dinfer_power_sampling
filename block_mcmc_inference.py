"""
BlockMCMC Inference Script

使用 BlockMCMCDiffusionLLM 进行推理，支持 MCMC Power Sampling 精炼。

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

# 1. 基本使用（默认启用 MCMC，不使用 KV Cache）
python block_mcmc_inference.py --verbose

# 2. 禁用 MCMC（仅使用扩散解码，退化为 BlockWiseDiffusionLLM）
python block_mcmc_inference.py --disable_mcmc --verbose

# 3. 使用 KV Cache（prefix 模式）
python block_mcmc_inference.py --use_kv_cache --kv_cache_type prefix --verbose

# 4. 使用 KV Cache（dual 模式，推荐）
python block_mcmc_inference.py --use_kv_cache --kv_cache_type dual --verbose

# 5. MCMC 提议生成使用 KV Cache 加速（需要先启用 KV Cache）
python block_mcmc_inference.py --use_kv_cache --kv_cache_type dual --mcmc_use_kv_cache --verbose

# 6. MCMC 提议生成不使用 KV Cache（即使主解码使用 KV Cache）
python block_mcmc_inference.py --use_kv_cache --kv_cache_type dual --no_mcmc_kv_cache --verbose

# 7. 使用 power-scaled 提议分布（proposal_alpha=4.0）
python block_mcmc_inference.py --proposal_alpha 4.0 --verbose

# 8. 完整配置示例（推荐生产环境）
python block_mcmc_inference.py \\
    --use_kv_cache --kv_cache_type dual \\
    --enable_mcmc --n_mcmc_steps 3 \\
    --mcmc_alpha 4.0 --proposal_alpha 1.0 \\
    --mcmc_use_kv_cache \\
    --gen_length 256 --block_length 32 \\
    --verbose

================================================================================
"""

import os
import torch
import time
import argparse
from transformers import AutoTokenizer, AutoConfig
from vllm import distributed
from vllm.config import ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config

from dinfer.model import LLaDAMoeModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import MCMCThresholdParallelDecoder, BlockMCMCDiffusionLLM

__version__ = "1.1.0"


def parse_args():
    parser = argparse.ArgumentParser(
        description='BlockMCMC Diffusion LLM Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python block_mcmc_inference.py
  
  # With KV Cache (dual mode)
  python block_mcmc_inference.py --use_kv_cache --kv_cache_type dual
  
  # MCMC with KV Cache acceleration
  python block_mcmc_inference.py --use_kv_cache --mcmc_use_kv_cache
        """
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
    parser.add_argument('--n_mcmc_steps', type=int, default=3, help='Number of MCMC steps per block')
    parser.add_argument('--mcmc_alpha', type=float, default=4.0, help='MCMC alpha (power parameter for target distribution)')
    parser.add_argument('--mcmc_temperature', type=float, default=0.9, help='MCMC temperature (default: 0.9)')
    
    # KV Cache settings
    parser.add_argument('--use_kv_cache', action='store_true', help='Enable KV cache for main decoding')
    parser.add_argument('--kv_cache_type', type=str, default='dual', choices=['prefix', 'dual'],
                        help='KV cache type: prefix (前缀缓存) or dual (双向缓存)')
    
    # MCMC KV Cache settings
    parser.add_argument('--mcmc_use_kv_cache', action='store_true', default=False,
                        help='Enable KV cache acceleration in MCMC proposal generation')
    parser.add_argument('--no_mcmc_kv_cache', action='store_true',
                        help='Disable KV cache in MCMC proposal generation (even if main decoding uses KV cache)')
    
    # Proposal alpha settings
    parser.add_argument('--proposal_alpha', type=float, default=4.0,
                        help='Power parameter for proposal distribution in MCMC (default: 4.0). '
                             '1.0 = standard decoding, >1.0 = power-scaled decoding for better proposal quality.')
    
    # Shift decoding (only effective when enable_mcmc=False)
    parser.add_argument('--use_shift', action='store_true', default=False,
                        help='Use shift decoding (only effective when MCMC is disabled)')
    
    # Output settings
    parser.add_argument('--verbose', action='store_true', default= True, help='Enable verbose output')
    parser.add_argument('--prompt', type=str, default=None, help='Custom prompt')
    
    args = parser.parse_args()
    
    # Handle enable/disable mcmc
    if args.disable_mcmc:
        args.enable_mcmc = False
    
    # Handle MCMC KV cache settings
    # 如果显式禁用，则设置为 False
    if args.no_mcmc_kv_cache:
        args.mcmc_use_kv_cache = False
    # 如果启用了主 KV cache 且没有显式禁用 MCMC KV cache，默认启用
    elif args.use_kv_cache and not args.no_mcmc_kv_cache:
        # 如果用户没有显式设置 mcmc_use_kv_cache，则跟随 use_kv_cache
        if not args.mcmc_use_kv_cache:
            args.mcmc_use_kv_cache = True
    
    return args


def print_version_info():
    """Print version and configuration information"""
    print("=" * 60)
    print(f"BlockMCMC Inference Script v{__version__}")
    print("=" * 60)
    print("\n📦 Dependencies:")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n🔧 Available options:")
    print("  KV Cache types: prefix, dual")
    print("  MCMC parameters: alpha, temperature, n_steps")
    print("  MCMC KV Cache: --mcmc_use_kv_cache / --no_mcmc_kv_cache")
    
    print("\n📖 For full help, run: python block_mcmc_inference.py --help")
    print("=" * 60)


def setup_distributed():
    """Initialize distributed environment"""
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'
    
    distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
    distributed.initialize_model_parallel(1, backend='nccl')


def load_model(model_path, device):
    """Load model and tokenizer"""
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"Loading model from {model_path}...")
    parallel_config = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = LLaDAMoeModelLM(config=model_config).eval()
        model.load_weights(model_path, torch_dtype=torch.bfloat16)
        model = model.to(device)
    
    return model, tokenizer


def create_dllm(model, tokenizer, args):
    """Create BlockMCMCDiffusionLLM instance"""
    # Get mask_id and eos_id from tokenizer
    mask_id = tokenizer.convert_tokens_to_ids('[MASK]') if '[MASK]' in tokenizer.get_vocab() else 156895
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 156892
    
    print(f"Using mask_id={mask_id}, eos_id={eos_id}")
    
    # Create decoder
    decoder = MCMCThresholdParallelDecoder(
        temperature=args.temperature,
        threshold=args.threshold,
        mask_id=mask_id,
        eos_id=eos_id
    )
    
    # Create KV cache factory if enabled
    cache_factory = None
    if args.use_kv_cache:
        cache_factory = KVCacheFactory(args.kv_cache_type)
        print(f"Using KV cache type: {args.kv_cache_type}")
    
    # Create BlockMCMCDiffusionLLM
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
        verbose=args.verbose
    )
    
    return dllm


def prepare_input(tokenizer, prompt, device):
    """Prepare input for generation"""
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(formatted_prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    return input_ids, formatted_prompt


def main():
    args = parse_args()
    
    # Handle --version flag
    if args.version:
        print_version_info()
        return
    
    print("=" * 60)
    print(f"BlockMCMC Diffusion LLM Inference v{__version__}")
    print("=" * 60)
    
    # Print configuration
    print("\n📋 Configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  Device: {args.device}")
    print(f"  Generation length: {args.gen_length}")
    print(f"  Block length: {args.block_length}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Threshold: {args.threshold}")
    
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
    
    print(f"\n🔧 Other Settings:")
    print(f"  Verbose: {args.verbose}")
    
    # Setup
    print("\n🔧 Setting up distributed environment...")
    setup_distributed()
    
    device = torch.device(args.device)
    
    # Load model
    print("\n📦 Loading model...")
    model, tokenizer = load_model(args.model_path, device)
    
    # Create DLLM
    print("\n🏗️ Creating BlockMCMCDiffusionLLM...")
    dllm = create_dllm(model, tokenizer, args)
    
    # Prepare prompt
    if args.prompt is None:
        # Default test prompt (math reasoning)
        prompt = """The vending machine sells drinks for 80 cents each. However, it gives you a 20-cent refund for each empty bottle you return. James has 2 dollars (200 cents). Assuming he can buy a drink, drink it, and immediately return the bottle for the refund (and repeat), how many drinks can he drink in total?"""
    else:
        prompt = args.prompt
    
    print("\n📝 Input prompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    # Prepare input
    input_ids, formatted_prompt = prepare_input(tokenizer, prompt, device)
    print(f"\nInput tokens: {input_ids.shape[1]}")
    
    # Generate
    print("\n🚀 Generating...")
    print("=" * 60)
    
    start_time = time.time()
    with torch.no_grad():
        output = dllm.generate(
            input_ids,
            gen_length=args.gen_length,
            block_length=args.block_length
        )
    end_time = time.time()
    
    generation_time = end_time - start_time
    
    # Decode output
    print("\n" + "=" * 60)
    print("📊 Results:")
    print("=" * 60)
    
    # Statistics
    generated_tokens = output.shape[1] - input_ids.shape[1]
    tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0
    
    print(f"\n⏱️ Generation time: {generation_time:.2f} seconds")
    print(f"📈 Generated tokens: {generated_tokens}")
    print(f"🚄 Speed: {tokens_per_second:.2f} tokens/second")
    print(f"🔄 Total forward passes: {dllm.num_forwards}")
    
    if args.enable_mcmc and dllm.proposal_generator is not None:
        print(f"   - Diffusion forwards: {dllm.diff_iteration.num_forwards}")
        print(f"   - Proposal forwards: {dllm.proposal_generator.num_forwards}")
    
    # Decode and print output
    generated_text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    
    print("\n📄 Generated text:")
    print("-" * 40)
    print(generated_text)
    print("-" * 40)
    
    # Debug info
    if args.verbose:
        print("\n🔍 Debug info:")
        generated_part = output[0, input_ids.shape[1]:]
        mask_count = (generated_part == 156895).sum().item()
        eos_count = (generated_part == 156892).sum().item()
        unique_tokens = torch.unique(generated_part).shape[0]
        
        print(f"  Mask tokens remaining: {mask_count}")
        print(f"  EOS tokens: {eos_count}")
        print(f"  Unique tokens: {unique_tokens}")
        print(f"  First 20 generated tokens: {generated_part[:20].tolist()}")
    
    print("\n✅ Done!")
    return output, generated_text


if __name__ == '__main__':
    main()
