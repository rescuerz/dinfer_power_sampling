"""
BlockMCMC Inference Script

使用 BlockMCMCDiffusionLLM 进行推理，支持 MCMC Power Sampling 精炼。

Usage:
    python block_mcmc_inference.py [--enable_mcmc] [--n_mcmc_steps N] [--mcmc_alpha ALPHA]
    
Example:
    python block_mcmc_inference.py --enable_mcmc --n_mcmc_steps 3 --mcmc_alpha 4.0
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


def parse_args():
    parser = argparse.ArgumentParser(description='BlockMCMC Diffusion LLM Inference')
    
    # Model settings
    parser.add_argument('--model_path', type=str, 
                        default="/home/zhounan/models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct-fused",
                        help='Path to the model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    
    # Generation settings
    parser.add_argument('--gen_length', type=int, default=256, help='Generation length')
    parser.add_argument('--block_length', type=int, default=32, help='Block length')
    parser.add_argument('--temperature', type=float, default=0.9, help='Sampling temperature')
    parser.add_argument('--threshold', type=float, default=0.9, help='Confidence threshold')
    
    # MCMC settings
    parser.add_argument('--enable_mcmc', action='store_true', default=True, help='Enable MCMC refinement')
    parser.add_argument('--disable_mcmc', action='store_true', help='Disable MCMC refinement')
    parser.add_argument('--n_mcmc_steps', type=int, default=3, help='Number of MCMC steps per block')
    parser.add_argument('--mcmc_alpha', type=float, default=4.0, help='MCMC alpha (power parameter)')
    parser.add_argument('--mcmc_temperature', type=float, default=0.9, help='MCMC temperature')
    
    # KV Cache settings
    parser.add_argument('--use_kv_cache', action='store_true', help='Enable KV cache')
    parser.add_argument('--kv_cache_type', type=str, default='dual', choices=['prefix', 'dual'],
                        help='KV cache type')
    
    # Output settings
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--prompt', type=str, default=None, help='Custom prompt')
    
    args = parser.parse_args()
    
    # Handle enable/disable mcmc
    if args.disable_mcmc:
        args.enable_mcmc = False
    
    return args


def setup_distributed():
    """Initialize distributed environment"""
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
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
    
    print("=" * 60)
    print("BlockMCMC Diffusion LLM Inference")
    print("=" * 60)
    
    # Print configuration
    print("\n📋 Configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  Device: {args.device}")
    print(f"  Generation length: {args.gen_length}")
    print(f"  Block length: {args.block_length}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Threshold: {args.threshold}")
    print(f"  MCMC enabled: {args.enable_mcmc}")
    if args.enable_mcmc:
        print(f"  MCMC steps: {args.n_mcmc_steps}")
        print(f"  MCMC alpha: {args.mcmc_alpha}")
        print(f"  MCMC temperature: {args.mcmc_temperature}")
    print(f"  KV cache: {args.use_kv_cache}")
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
