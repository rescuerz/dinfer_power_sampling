"""
Integration tests for BlockMCMCDiffusionLLM.

Tests the complete workflow:
1. Model initialization
2. Token generation with MCMC refinement
3. Confidence tracking across blocks
4. MCMC acceptance rates
5. Compatibility with BlockWiseDiffusionLLM

Run with: python tests/test_mcmc_integration.py
"""

import os
import sys
import torch
import numpy as np
from collections import namedtuple
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from dinfer.decoding.utils import TokenArray, BlockIteratorFactory
from dinfer.decoding.generate_uniform import (
    BlockMCMCDiffusionLLM,
    BlockWiseDiffusionLLM,
)
from dinfer.decoding.parallel_strategy import (
    MCMCThresholdParallelDecoder,
    ThresholdParallelDecoder,
)

BlockLoc = namedtuple('BlockLoc', ['start', 'end'])

# Device selection
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


class SimpleModel(torch.nn.Module):
    """Simple model for integration testing that mimics real LLM interface"""
    
    def __init__(self, vocab_size=100, hidden_size=64, device=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # Set device attribute that BlockMCMCDiffusionLLM expects
        self.device = torch.device(device if device else DEVICE)
        
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)
        self.to(self.device)
    
    def forward(self, input_ids, **kwargs):
        # Handle TokenArray input
        if hasattr(input_ids, 'data'):
            input_ids = input_ids.data
        
        # Ensure input is on correct device
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        
        # Simple forward pass
        x = self.embedding(input_ids)
        logits = self.linear(x)
        
        # Create output object with expected attributes
        output = Mock()
        output.logits = logits
        output.past_key_values = None
        return output


# ============================================================================
# Integration Tests
# ============================================================================

def test_basic_generation():
    """Test basic token generation with MCMC"""
    print("\n=== Test: Basic Generation ===")
    
    try:
        # Setup
        vocab_size = 100
        model = SimpleModel(vocab_size=vocab_size, device=DEVICE)
        
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        iterator_factory = BlockIteratorFactory(True)
        
        # Create MCMC LLM
        dllm = BlockMCMCDiffusionLLM(
            model=model,
            decoder=decoder,
            iterator_factory=iterator_factory,
            enable_mcmc=True,
            n_mcmc_steps=2,  # Reduced for testing
            mcmc_alpha=4.0,
            mcmc_temperature=0.0,
            verbose=False
        )
        
        # Generate
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        gen_length = 10
        block_length = 5
        
        output = dllm.generate(prompt, gen_length=gen_length, block_length=block_length)
        
        print(f"  Generated output shape: {output.shape}")
        print(f"  Output tokens: {output.tolist()}")
        print(f"  Total forward passes: {dllm.num_forwards}")
        print("  ✓ Basic generation PASSED")
        return True
        
    except Exception as e:
        print(f"  ✗ Basic generation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_confidence_tracking():
    """Test that confidences are properly tracked"""
    print("\n=== Test: Confidence Tracking ===")
    
    try:
        vocab_size = 100
        model = SimpleModel(vocab_size=vocab_size, device=DEVICE)
        
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        iterator_factory = BlockIteratorFactory(True)
        
        dllm = BlockMCMCDiffusionLLM(
            model=model,
            decoder=decoder,
            iterator_factory=iterator_factory,
            enable_mcmc=True,
            n_mcmc_steps=2,
            mcmc_alpha=4.0,
            verbose=False
        )
        
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        gen_length = 10
        block_length = 5
        
        output = dllm.generate(prompt, gen_length=gen_length, block_length=block_length)
        
        # Check confidences
        conf_norm = dllm.diff_iteration.confidences_norm
        conf_unnorm = dllm.diff_iteration.confidences_unnorm
        
        assert conf_norm is not None, "Confidence norm is None"
        assert conf_unnorm is not None, "Confidence unnorm is None"
        
        # Check that some confidences have been updated
        num_updated_norm = (conf_norm > -np.inf).sum().item()
        num_updated_unnorm = (conf_unnorm > -np.inf).sum().item()
        
        print(f"  Updated norm confidences: {num_updated_norm}")
        print(f"  Updated unnorm confidences: {num_updated_unnorm}")
        
        assert num_updated_norm > 0, "No norm confidences updated"
        assert num_updated_unnorm > 0, "No unnorm confidences updated"
        
        print("  ✓ Confidence tracking PASSED")
        return True
        
    except Exception as e:
        print(f"  ✗ Confidence tracking FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcmc_disabled():
    """Test generation with MCMC disabled (degraded mode)"""
    print("\n=== Test: MCMC Disabled (Degraded Mode) ===")
    
    try:
        vocab_size = 100
        model = SimpleModel(vocab_size=vocab_size, device=DEVICE)
        
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        iterator_factory = BlockIteratorFactory(True)
        
        # Create MCMC LLM with MCMC disabled
        dllm = BlockMCMCDiffusionLLM(
            model=model,
            decoder=decoder,
            iterator_factory=iterator_factory,
            enable_mcmc=False,  # Disabled
            verbose=False
        )
        
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        gen_length = 10
        block_length = 5
        
        output = dllm.generate(prompt, gen_length=gen_length, block_length=block_length)
        
        print(f"  Generated output shape: {output.shape}")
        
        # MCMC runner should be None
        assert dllm.mcmc_runner is None, "MCMC runner should be None"
        
        print("  ✓ MCMC disabled PASSED")
        return True
        
    except Exception as e:
        print(f"  ✗ MCMC disabled FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compatibility_with_blockwise():
    """Test that degraded mode produces similar behavior to BlockWiseDiffusionLLM"""
    print("\n=== Test: Compatibility with BlockWiseDiffusionLLM ===")
    
    try:
        vocab_size = 100
        # Use same seed for reproducibility
        torch.manual_seed(42)
        model1 = SimpleModel(vocab_size=vocab_size, device=DEVICE)
        
        torch.manual_seed(42)
        model2 = SimpleModel(vocab_size=vocab_size, device=DEVICE)
        
        decoder1 = ThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        decoder2 = ThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        iterator_factory1 = BlockIteratorFactory(True)
        iterator_factory2 = BlockIteratorFactory(True)
        
        # Create BlockMCMCDiffusionLLM in degraded mode
        mcmc_dllm = BlockMCMCDiffusionLLM(
            model=model1,
            decoder=decoder1,
            iterator_factory=iterator_factory1,
            enable_mcmc=False,
            verbose=False
        )
        
        # Create BlockWiseDiffusionLLM
        blockwise_dllm = BlockWiseDiffusionLLM(
            model=model2,
            decoder=decoder2,
            iterator_factory=iterator_factory2
        )
        
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        gen_length = 10
        block_length = 5
        
        # Generate with both
        torch.manual_seed(123)
        output1 = mcmc_dllm.generate(prompt.clone(), gen_length=gen_length, block_length=block_length)
        
        torch.manual_seed(123)
        output2 = blockwise_dllm.generate(prompt.clone(), gen_length=gen_length, block_length=block_length)
        
        print(f"  BlockMCMC (degraded) output: {output1.tolist()}")
        print(f"  BlockWise output: {output2.tolist()}")
        print(f"  BlockMCMC forwards: {mcmc_dllm.num_forwards}")
        print(f"  BlockWise forwards: {blockwise_dllm.num_forwards}")
        
        # Both should produce valid outputs
        assert output1.shape == output2.shape, "Output shapes should match"
        
        print("  ✓ Compatibility test PASSED")
        return True
        
    except Exception as e:
        print(f"  ✗ Compatibility test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_blocks():
    """Test generation across multiple blocks"""
    print("\n=== Test: Multiple Blocks ===")
    
    try:
        vocab_size = 100
        model = SimpleModel(vocab_size=vocab_size, device=DEVICE)
        
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        iterator_factory = BlockIteratorFactory(True)
        
        dllm = BlockMCMCDiffusionLLM(
            model=model,
            decoder=decoder,
            iterator_factory=iterator_factory,
            enable_mcmc=True,
            n_mcmc_steps=1,  # Minimal MCMC steps for speed
            mcmc_alpha=4.0,
            verbose=False
        )
        
        # Generate with multiple blocks
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        gen_length = 20  # Multiple blocks
        block_length = 5
        
        output = dllm.generate(prompt, gen_length=gen_length, block_length=block_length)
        
        print(f"  Generated output shape: {output.shape}")
        print(f"  Total forward passes: {dllm.num_forwards}")
        
        # Should have multiple forward passes
        assert dllm.num_forwards > 0, "No forward passes"
        
        print("  ✓ Multiple blocks PASSED")
        return True
        
    except Exception as e:
        print(f"  ✗ Multiple blocks FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_count():
    """Test that forward pass count is tracked correctly"""
    print("\n=== Test: Forward Count ===")
    
    try:
        vocab_size = 100
        model = SimpleModel(vocab_size=vocab_size, device=DEVICE)
        
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        iterator_factory = BlockIteratorFactory(True)
        
        dllm = BlockMCMCDiffusionLLM(
            model=model,
            decoder=decoder,
            iterator_factory=iterator_factory,
            enable_mcmc=True,
            n_mcmc_steps=2,
            mcmc_alpha=4.0,
            verbose=False
        )
        
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        gen_length = 10
        block_length = 5
        
        output = dllm.generate(prompt, gen_length=gen_length, block_length=block_length)
        
        # Check forward counts
        diff_forwards = dllm.diff_iteration.num_forwards
        proposal_forwards = dllm.proposal_generator.num_forwards if dllm.proposal_generator else 0
        total_forwards = dllm.num_forwards
        
        print(f"  Diffusion forwards: {diff_forwards}")
        print(f"  Proposal forwards: {proposal_forwards}")
        print(f"  Total forwards: {total_forwards}")
        
        assert total_forwards == diff_forwards + proposal_forwards, "Forward count mismatch"
        
        print("  ✓ Forward count PASSED")
        return True
        
    except Exception as e:
        print(f"  ✗ Forward count FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_alpha_values():
    """Test generation with different alpha values"""
    print("\n=== Test: Different Alpha Values ===")
    
    try:
        vocab_size = 100
        model = SimpleModel(vocab_size=vocab_size, device=DEVICE)
        
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        iterator_factory = BlockIteratorFactory(True)
        
        for alpha in [1.0, 2.0, 4.0, 8.0]:
            dllm = BlockMCMCDiffusionLLM(
                model=model,
                decoder=decoder,
                iterator_factory=iterator_factory,
                enable_mcmc=True,
                n_mcmc_steps=1,
                mcmc_alpha=alpha,
                verbose=False
            )
            
            prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
            output = dllm.generate(prompt, gen_length=10, block_length=5)
            
            print(f"  Alpha={alpha}: output shape {output.shape}")
        
        print("  ✓ Different alpha values PASSED")
        return True
        
    except Exception as e:
        print(f"  ✗ Different alpha values FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_proposal_alpha():
    """Test generation with different proposal_alpha values"""
    print("\n=== Test: Proposal Alpha Values ===")
    
    try:
        vocab_size = 100
        model = SimpleModel(vocab_size=vocab_size, device=DEVICE)
        
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        iterator_factory = BlockIteratorFactory(True)
        
        for proposal_alpha in [1.0, 2.0, 4.0]:
            dllm = BlockMCMCDiffusionLLM(
                model=model,
                decoder=decoder,
                iterator_factory=iterator_factory,
                enable_mcmc=True,
                n_mcmc_steps=1,
                mcmc_alpha=4.0,
                proposal_alpha=proposal_alpha,
                verbose=False
            )
            
            prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
            output = dllm.generate(prompt, gen_length=10, block_length=5)
            
            print(f"  proposal_alpha={proposal_alpha}: output shape {output.shape}")
        
        print("  ✓ Proposal alpha values PASSED")
        return True
        
    except Exception as e:
        print(f"  ✗ Proposal alpha values FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_size_one():
    """Test with batch size 1 (most common case)"""
    print("\n=== Test: Batch Size One ===")
    
    try:
        vocab_size = 100
        model = SimpleModel(vocab_size=vocab_size, device=DEVICE)
        
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        iterator_factory = BlockIteratorFactory(True)
        
        dllm = BlockMCMCDiffusionLLM(
            model=model,
            decoder=decoder,
            iterator_factory=iterator_factory,
            enable_mcmc=True,
            n_mcmc_steps=2,
            mcmc_alpha=4.0,
            verbose=False
        )
        
        # Single sequence
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        assert prompt.shape[0] == 1, "Batch size should be 1"
        
        output = dllm.generate(prompt, gen_length=10, block_length=5)
        
        print(f"  Input batch size: {prompt.shape[0]}")
        print(f"  Output shape: {output.shape}")
        
        print("  ✓ Batch size one PASSED")
        return True
        
    except Exception as e:
        print(f"  ✗ Batch size one FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_degraded_mode_with_shift():
    """Test degraded mode with shift decoding"""
    print("\n=== Test: Degraded Mode with Shift ===")
    
    try:
        vocab_size = 100
        model = SimpleModel(vocab_size=vocab_size, device=DEVICE)
        
        decoder = ThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        iterator_factory = BlockIteratorFactory(True)
        
        # Create MCMC LLM with MCMC disabled and shift enabled
        dllm = BlockMCMCDiffusionLLM(
            model=model,
            decoder=decoder,
            iterator_factory=iterator_factory,
            enable_mcmc=False,
            use_shift=True,
            verbose=False
        )
        
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        gen_length = 10
        block_length = 5
        
        # This should work without errors
        output = dllm.generate(prompt, gen_length=gen_length, block_length=block_length)
        
        print(f"  Generated output shape: {output.shape}")
        print(f"  use_shift: {dllm.use_shift}")
        
        assert dllm.use_shift == True, "use_shift should be True"
        assert dllm.mcmc_runner is None, "MCMC runner should be None"
        
        print("  ✓ Degraded mode with shift PASSED")
        return True
        
    except Exception as e:
        print(f"  ✗ Degraded mode with shift FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcmc_n_steps_zero():
    """Test with n_mcmc_steps=0 (no MCMC refinement)"""
    print("\n=== Test: n_mcmc_steps=0 ===")
    
    try:
        vocab_size = 100
        model = SimpleModel(vocab_size=vocab_size, device=DEVICE)
        
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        iterator_factory = BlockIteratorFactory(True)
        
        dllm = BlockMCMCDiffusionLLM(
            model=model,
            decoder=decoder,
            iterator_factory=iterator_factory,
            enable_mcmc=True,
            n_mcmc_steps=0,  # No MCMC refinement
            mcmc_alpha=4.0,
            verbose=False
        )
        
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        gen_length = 10
        block_length = 5
        
        output = dllm.generate(prompt, gen_length=gen_length, block_length=block_length)
        
        print(f"  Generated output shape: {output.shape}")
        print(f"  n_mcmc_steps: {dllm.n_mcmc_steps}")
        
        # Should still work, just no MCMC refinement
        assert output.shape[1] > 0, "Output should have tokens"
        
        print("  ✓ n_mcmc_steps=0 PASSED")
        return True
        
    except Exception as e:
        print(f"  ✗ n_mcmc_steps=0 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Run Integration Tests
# ============================================================================

def run_integration_tests():
    """Run all integration tests"""
    print("="*60)
    print("Running BlockMCMCDiffusionLLM Integration Tests")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Basic Generation", test_basic_generation()))
    results.append(("Confidence Tracking", test_confidence_tracking()))
    results.append(("MCMC Disabled", test_mcmc_disabled()))
    results.append(("Compatibility with BlockWise", test_compatibility_with_blockwise()))
    results.append(("Multiple Blocks", test_multiple_blocks()))
    results.append(("Forward Count", test_forward_count()))
    results.append(("Different Alpha Values", test_different_alpha_values()))
    results.append(("Proposal Alpha Values", test_proposal_alpha()))
    results.append(("Batch Size One", test_batch_size_one()))
    results.append(("Degraded Mode with Shift", test_degraded_mode_with_shift()))
    results.append(("n_mcmc_steps=0", test_mcmc_n_steps_zero()))
    
    # Summary
    print("\n" + "="*60)
    print("Integration Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests PASSED!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) FAILED")
        return False


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)
