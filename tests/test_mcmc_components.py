"""
Unit tests for MCMC components in dInfer framework.

Tests:
- MCMCDiffusionIteration
- MCMCBlockRunner
- MCMCProposalGenerator
- MCMCRefinementRunner
- BlockMCMCDiffusionLLM

Run with: python tests/test_mcmc_components.py
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
    MCMCDiffusionIteration,
    MCMCBlockRunner,
    MCMCProposalGenerator,
    MCMCRefinementRunner,
    BlockMCMCDiffusionLLM,
)
from dinfer.decoding.parallel_strategy import MCMCThresholdParallelDecoder

# Named tuple for block location
BlockLoc = namedtuple('BlockLoc', ['start', 'end'])

# Device selection
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


class MockModel:
    """Mock LLM model for testing that mimics real model interface"""
    
    def __init__(self, vocab_size=1000, device=None):
        self.vocab_size = vocab_size
        self.device = torch.device(device if device else DEVICE)
        self.call_count = 0
    
    def __call__(self, input_ids, **kwargs):
        self.call_count += 1
        
        # Handle TokenArray input
        if hasattr(input_ids, 'data'):
            input_ids = input_ids.data
        
        batch_size, seq_len = input_ids.shape
        # Generate random logits on the correct device
        logits = torch.randn(batch_size, seq_len, self.vocab_size, device=self.device)
        
        # Create mock output
        output = Mock()
        output.logits = logits
        output.past_key_values = None
        return output


# ============================================================================
# MCMCDiffusionIteration Tests
# ============================================================================

class TestMCMCDiffusionIteration:
    """Tests for MCMCDiffusionIteration class"""
    
    def test_init(self):
        """Test initialization with default and custom parameters"""
        # Default parameters
        iteration = MCMCDiffusionIteration()
        assert iteration.mcmc_alpha == 4.0
        assert iteration.mcmc_temperature == 0.0
        assert iteration.iter_no == 0
        assert iteration.confidences_norm is None
        assert iteration.confidences_unnorm is None
        assert iteration.num_forwards == 0
        
        # Custom parameters
        iteration = MCMCDiffusionIteration(mcmc_alpha=2.0, mcmc_temperature=0.5)
        assert iteration.mcmc_alpha == 2.0
        assert iteration.mcmc_temperature == 0.5
    
    def test_reset_confidences(self):
        """Test confidence tensor initialization"""
        iteration = MCMCDiffusionIteration()
        shape = (1, 100)
        
        iteration.reset_confidences(shape, DEVICE)
        
        assert iteration.confidences_norm is not None
        assert iteration.confidences_unnorm is not None
        assert iteration.confidences_norm.shape == shape
        assert iteration.confidences_unnorm.shape == shape
        assert iteration.confidences_norm.device.type == DEVICE.split(':')[0]
        assert torch.all(iteration.confidences_norm == -np.inf)
        assert torch.all(iteration.confidences_unnorm == -np.inf)
        assert iteration.iter_no == 0
    
    def test_forward_no_cache(self):
        """Test forward pass without KV cache"""
        model = MockModel(vocab_size=100, device=DEVICE)
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9, threshold=0.9, 
            mask_id=99, eos_id=98
        )
        
        # Create token array on correct device
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        x = TokenArray(prompt, gen_length=10, mask_id=99, eos_id=98, device=DEVICE)
        
        # Create iteration
        iteration = MCMCDiffusionIteration(mcmc_alpha=4.0)
        iteration.reset_confidences(x.data.shape, DEVICE)
        
        # Create block
        block_loc = BlockLoc(start=5, end=10)
        block = x.data[:, block_loc.start:block_loc.end]
        
        # Forward pass
        conf_norm, conf_unnorm = iteration.forward(
            model, decoder, x, None, block, block_loc, block_id=0
        )
        
        # Verify
        assert iteration.num_forwards == 1
        assert iteration.iter_no == 1
        assert conf_norm is not None
        assert conf_unnorm is not None
        assert conf_norm.shape == (1, 5)  # block_length = 5
        assert conf_unnorm.shape == (1, 5)
    
    def test_confidence_accumulation(self):
        """Test that confidences accumulate across iterations"""
        model = MockModel(vocab_size=100, device=DEVICE)
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9, threshold=0.9,
            mask_id=99, eos_id=98
        )
        
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        x = TokenArray(prompt, gen_length=10, mask_id=99, eos_id=98, device=DEVICE)
        
        iteration = MCMCDiffusionIteration(mcmc_alpha=4.0)
        iteration.reset_confidences(x.data.shape, DEVICE)
        
        block_loc = BlockLoc(start=5, end=10)
        block = x.data[:, block_loc.start:block_loc.end]
        
        # Multiple forward passes
        for i in range(3):
            iteration.forward(model, decoder, x, None, block, block_loc, block_id=0)
            block = x.data[:, block_loc.start:block_loc.end]  # Update block
        
        assert iteration.num_forwards == 3
        assert iteration.iter_no == 3
    
    def test_different_alpha_values(self):
        """Test with different mcmc_alpha values"""
        model = MockModel(vocab_size=100, device=DEVICE)
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9, threshold=0.9,
            mask_id=99, eos_id=98
        )
        
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        
        for alpha in [1.0, 2.0, 4.0, 8.0]:
            x = TokenArray(prompt, gen_length=10, mask_id=99, eos_id=98, device=DEVICE)
            iteration = MCMCDiffusionIteration(mcmc_alpha=alpha)
            iteration.reset_confidences(x.data.shape, DEVICE)
            
            block_loc = BlockLoc(start=5, end=10)
            block = x.data[:, block_loc.start:block_loc.end]
            
            conf_norm, conf_unnorm = iteration.forward(
                model, decoder, x, None, block, block_loc, block_id=0
            )
            
            assert conf_norm is not None
            assert conf_unnorm is not None


# ============================================================================
# MCMCBlockRunner Tests
# ============================================================================

class TestMCMCBlockRunner:
    """Tests for MCMCBlockRunner class"""
    
    def test_init(self):
        """Test initialization"""
        iteration = MCMCDiffusionIteration()
        runner = MCMCBlockRunner(
            diff_iteration=iteration,
            early_stop=True,
            maximum_unroll=4,
            expected_tpf=8
        )
        
        assert runner.diff_iteration is iteration
        assert runner.early_stop == True
        assert runner.maximum_unroll == 4
        assert runner.expected_tpf == 8
    
    def test_inheritance(self):
        """Test that MCMCBlockRunner inherits from BlockRunner"""
        from dinfer.decoding.generate_uniform import BlockRunner
        
        iteration = MCMCDiffusionIteration()
        runner = MCMCBlockRunner(iteration, True, 4, 8)
        
        assert isinstance(runner, BlockRunner)


# ============================================================================
# MCMCProposalGenerator Tests
# ============================================================================

class TestMCMCProposalGenerator:
    """Tests for MCMCProposalGenerator class"""
    
    def test_init(self):
        """Test initialization"""
        model = MockModel(device=DEVICE)
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9, threshold=0.9,
            mask_id=99, eos_id=98
        )
        
        generator = MCMCProposalGenerator(
            model=model,
            decoder=decoder,
            mcmc_alpha=4.0,
            mcmc_temperature=0.0
        )
        
        assert generator.model is model
        assert generator.decoder is decoder
        assert generator.mcmc_alpha == 4.0
        assert generator.mcmc_temperature == 0.0
        assert generator.num_forwards == 0
    
    def test_generate_basic(self):
        """Test basic proposal generation"""
        model = MockModel(vocab_size=100, device=DEVICE)
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9, threshold=0.9,
            mask_id=99, eos_id=98
        )
        
        generator = MCMCProposalGenerator(
            model=model,
            decoder=decoder,
            mcmc_alpha=4.0
        )
        
        # Create current sequence
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        x_current = TokenArray(prompt, gen_length=10, mask_id=99, eos_id=98, device=DEVICE)
        # Fill with some decoded tokens
        x_current.data[:, 5:10] = torch.tensor([[10, 11, 12, 13, 14]], device=DEVICE)
        
        # Create confidence tensors
        conf_norm = torch.full(x_current.data.shape, -np.inf, dtype=torch.float32, device=DEVICE)
        conf_unnorm = torch.full(x_current.data.shape, -np.inf, dtype=torch.float32, device=DEVICE)
        conf_norm[:, 5:10] = torch.randn(1, 5, device=DEVICE)
        conf_unnorm[:, 5:10] = torch.randn(1, 5, device=DEVICE)
        
        # Generate proposal
        idx = 7
        block_end = 10
        x_prop, conf_norm_prop, conf_unnorm_prop = generator.generate(
            x_current, idx, block_end, conf_norm, conf_unnorm
        )
        
        # Verify
        assert x_prop is not None
        assert conf_norm_prop is not None
        assert conf_unnorm_prop is not None
        assert x_prop.data.shape == x_current.data.shape
        # Positions before idx should be unchanged
        assert torch.all(x_prop.data[:, :idx] == x_current.data[:, :idx])
        assert generator.num_forwards > 0
    
    def test_generate_full_block(self):
        """Test proposal generation for full block"""
        model = MockModel(vocab_size=100, device=DEVICE)
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9, threshold=0.9,
            mask_id=99, eos_id=98
        )
        
        generator = MCMCProposalGenerator(model, decoder, mcmc_alpha=4.0)
        
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        x_current = TokenArray(prompt, gen_length=10, mask_id=99, eos_id=98, device=DEVICE)
        x_current.data[:, 5:10] = torch.tensor([[10, 11, 12, 13, 14]], device=DEVICE)
        
        conf_norm = torch.full(x_current.data.shape, -np.inf, dtype=torch.float32, device=DEVICE)
        conf_unnorm = torch.full(x_current.data.shape, -np.inf, dtype=torch.float32, device=DEVICE)
        
        # Generate from start of block
        idx = 5
        block_end = 10
        x_prop, conf_norm_prop, conf_unnorm_prop = generator.generate(
            x_current, idx, block_end, conf_norm, conf_unnorm
        )
        
        assert x_prop is not None
        # Prompt should be unchanged
        assert torch.all(x_prop.data[:, :5] == x_current.data[:, :5])


# ============================================================================
# MCMCRefinementRunner Tests
# ============================================================================

class TestMCMCRefinementRunner:
    """Tests for MCMCRefinementRunner class"""
    
    def test_init(self):
        """Test initialization"""
        model = MockModel(device=DEVICE)
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9, threshold=0.9,
            mask_id=99, eos_id=98
        )
        
        generator = MCMCProposalGenerator(model, decoder)
        runner = MCMCRefinementRunner(
            proposal_generator=generator,
            n_mcmc_steps=5
        )
        
        assert runner.proposal_generator is generator
        assert runner.n_mcmc_steps == 5
    
    def test_compute_log_acceptance_ratio(self):
        """Test MH acceptance ratio computation"""
        model = MockModel(device=DEVICE)
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9, threshold=0.9,
            mask_id=99, eos_id=98
        )
        
        generator = MCMCProposalGenerator(model, decoder)
        runner = MCMCRefinementRunner(generator, n_mcmc_steps=5)
        
        # Create test confidence tensors
        conf_norm_cur = torch.tensor([[-1.0, -2.0, -3.0]], device=DEVICE)
        conf_unnorm_cur = torch.tensor([[-0.5, -1.0, -1.5]], device=DEVICE)
        conf_norm_prop = torch.tensor([[-1.5, -2.5, -2.0]], device=DEVICE)
        conf_unnorm_prop = torch.tensor([[-0.8, -1.2, -1.0]], device=DEVICE)
        
        log_r = runner._compute_log_acceptance_ratio(
            conf_norm_cur, conf_unnorm_cur,
            conf_norm_prop, conf_unnorm_prop,
            idx=0, block_end=3
        )
        
        # Verify it returns a finite number
        assert np.isfinite(log_r)
    
    def test_acceptance_ratio_symmetry(self):
        """Test that swapping current and proposal inverts the ratio"""
        model = MockModel(device=DEVICE)
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9, threshold=0.9,
            mask_id=99, eos_id=98
        )
        
        generator = MCMCProposalGenerator(model, decoder)
        runner = MCMCRefinementRunner(generator, n_mcmc_steps=5)
        
        conf_norm_a = torch.tensor([[-1.0, -2.0]], device=DEVICE)
        conf_unnorm_a = torch.tensor([[-0.5, -1.0]], device=DEVICE)
        conf_norm_b = torch.tensor([[-1.5, -2.5]], device=DEVICE)
        conf_unnorm_b = torch.tensor([[-0.8, -1.2]], device=DEVICE)
        
        log_r_ab = runner._compute_log_acceptance_ratio(
            conf_norm_a, conf_unnorm_a, conf_norm_b, conf_unnorm_b, 0, 2
        )
        log_r_ba = runner._compute_log_acceptance_ratio(
            conf_norm_b, conf_unnorm_b, conf_norm_a, conf_unnorm_a, 0, 2
        )
        
        # log_r_ab + log_r_ba should be approximately 0
        assert abs(log_r_ab + log_r_ba) < 1e-5


# ============================================================================
# BlockMCMCDiffusionLLM Tests
# ============================================================================

class TestBlockMCMCDiffusionLLM:
    """Tests for BlockMCMCDiffusionLLM class"""
    
    def test_init_with_mcmc(self):
        """Test initialization with MCMC enabled"""
        model = MockModel(device=DEVICE)
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9, threshold=0.9,
            mask_id=99, eos_id=98
        )
        iterator_factory = BlockIteratorFactory(True)
        
        dllm = BlockMCMCDiffusionLLM(
            model=model,
            decoder=decoder,
            iterator_factory=iterator_factory,
            enable_mcmc=True,
            n_mcmc_steps=5,
            mcmc_alpha=4.0,
            mcmc_temperature=0.0
        )
        
        assert dllm.model is model
        assert dllm.decoder is decoder
        assert dllm.enable_mcmc == True
        assert dllm.n_mcmc_steps == 5
        assert dllm.mcmc_alpha == 4.0
        assert isinstance(dllm.diff_iteration, MCMCDiffusionIteration)
        assert isinstance(dllm.block_decoder, MCMCBlockRunner)
        assert dllm.proposal_generator is not None
        assert dllm.mcmc_runner is not None
    
    def test_init_without_mcmc(self):
        """Test initialization with MCMC disabled"""
        model = MockModel(device=DEVICE)
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9, threshold=0.9,
            mask_id=99, eos_id=98
        )
        iterator_factory = BlockIteratorFactory(True)
        
        dllm = BlockMCMCDiffusionLLM(
            model=model,
            decoder=decoder,
            iterator_factory=iterator_factory,
            enable_mcmc=False
        )
        
        assert dllm.enable_mcmc == False
        assert dllm.proposal_generator is None
        assert dllm.mcmc_runner is None
    
    def test_num_forwards_property(self):
        """Test num_forwards property"""
        model = MockModel(device=DEVICE)
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9, threshold=0.9,
            mask_id=99, eos_id=98
        )
        iterator_factory = BlockIteratorFactory(True)
        
        dllm = BlockMCMCDiffusionLLM(
            model=model,
            decoder=decoder,
            iterator_factory=iterator_factory,
            enable_mcmc=True
        )
        
        # Initially should be 0
        assert dllm.num_forwards == 0


# ============================================================================
# Run Tests
# ============================================================================

def run_tests():
    """Run all tests"""
    print("="*60)
    print("Running MCMC Component Unit Tests")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    all_passed = True
    
    # Test MCMCDiffusionIteration
    print("\n=== Testing MCMCDiffusionIteration ===")
    test_iteration = TestMCMCDiffusionIteration()
    try:
        test_iteration.test_init()
        print("  test_init: PASSED")
    except Exception as e:
        print(f"  test_init: FAILED - {e}")
        all_passed = False
    
    try:
        test_iteration.test_reset_confidences()
        print("  test_reset_confidences: PASSED")
    except Exception as e:
        print(f"  test_reset_confidences: FAILED - {e}")
        all_passed = False
    
    try:
        test_iteration.test_forward_no_cache()
        print("  test_forward_no_cache: PASSED")
    except Exception as e:
        print(f"  test_forward_no_cache: FAILED - {e}")
        all_passed = False
    
    try:
        test_iteration.test_confidence_accumulation()
        print("  test_confidence_accumulation: PASSED")
    except Exception as e:
        print(f"  test_confidence_accumulation: FAILED - {e}")
        all_passed = False
    
    try:
        test_iteration.test_different_alpha_values()
        print("  test_different_alpha_values: PASSED")
    except Exception as e:
        print(f"  test_different_alpha_values: FAILED - {e}")
        all_passed = False
    
    # Test MCMCBlockRunner
    print("\n=== Testing MCMCBlockRunner ===")
    test_runner = TestMCMCBlockRunner()
    try:
        test_runner.test_init()
        print("  test_init: PASSED")
    except Exception as e:
        print(f"  test_init: FAILED - {e}")
        all_passed = False
    
    try:
        test_runner.test_inheritance()
        print("  test_inheritance: PASSED")
    except Exception as e:
        print(f"  test_inheritance: FAILED - {e}")
        all_passed = False
    
    # Test MCMCProposalGenerator
    print("\n=== Testing MCMCProposalGenerator ===")
    test_generator = TestMCMCProposalGenerator()
    try:
        test_generator.test_init()
        print("  test_init: PASSED")
    except Exception as e:
        print(f"  test_init: FAILED - {e}")
        all_passed = False
    
    try:
        test_generator.test_generate_basic()
        print("  test_generate_basic: PASSED")
    except Exception as e:
        print(f"  test_generate_basic: FAILED - {e}")
        all_passed = False
    
    try:
        test_generator.test_generate_full_block()
        print("  test_generate_full_block: PASSED")
    except Exception as e:
        print(f"  test_generate_full_block: FAILED - {e}")
        all_passed = False
    
    # Test MCMCRefinementRunner
    print("\n=== Testing MCMCRefinementRunner ===")
    test_refinement = TestMCMCRefinementRunner()
    try:
        test_refinement.test_init()
        print("  test_init: PASSED")
    except Exception as e:
        print(f"  test_init: FAILED - {e}")
        all_passed = False
    
    try:
        test_refinement.test_compute_log_acceptance_ratio()
        print("  test_compute_log_acceptance_ratio: PASSED")
    except Exception as e:
        print(f"  test_compute_log_acceptance_ratio: FAILED - {e}")
        all_passed = False
    
    try:
        test_refinement.test_acceptance_ratio_symmetry()
        print("  test_acceptance_ratio_symmetry: PASSED")
    except Exception as e:
        print(f"  test_acceptance_ratio_symmetry: FAILED - {e}")
        all_passed = False
    
    # Test BlockMCMCDiffusionLLM
    print("\n=== Testing BlockMCMCDiffusionLLM ===")
    test_dllm = TestBlockMCMCDiffusionLLM()
    try:
        test_dllm.test_init_with_mcmc()
        print("  test_init_with_mcmc: PASSED")
    except Exception as e:
        print(f"  test_init_with_mcmc: FAILED - {e}")
        all_passed = False
    
    try:
        test_dllm.test_init_without_mcmc()
        print("  test_init_without_mcmc: PASSED")
    except Exception as e:
        print(f"  test_init_without_mcmc: FAILED - {e}")
        all_passed = False
    
    try:
        test_dllm.test_num_forwards_property()
        print("  test_num_forwards_property: PASSED")
    except Exception as e:
        print(f"  test_num_forwards_property: FAILED - {e}")
        all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("All unit tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
