"""
Unit tests for MCMC components in dInfer framework.

Tests:
- MCMCDiffusionIteration
- MCMCBlockRunner
- MCMCProposalGenerator
- MCMCRefinementRunner
- BlockMCMCDiffusionLLM
- Compatibility with BlockWiseDiffusionLLM

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
    BlockWiseDiffusionLLM,
    BaseDiffusionIteration,
    ShiftDiffusionIteration,
    BlockRunner,
)
from dinfer.decoding.parallel_strategy import (
    MCMCThresholdParallelDecoder,
    ThresholdParallelDecoder,
)

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
    
    def test_init_with_proposal_alpha(self):
        """Test initialization with proposal_alpha parameter"""
        model = MockModel(device=DEVICE)
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9, threshold=0.9,
            mask_id=99, eos_id=98
        )
        
        generator = MCMCProposalGenerator(
            model=model,
            decoder=decoder,
            mcmc_alpha=4.0,
            proposal_alpha=2.0
        )
        
        assert generator.proposal_alpha == 2.0
    
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
        result = generator.generate(
            x_current, idx, block_end, conf_norm, conf_unnorm
        )
        
        # Verify - should return 5 values now (including reverse confidences)
        assert len(result) == 5
        x_prop, conf_norm_prop, conf_unnorm_prop, reverse_conf_norm, reverse_conf_unnorm = result
        
        assert x_prop is not None
        assert conf_norm_prop is not None
        assert conf_unnorm_prop is not None
        assert reverse_conf_norm is not None
        assert reverse_conf_unnorm is not None
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
        result = generator.generate(
            x_current, idx, block_end, conf_norm, conf_unnorm
        )
        
        x_prop = result[0]
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
        target_unnorm_cur = torch.tensor([[-0.5, -1.0, -1.5]], device=DEVICE)
        target_unnorm_prop = torch.tensor([[-0.8, -1.2, -1.0]], device=DEVICE)
        proposal_forward = torch.tensor([[-1.0, -2.0, -3.0]], device=DEVICE)
        proposal_reverse = torch.tensor([[-1.5, -2.5, -2.0]], device=DEVICE)
        
        log_r = runner._compute_log_acceptance_ratio(
            target_unnorm_cur, target_unnorm_prop,
            proposal_forward, proposal_reverse,
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
        
        # Create symmetric test case
        target_a = torch.tensor([[-0.5, -1.0]], device=DEVICE)
        target_b = torch.tensor([[-0.8, -1.2]], device=DEVICE)
        q_forward = torch.tensor([[-1.0, -2.0]], device=DEVICE)
        q_reverse = torch.tensor([[-1.5, -2.5]], device=DEVICE)
        
        # log r(a->b) = log p(b) + log q(a|b) - log p(a) - log q(b|a)
        log_r_ab = runner._compute_log_acceptance_ratio(
            target_a, target_b, q_forward, q_reverse, 0, 2
        )
        # log r(b->a) = log p(a) + log q(b|a) - log p(b) - log q(a|b)
        log_r_ba = runner._compute_log_acceptance_ratio(
            target_b, target_a, q_reverse, q_forward, 0, 2
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
        # Should use BaseDiffusionIteration when enable_mcmc=False and use_shift=False
        assert isinstance(dllm.diff_iteration, BaseDiffusionIteration)
        assert isinstance(dllm.block_decoder, BlockRunner)

    def test_init_with_shift_mode(self):
        """Test initialization with shift mode (enable_mcmc=False, use_shift=True)"""
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
            enable_mcmc=False,
            use_shift=True
        )
        
        assert dllm.enable_mcmc == False
        assert dllm.use_shift == True
        # Should use ShiftDiffusionIteration when enable_mcmc=False and use_shift=True
        assert isinstance(dllm.diff_iteration, ShiftDiffusionIteration)
        assert isinstance(dllm.block_decoder, BlockRunner)
    
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
    
    def test_proposal_alpha_parameter(self):
        """Test proposal_alpha parameter is correctly passed"""
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
            proposal_alpha=2.0
        )
        
        assert dllm.proposal_alpha == 2.0
        assert dllm.proposal_generator.proposal_alpha == 2.0


# ============================================================================
# Compatibility Tests
# ============================================================================

class TestBlockMCMCCompatibility:
    """Tests for BlockMCMCDiffusionLLM compatibility with BlockWiseDiffusionLLM"""
    
    def test_degraded_mode_uses_same_components(self):
        """Test that degraded mode uses same components as BlockWiseDiffusionLLM"""
        model = MockModel(device=DEVICE)
        decoder = ThresholdParallelDecoder(
            temperature=0.9, threshold=0.9,
            mask_id=99, eos_id=98
        )
        iterator_factory = BlockIteratorFactory(True)
        
        # Create BlockMCMCDiffusionLLM in degraded mode
        mcmc_dllm = BlockMCMCDiffusionLLM(
            model=model,
            decoder=decoder,
            iterator_factory=iterator_factory,
            enable_mcmc=False,
            use_shift=False
        )
        
        # Create BlockWiseDiffusionLLM
        blockwise_dllm = BlockWiseDiffusionLLM(
            model=model,
            decoder=decoder,
            iterator_factory=iterator_factory,
            use_shift=False
        )
        
        # Both should use BaseDiffusionIteration
        assert type(mcmc_dllm.diff_iteration) == type(blockwise_dllm.diff_iteration)
        # Both should use BlockRunner
        assert type(mcmc_dllm.block_decoder) == type(blockwise_dllm.block_decoder)

    def test_degraded_mode_with_shift_uses_same_components(self):
        """Test that degraded mode with shift uses same components as BlockWiseDiffusionLLM with shift"""
        model = MockModel(device=DEVICE)
        decoder = ThresholdParallelDecoder(
            temperature=0.9, threshold=0.9,
            mask_id=99, eos_id=98
        )
        iterator_factory = BlockIteratorFactory(True)
        
        # Create BlockMCMCDiffusionLLM in degraded mode with shift
        mcmc_dllm = BlockMCMCDiffusionLLM(
            model=model,
            decoder=decoder,
            iterator_factory=iterator_factory,
            enable_mcmc=False,
            use_shift=True
        )
        
        # Create BlockWiseDiffusionLLM with shift
        blockwise_dllm = BlockWiseDiffusionLLM(
            model=model,
            decoder=decoder,
            iterator_factory=iterator_factory,
            use_shift=True
        )
        
        # Both should use ShiftDiffusionIteration
        assert type(mcmc_dllm.diff_iteration) == type(blockwise_dllm.diff_iteration)
        # Both should use BlockRunner
        assert type(mcmc_dllm.block_decoder) == type(blockwise_dllm.block_decoder)
    
    def test_mcmc_mode_uses_mcmc_components(self):
        """Test that MCMC mode uses MCMC-specific components"""
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
        
        # Should use MCMC-specific components
        assert isinstance(dllm.diff_iteration, MCMCDiffusionIteration)
        assert isinstance(dllm.block_decoder, MCMCBlockRunner)
        assert isinstance(dllm.proposal_generator, MCMCProposalGenerator)
        assert isinstance(dllm.mcmc_runner, MCMCRefinementRunner)
    
    def test_inheritance_from_blockwise(self):
        """Test that BlockMCMCDiffusionLLM inherits from BlockWiseDiffusionLLM"""
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
        
        assert isinstance(dllm, BlockWiseDiffusionLLM)


# ============================================================================
# MCMCThresholdParallelDecoder Tests
# ============================================================================

class TestMCMCThresholdParallelDecoder:
    """Tests for MCMCThresholdParallelDecoder class"""
    
    def test_init(self):
        """Test initialization"""
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        assert decoder.temperature == 0.9
        assert decoder.threshold == 0.9
        assert decoder.mask_id == 99
        assert decoder.eos_id == 98
    
    def test_inheritance(self):
        """Test that MCMCThresholdParallelDecoder inherits from ThresholdParallelDecoder"""
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        assert isinstance(decoder, ThresholdParallelDecoder)

    def test_decode_returns_confidences(self):
        """Test that decode returns confidence tensors"""
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        # Create test data
        batch_size, seq_len, vocab_size = 1, 10, 100
        logits = torch.randn(batch_size, 5, vocab_size, device=DEVICE)
        
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        x = TokenArray(prompt, gen_length=5, mask_id=99, eos_id=98, device=DEVICE)
        
        # Decode
        conf_norm, conf_unnorm = decoder.decode(
            logits, block_start=5, block_end=10, x=x, mcmc_alpha=4.0
        )
        
        assert conf_norm is not None
        assert conf_unnorm is not None
        assert conf_norm.shape == (1, 5)
        assert conf_unnorm.shape == (1, 5)
    
    def test_decode_with_proposal_alpha(self):
        """Test decode with proposal_alpha parameter"""
        decoder = MCMCThresholdParallelDecoder(
            temperature=0.9,
            threshold=0.9,
            mask_id=99,
            eos_id=98
        )
        
        batch_size, seq_len, vocab_size = 1, 10, 100
        logits = torch.randn(batch_size, 5, vocab_size, device=DEVICE)
        
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        x = TokenArray(prompt, gen_length=5, mask_id=99, eos_id=98, device=DEVICE)
        
        # Decode with proposal_alpha > 1.0
        conf_norm, conf_unnorm = decoder.decode(
            logits, block_start=5, block_end=10, x=x, 
            mcmc_alpha=4.0, proposal_alpha=2.0
        )
        
        assert conf_norm is not None
        assert conf_unnorm is not None


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
    for test_name in ['test_init', 'test_reset_confidences', 'test_forward_no_cache', 
                      'test_confidence_accumulation', 'test_different_alpha_values']:
        try:
            getattr(test_iteration, test_name)()
            print(f"  {test_name}: PASSED")
        except Exception as e:
            print(f"  {test_name}: FAILED - {e}")
            all_passed = False
    
    # Test MCMCBlockRunner
    print("\n=== Testing MCMCBlockRunner ===")
    test_runner = TestMCMCBlockRunner()
    for test_name in ['test_init', 'test_inheritance']:
        try:
            getattr(test_runner, test_name)()
            print(f"  {test_name}: PASSED")
        except Exception as e:
            print(f"  {test_name}: FAILED - {e}")
            all_passed = False
    
    # Test MCMCProposalGenerator
    print("\n=== Testing MCMCProposalGenerator ===")
    test_generator = TestMCMCProposalGenerator()
    for test_name in ['test_init', 'test_init_with_proposal_alpha', 
                      'test_generate_basic', 'test_generate_full_block']:
        try:
            getattr(test_generator, test_name)()
            print(f"  {test_name}: PASSED")
        except Exception as e:
            print(f"  {test_name}: FAILED - {e}")
            all_passed = False

    # Test MCMCRefinementRunner
    print("\n=== Testing MCMCRefinementRunner ===")
    test_refinement = TestMCMCRefinementRunner()
    for test_name in ['test_init', 'test_compute_log_acceptance_ratio', 
                      'test_acceptance_ratio_symmetry']:
        try:
            getattr(test_refinement, test_name)()
            print(f"  {test_name}: PASSED")
        except Exception as e:
            print(f"  {test_name}: FAILED - {e}")
            all_passed = False
    
    # Test BlockMCMCDiffusionLLM
    print("\n=== Testing BlockMCMCDiffusionLLM ===")
    test_dllm = TestBlockMCMCDiffusionLLM()
    for test_name in ['test_init_with_mcmc', 'test_init_without_mcmc', 
                      'test_init_with_shift_mode', 'test_num_forwards_property',
                      'test_proposal_alpha_parameter']:
        try:
            getattr(test_dllm, test_name)()
            print(f"  {test_name}: PASSED")
        except Exception as e:
            print(f"  {test_name}: FAILED - {e}")
            all_passed = False
    
    # Test Compatibility
    print("\n=== Testing BlockMCMC Compatibility ===")
    test_compat = TestBlockMCMCCompatibility()
    for test_name in ['test_degraded_mode_uses_same_components', 
                      'test_degraded_mode_with_shift_uses_same_components',
                      'test_mcmc_mode_uses_mcmc_components',
                      'test_inheritance_from_blockwise']:
        try:
            getattr(test_compat, test_name)()
            print(f"  {test_name}: PASSED")
        except Exception as e:
            print(f"  {test_name}: FAILED - {e}")
            all_passed = False
    
    # Test MCMCThresholdParallelDecoder
    print("\n=== Testing MCMCThresholdParallelDecoder ===")
    test_decoder = TestMCMCThresholdParallelDecoder()
    for test_name in ['test_init', 'test_inheritance', 
                      'test_decode_returns_confidences', 'test_decode_with_proposal_alpha']:
        try:
            getattr(test_decoder, test_name)()
            print(f"  {test_name}: PASSED")
        except Exception as e:
            print(f"  {test_name}: FAILED - {e}")
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
