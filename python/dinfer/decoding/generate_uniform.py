import torch
import numpy as np
import logging
import random

from transformers.models.layoutlmv2.modeling_layoutlmv2 import relative_position_bucket

from .utils import TokenArray, DistAlignedTokenArray, gather_sequence_block
from .utils import calculate_op_num, add_gumbel_noise_power, get_num_transfer_tokens

logger = logging.getLogger(__name__)

class DiffusionLLM:
    """ Diffusion LLM inference
    """

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations.

        Parameters:
        ----------
        prompt: Torch.Tensor
            A tensor of shape (1, L) that contains the input prompt.
        gen_length: int
            Generated answer length.
        block_length: int
            Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.

        Returns
        -------
        Torch.Tensor: A tensor of shape (1, L') that contains the prompt tokens and the generated tokens.
            EOS and any tokens after EOS have been removed.
        '''

def select_undecoded(seq_idx, orig_x, x, block, block_loc, mask_id, writeback=False):
    """ 选择未完成解码的序列。

    在批量解码过程中，某些序列可能比其他序列先完成解码（即不再包含mask token）。
    此函数用于筛选出仍需解码的序列，并可选地将已完成解码的序列写回原始token数组。
    """
    if x.batch_size == 1:
        return seq_idx, x
    bool_idx = torch.all(block != mask_id, dim=1)

    if writeback:
        # Write the decoded tokens back
        finished_idx = seq_idx[bool_idx]
        orig_x[finished_idx, block_loc.start:block_loc.end] = block[bool_idx]

    # Select the undecoded sequences
    return seq_idx, x

class BlockRunner:
    """ The class decodes all tokens in a block

    Parameters
    ----------
    diff_iteration : DiffusionIteration
        Run forward computation on a block to decode tokens
    early_stop : bool
        Whether or not to have early stop
    maximum_unroll : int
        The max number of iterations to unroll
    expected_tpf : int
        The expected TPF for loop unrolling.
    """
    def __init__(self, diff_iteration, early_stop, maximum_unroll, expected_tpf):
        self.diff_iteration = diff_iteration
        self.early_stop = early_stop
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf

    def decode(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode all tokens in a block.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input tokens in the block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID

        Returns
        -------
        torch.Tensor : a bool tensor that indicates whether the sequences have finished decoding.
        """
        orig_x = x
        seq_idx = torch.arange(x.batch_size, device=block.device)
        # 初始筛选未解码序列
        seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=False)
        block = x[:, block_loc.start:block_loc.end]
        batch_size = x.batch_size
        while (block == decoder.mask_id).sum() > 0:
            # 限制forward的次数，计算最大的forward次数
            unroll_k = int(max(min((block == decoder.mask_id).sum()//self.expected_tpf, self.maximum_unroll), 1))
            for unroll_i in range(unroll_k):
                self.diff_iteration.forward(model, decoder, x, kv_cache, block, block_loc, block_id)

            # If there are more than one sequence, we should filter the sequences and only decode
            # on the sequences that still have masked tokens.
            if batch_size > 1:
                seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=True)
                block = x[:, block_loc.start:block_loc.end]
                # If all blocks have been decoded, we can jumpt out.
                if len(seq_idx) == 0:
                    break
            batch_size = x.batch_size

        eos_idx = torch.any(orig_x[:, block_loc.start:block_loc.end] == decoder.eos_id, dim=1)
        if self.early_stop:
            # Find the first location of EOS and set all tokens after the location to EOS.
            # Here we assume that don't perform remasking.
            orig_x[eos_idx, block_loc.end:] = decoder.eos_id
        return eos_idx

class BlockDiffusionRunner(BlockRunner):
    """ The class decodes all tokens in a block

    Parameters
    ----------
    diff_iteration : BlockDiffusionIteration
        Run forward computation on a block to decode tokens
    early_stop : bool
        Whether or not to have early stop
    maximum_unroll : int
        The max number of iterations to unroll
    expected_tpf : int
        The expected TPF for loop unrolling.
    """
    def __init__(self, diff_iteration, early_stop, maximum_unroll, expected_tpf, backend):
        super().__init__(diff_iteration, early_stop, maximum_unroll, expected_tpf)
        self.backend = backend

    def prefill(self, model, block, kv_cache, pos_ids, attn_mask):
        """ Prefill for KV Cache
        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        block : torch.Tensor
            The input IDs of the tokens in the prefilling range.
        kv_cache: KVCache
            The KV-cache
        pos_ids: torch.Tensor
            The position IDs of the tokens in the prefilling range.
        attn_mask: torch.Tensor
            The attention mask of the tokens in the prefilling range.
        """
        if kv_cache is None:
            return
        else:
            # 执行一次前向传播以获取 KV cache
            output = model(block.clone(memory_format=torch.contiguous_format), use_cache=True, attention_mask=attn_mask, position_ids=pos_ids.clone(memory_format=torch.contiguous_format))
            if self.backend == 'vllm':
                kv_cache.update(output.past_key_values)
            else:
                kv_cache.range_update(output.past_key_values, 0, block.size(1), 0)
            self.diff_iteration.num_forwards +=1
            self.diff_iteration.iter_no +=1

    def decode(self, model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask):
        """ Decode all tokens in a block.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        pos_ids: torch.Tensor
            The position IDs of all the tokens.
        attn_mask: torch.Tensor
            The attention mask of all the tokens. 
        Returns
        -------
        torch.Tensor : a bool tensor that indicates whether the sequences have finished decoding.
        """
        orig_x = x
        seq_idx = torch.arange(x.batch_size, device=block.device)
        seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=False)
        block = x[:, block_loc.start:block_loc.end]
        batch_size = x.batch_size

        # 准备 KV cache
        if kv_cache is not None:
            kv_cache.extend_cache(block_loc.end)
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
        else:
            past_key_values, replace_position = None, None

        input_block_mask_number = 0
        while (block == decoder.mask_id).sum() > 0:
            unroll_k = int(max(min((block == decoder.mask_id).sum()//self.expected_tpf, self.maximum_unroll), 2))
            for unroll_i in range(unroll_k):
                input_block_mask_number = (block == decoder.mask_id).sum()
                output = self.diff_iteration.forward(model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, past_key_values, replace_position, self.backend)
            if batch_size > 1:
                seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=True)
                block = x[:, block_loc.start:block_loc.end]
                # If all blocks have been decoded, we can jumpt out.
                if len(seq_idx) == 0:
                    break
        # additional forward to update kvcache for the last decoding step in the current block
        # 额外的一次前向传播，用于更新当前块最后一步解码的 KV cache
        if kv_cache is not None:
            if input_block_mask_number > 0:
                output = model(block.clone(memory_format=torch.contiguous_format), 
                    past_key_values=past_key_values,
                    use_cache=True, 
                    position_ids=pos_ids[:, block_loc.start:block_loc.end].clone(memory_format=torch.contiguous_format),
                    replace_position=(0,0) if self.backend=='sglang' else replace_position)
                self.diff_iteration.num_forwards +=1
                self.diff_iteration.iter_no +=1
            if self.backend=='vllm':
                kv_cache.update(output.past_key_values)
            else:
                kv_cache.range_update(output.past_key_values, 0, block_loc.end, block_loc.end - block_loc.start)



        eos_idx = torch.any(orig_x[:, block_loc.start:block_loc.end] == decoder.eos_id, dim=1)
        if self.early_stop:
            orig_x[eos_idx, block_loc.end:] = decoder.eos_id
        return eos_idx

class DiffusionIteration:
    """ A diffusion iteration to decode tokens
    """
    def __init__(self):
        self.num_forwards = 0
        self.cache_updates = 0

    def forward(self, model, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        pass

class BaseDiffusionIteration(DiffusionIteration):
    """ A base implementation of diffusion iteration to decode.
    """
    def __init__(self):
        super().__init__()
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode tokens in a forward run on a block.

        The forward run decodes tokens in the input array.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        """
        cache_update_kv = None
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_loc.start, block_loc.end):
            output = model(x.data, use_cache=True)
            cache_update_kv = output.past_key_values
            self.num_forwards += 1
            # use the generated output to decode.
            decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x)
            # update KV-cache
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1

        # 根据 KV cache 的状态和类型执行不同的前向传播逻辑
        if kv_cache is None:
            # 如果没有 KV cache，直接对整个输入进行前向传播
            logits = model(x.data).logits[:, block_loc.start:block_loc.end]
        elif kv_cache.cache_type == 'prefix':
            # 如果是前缀缓存，获取对应的 KV cache 和替换位置
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            # 仅对当前块及其后续部分进行前向传播
            logits = model(x[:, block_loc.start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_loc.end - block_loc.start
            logits = logits[:, :block_length]
        else:
            # 其他类型的缓存（如双向缓存）
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            # cache position is the position between current_block_start and current_block_end
            logits = model(block, past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits

        decoder.decode(logits, block_loc.start, block_loc.end, x)
        self.num_forwards += 1
        self.iter_no += 1
        return cache_update_kv, logits


class MCMCDiffusionIteration(DiffusionIteration):
    """
    MCMC-aware diffusion iteration that tracks dual confidences
    for Metropolis-Hastings acceptance ratio computation.

    This iteration class works with MCMCThresholdParallelDecoder to track
    both normalized (α=1) and unnormalized (α=mcmc_alpha) log probabilities
    for each decoded token, enabling Power Sampling MCMC refinement.

    Parameters
    ----------
    mcmc_alpha : float
        Power parameter α for target distribution p^α (default: 4.0)
    mcmc_temperature : float
        Temperature for Gumbel noise sampling (default: 0.0)
    """

    def __init__(self, mcmc_alpha=4.0, mcmc_temperature=0.0):
        super().__init__()
        self.mcmc_alpha = mcmc_alpha
        self.mcmc_temperature = mcmc_temperature
        self.iter_no = 0

        # 置信度追踪 (全局，跨迭代累积)
        self.confidences_norm = None      # log p(x), α=1.0
        self.confidences_unnorm = None    # log p^α(x), α=mcmc_alpha

    def reset_confidences(self, shape, device):
        """Reset confidence tensors (called at the beginning of generate())

        Parameters
        ----------
        shape : tuple
            Shape of the token array (batch_size, seq_len)
        device : torch.device
            Device to place tensors on
        """
        self.confidences_norm = torch.full(shape, -np.inf, dtype=torch.float32, device=device)
        self.confidences_unnorm = torch.full(shape, -np.inf, dtype=torch.float32, device=device)

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """
        Single diffusion iteration with dual confidence tracking.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : MCMCThresholdParallelDecoder
            The decoder (must be MCMCThresholdParallelDecoder for confidence tracking)
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache : KVCache or None
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID

        Returns
        -------
        tuple : (conf_norm_block, conf_unnorm_block)
            Confidence tensors for the current block
        """
        from .parallel_strategy import MCMCThresholdParallelDecoder

        block_length = block_loc.end - block_loc.start

        # Step 1: Handle KV Cache update if needed
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_loc.start, block_loc.end):
            output = model(x.data, use_cache=True)
            self.num_forwards += 1
            logits = output.logits[:, block_loc.start:block_loc.end]
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1

            # Decode with confidence tracking
            if isinstance(decoder, MCMCThresholdParallelDecoder):
                conf_norm_block, conf_unnorm_block = decoder.decode(
                    logits, block_loc.start, block_loc.end, x,
                    mcmc_alpha=self.mcmc_alpha
                )
            else:
                decoder.decode(logits, block_loc.start, block_loc.end, x)
                conf_norm_block, conf_unnorm_block = None, None

            # Update global confidences
            if conf_norm_block is not None:
                self._update_global_confidences(conf_norm_block, conf_unnorm_block, block_loc)

            self.iter_no += 1
            return conf_norm_block, conf_unnorm_block

        # Step 2: Forward pass based on KV cache type
        if kv_cache is None:
            logits = model(x.data).logits[:, block_loc.start:block_loc.end]
        elif kv_cache.cache_type == 'prefix':
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            output = model(x[:, block_loc.start:], past_key_values=past_key_values, use_cache=True,
                          replace_position=replace_position)
            logits = output.logits[:, :block_length]
        else:  # dual cache
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            output = model(block, past_key_values=past_key_values, use_cache=True,
                          replace_position=replace_position)
            logits = output.logits

        # Step 3: Decode with confidence tracking
        if isinstance(decoder, MCMCThresholdParallelDecoder):
            conf_norm_block, conf_unnorm_block = decoder.decode(
                logits, block_loc.start, block_loc.end, x,
                mcmc_alpha=self.mcmc_alpha
            )
        else:
            # Fallback for non-MCMC decoders
            decoder.decode(logits, block_loc.start, block_loc.end, x)
            conf_norm_block, conf_unnorm_block = None, None

        # Step 4: Update global confidences
        if conf_norm_block is not None:
            self._update_global_confidences(conf_norm_block, conf_unnorm_block, block_loc)

        self.num_forwards += 1
        self.iter_no += 1

        return conf_norm_block, conf_unnorm_block

    # 调试开关
    DEBUG_MCMC_ITERATION = False
    
    def _update_global_confidences(self, conf_norm_block, conf_unnorm_block, block_loc):
        """Update global confidence tensors with block-level confidences."""
        if self.confidences_norm is None:
            return

        mask_updated = conf_norm_block > -np.inf
        num_to_update = mask_updated.sum().item()
        
        # DEBUG: 记录更新前的状态
        block_region = self.confidences_norm[:, block_loc.start:block_loc.end]
        num_inf_before = (block_region == -np.inf).sum().item()
        
        if mask_updated.any():
            self.confidences_norm[:, block_loc.start:block_loc.end][mask_updated] = \
                conf_norm_block[mask_updated]
            self.confidences_unnorm[:, block_loc.start:block_loc.end][mask_updated] = \
                conf_unnorm_block[mask_updated]
        
        # DEBUG: 记录更新后的状态
        block_region_after = self.confidences_norm[:, block_loc.start:block_loc.end]
        num_inf_after = (block_region_after == -np.inf).sum().item()
        
        if self.DEBUG_MCMC_ITERATION:
            print(f"[MCMCIteration._update] block=[{block_loc.start},{block_loc.end}), "
                  f"to_update={num_to_update}, inf_before={num_inf_before}, inf_after={num_inf_after}")


class BlockDiffusionIteration:
    """ An implementation of block diffusion iteration to decode.
    """
    def __init__(self):
        self.num_forwards = 0
        self.cache_updates = 0
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, past_key_values, replace_position, backend):
        """ Decode tokens in a forward run on a block.

        The forward run decodes tokens in the input array.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        pos_ids: torch.Tensor
            The position IDs of all the tokens.
        attn_mask: torch.Tensor
            The attention mask of all the tokens. 
        past_key_values: List[List[torch.Tensor]]
            The key-values required to decode the specified block.
        replace_position: torch.Tensor 
            The tensor indicates the valid locations in the returned key-values.
        """
        if kv_cache is None:
            output = model(x.data[:, :block_loc.end], 
                attention_mask=attn_mask[:,:block_loc.end,:block_loc.end],
                position_ids=pos_ids[:, :block_loc.end])
            logits = output.logits[:, block_loc.start:block_loc.end]
        else:
            output = model(block.clone(memory_format=torch.contiguous_format),
                position_ids=pos_ids[:,block_loc.start:block_loc.end].clone(memory_format=torch.contiguous_format),
                use_cache=True,
                past_key_values=past_key_values,
                replace_position=(0,0) if backend=='sglang' else replace_position)
            logits = output.logits
            # TODO(dulun): we don't need update kv cache for every step.
            if backend == 'vllm':
                kv_cache.update(output.past_key_values)
            
        decoder.decode(logits, block_loc.start, block_loc.end, x)
        self.num_forwards += 1
        self.iter_no += 1
        return output


class ShiftDiffusionIteration(DiffusionIteration):
    """ A shift implementation of diffusion iteration to decode.
    """
    def __init__(self, use_shift = False):
        super().__init__()
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode tokens in a forward run on a block.

        The forward run decodes tokens in the input array.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        """
        # 计算移位后的块起始和结束位置
        block_start, block_end = block_loc.start-1, block_loc.end-1
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_start, block_end):
            output = model(x.data, use_cache=True)
            self.num_forwards += 1
            # use the generated output to decode.
            # TODO(dulun): need to improve efficiency
            # 创建移位后的 TokenArray
            x_shifted = TokenArray(x.data[:, 1:], 0, decoder.mask_id, decoder.eos_id, model.device)
            # 使用生成的 logits 解码移位后的 TokenArray
            decoder.decode(output.logits[:, block_start:block_end], block_start, block_end, x_shifted)
            # 将解码结果写回原始 TokenArray
            x.data[:, 1:] = x_shifted.data
            # update KV-cache
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1

        # 根据 KV cache 的状态和类型执行不同的前向传播逻辑
        if kv_cache is None:
            logits = model(x.data).logits[:, block_start:block_end]
        elif kv_cache.cache_type == 'prefix':
            past_key_values, replace_position = kv_cache.get_key_values(block_start, block_end)
            logits = model(x[:, block_start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_end - block_start
            logits = logits[:, :block_length]
        else:
            # cache position is the position between current_block_start and current_block_end
            past_key_values, replace_position = kv_cache.get_key_values(block_start, block_end)
            logits = model(x[:, block_start:block_end], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
        # TODO(dulun): need to improve efficiency
        # 再次创建移位后的 TokenArray 并解码
        x_shifted = TokenArray(x.data[:, 1:], 0, decoder.mask_id, decoder.eos_id, model.device)
        decoder.decode(logits, block_start, block_end, x_shifted)
        x.data[:, 1:] = x_shifted.data
        self.num_forwards += 1
        self.iter_no += 1

class BlockWiseDiffusionLLM(DiffusionLLM):
    """ Diffusion LLM inference

    This diffusion LLM inference generates tokens block by block.

    The decoding algorithm break the generation sequence into blocks.
    It runs diffusion iterations on the first block and decodes all tokens
    in the block before moving to the next block.
    This is a classifical dLLM decoding algorithm.

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    cache_factory : KVCacheFactory (optional)
        The KV-cache factory that generates a kv-cache for LLM.
    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8, use_shift=False):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        # 根据是否使用移位选择不同的扩散迭代实现
        if use_shift:
            self.diff_iteration = ShiftDiffusionIteration()
        else:
            self.diff_iteration = BaseDiffusionIteration()
        # 初始化块解码器
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)
        

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        # 初始化 TokenArray
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        # 创建迭代器
        it = self.iterator_factory.create(x, block_length)

        # We need to reset iter_no at the beginning of generating a sequence.
        # 重置迭代计数器
        self.diff_iteration.iter_no = 0
        # 创建 KV cache
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        # 遍历每个块进行解码
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            # 解码当前块
            decode_compl = self.block_decoder.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id)
            # If all sequences have EOS, we have finished decoding.
            # 如果所有序列都已完成解码，退出循环
            if torch.all(decode_compl):
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

class IterationSmooth(DiffusionIteration):
    """ A diffusion iteration to decode tokens
    """
    def __init__(self, model, cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        super().__init__()
        self.cont_weight = cont_weight
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            self.h2e = model.module.h2e
        else:
            self.h2e = model.h2e
        self.cont_weight_init = cont_weight_init
        self.cont_weight_growth = cont_weight_growth
        self.threshold_decay = threshold_decay
        self.inputs_embeds = None
        self.iter_no = 0

    def reset_input_embeds(self, x):
        """ Reset input embedding with new input sequence
        """
        self.inputs_embeds = self.h2e(x.data)

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_loc.start, block_loc.end):
            output = model(inputs_embeds=self.inputs_embeds, use_cache=True)
            self.num_forwards += 1
            # use the generated output to decode.
            decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x, iter_threshold)
            # update KV-cache
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, output.logits, iter_cont_weight)
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1
            self.iter_no += 1

        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        if kv_cache is None:
            logits = model(inputs_embeds=self.inputs_embeds).logits
            decoder.decode(logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x, iter_threshold)
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, logits, iter_cont_weight)
        elif kv_cache.cache_type == 'prefix':
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            logits = model(inputs_embeds=self.inputs_embeds[:, block_loc.start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_loc.end - block_loc.start
            decoder.decode(logits[:, :block_length], block_loc.start, block_loc.end, x, iter_threshold)
            mask_index = (x.data[:, block_loc.start:] == decoder.mask_id)
            self.inputs_embeds[:, block_loc.start:] = self.h2e(x.data[:, block_loc.start:], mask_index, logits, iter_cont_weight)
        else:
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            # cache position is the position between current_block_start and current_block_end
            logits = model(inputs_embeds=self.inputs_embeds[:, block_loc.start:block_loc.end], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            decoder.decode(logits, block_loc.start, block_loc.end, x, iter_threshold)
            mask_index = (x.data[:, block_loc.start:block_loc.end] == decoder.mask_id)
            self.inputs_embeds[:, block_loc.start:block_loc.end] = self.h2e(x.data[:, block_loc.start:block_loc.end], mask_index, logits, iter_cont_weight)
        self.num_forwards += 1
        self.iter_no += 1

class IterSmoothDiffusionLLM(BlockWiseDiffusionLLM):
    """ This diffusion LLM inference generates tokens block by block.

    The decoding algorithm break the generation sequence into blocks.
    It runs diffusion iterations on the first block and decodes all tokens
    in the block before moving to the next block.
    This is a classifical dLLM decoding algorithm.
    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8,
                cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.early_stop = early_stop
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf
        self.diff_iteration = IterationSmooth(self.model, cont_weight, cont_weight_init, cont_weight_growth, threshold_decay)
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates
    
    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        self.diff_iteration.reset_input_embeds(x)
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            decode_compl = self.block_decoder.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id)
            # If all sequences have EOS, we have finished decoding.
            if torch.all(decode_compl):
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

class VicinityCacheIteration(DiffusionIteration):
    """ A diffusion iteration to decode tokens
    """
    def __init__(self, prefix_look, after_look, warmup_steps):
        super().__init__()
        self.prefix_look = int(prefix_look)
        self.after_look = int(after_look)
        self.warmup_steps = int(warmup_steps)
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        total_len = x.total_length
        block_start, block_end = block_loc.start, block_loc.end
        left_start = max(0, block_start - self.prefix_look)
        right_end = min(total_len, block_end + self.after_look)

        if self.iter_no < self.warmup_steps:
            out_full = model(x.data)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x)
            self.iter_no += 1
            return

        if kv_cache.past_key_values is None or (kv_cache.require_update(self.iter_no, block_start, block_end) and block_id > 0):
            out_full = model(x.data, use_cache=True)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x)
            kv_cache.update(out_full.past_key_values)
            self.cache_updates += 1
            self.iter_no += 1

        window_input = x.data[:, left_start:right_end]
        past_key_values, replace_position = kv_cache.get_key_values(left_start, right_end)
        out_step = model(window_input, past_key_values=past_key_values, use_cache=True, replace_position=replace_position)
        self.num_forwards += 1
        offset = block_start - left_start
        logits_block = out_step.logits[:, offset:offset + (block_end - block_start)]
        decoder.decode(logits_block, block_start, block_end, x)
        self.iter_no += 1

class VicinityCacheDiffusionLLM(BlockWiseDiffusionLLM):
    """ This diffusion LLM inference generates tokens with Vicinity Cache Update.

    The decoding algorithm defines a window to update KV-cache in each diffusion iteration.
    The window can be larger than the decoding block.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, maximum_unroll=4, expected_tpf=8,
                 prefix_look=0, after_look=0, warmup_steps=0, early_stop=True):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        assert cache_factory is not None, "This class requires a KV-cache."
        self.diff_iteration = VicinityCacheIteration(prefix_look, after_look, warmup_steps)
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

class IterSmoothWithVicinityCache(DiffusionIteration):
    """ A diffusion iteration to decode tokens
    """
    def __init__(self, model, prefix_look, after_look, warmup_steps,
            cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        super().__init__()
        self.prefix_look = int(prefix_look)
        self.after_look = int(after_look)
        self.warmup_steps = int(warmup_steps)

        self.cont_weight = cont_weight
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            self.h2e = model.module.h2e
        else:
            self.h2e = model.h2e
        self.cont_weight_init = cont_weight_init
        self.cont_weight_growth = cont_weight_growth
        self.threshold_decay = threshold_decay
        self.inputs_embeds = None
        self.iter_no = 0
    
    def reset_input_embeds(self, x):
        """ Reset input embedding with new input sequence
        """
        self.inputs_embeds = self.h2e(x.data)

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        total_len = x.total_length
        block_start, block_end = block_loc.start, block_loc.end
        left_start = max(0, block_start - self.prefix_look)
        right_end = min(total_len, block_end + self.after_look)

        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        if self.iter_no < self.warmup_steps:
            out_full = model(inputs_embeds=self.inputs_embeds)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x, iter_threshold)
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, out_full.logits, iter_cont_weight)
            self.iter_no += 1
            return

        if kv_cache.past_key_values is None or (kv_cache.require_update(self.iter_no, block_start, block_end) and block_id > 0):
            out_full = model(inputs_embeds=self.inputs_embeds, use_cache=True)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x, iter_threshold)
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, out_full.logits, iter_cont_weight)
            kv_cache.update(out_full.past_key_values)
            self.cache_updates += 1
            self.iter_no += 1

        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        past_key_values, replace_position = kv_cache.get_key_values(left_start, right_end)
        out_step = model(
                inputs_embeds=self.inputs_embeds[:, left_start:right_end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position
        )

        self.num_forwards += 1
        self.iter_no += 1
        offset = block_start - left_start
        logits_block = out_step.logits[:, offset:offset + (block_end - block_start)]
        decoder.decode(logits_block, block_start, block_end, x, iter_threshold)
        mask_index = (x.data[:, left_start:right_end] == decoder.mask_id)
        self.inputs_embeds[:, left_start:right_end] = self.h2e(x.data[:, left_start:right_end], mask_index, out_step.logits, iter_cont_weight)

class IterSmoothWithVicinityCacheDiffusionLLM(IterSmoothDiffusionLLM):
    """ This diffusion LLM inference generates tokens with vicinity cache and iteration smoothing.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, maximum_unroll=4, expected_tpf=8,
                 prefix_look=0, after_look=0, warmup_steps=0, early_stop=True, cont_weight=0.3,
                 cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        assert cache_factory is not None, "This class requires a KV-cache."
        self.diff_iteration = IterSmoothWithVicinityCache(model, prefix_look, after_look, warmup_steps,
                cont_weight=cont_weight, cont_weight_init=cont_weight_init, cont_weight_growth=cont_weight_growth,
                threshold_decay=threshold_decay)
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates


class BlockWiseDiffusionLLMWithSP(DiffusionLLM):
    """ Diffusion LLM inference with sequence parallel.

    This class performs diffusion LLM inference with sequence parallel.

    Parameters
    ----------
    rank : int
        The rank of the process
    world_size : int
        The number of processes to perform diffusion LLM inference with sequence parallel.
    model : Torch.Module
        The diffusion LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    """
    def __init__(self, rank, world_size, model, decoder, iterator_factory):
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.rank = rank
        self.world_size = world_size
        self.num_forwards = 0

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        '''
        Args:
            prompt: A tensor of shape (1, L).
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        '''
        op_num = 0
        x = DistAlignedTokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device, self.rank, self.world_size)
        it = self.iterator_factory.create(x, block_length)

        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            while (block == self.decoder.mask_id).sum()>0:
                part = x.total_length // self.world_size
                # TODO(zhengda) How does the model collect KV from other processes.
                partial_logits = self.model(x[:, (self.rank * part):((self.rank + 1) * part)].clone()).logits
                op_num += calculate_op_num(x[:, self.rank*part:(self.rank+1)*part])

                logits = gather_sequence_block(partial_logits, self.rank * part, (self.rank + 1) * part, block_loc.start, block_loc.end,
                        self.rank, self.world_size)
                self.decoder.decode(logits, block_loc.start, block_loc.end, x)
                self.num_forwards += 1
        return x.get_generated_tokens()

class BlockDiffusionLLMAttnmask(DiffusionLLM):
    """ Diffusion LLM inference

    This diffusion LLM inference generates tokens block by block with the implementation of Attention Mask.

    Comparing to the BlockWiseDiffusionLLM, this one does not feed the subsequent blocks 
    (which consist only of mask tokens) into the transformer when generating the earlier blocks, 
    thereby reducing overhead.

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.

    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, maximum_unroll=4, expected_tpf=8, backend='vllm'):
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.diff_iteration = BlockDiffusionIteration()
        self.block_runner = BlockDiffusionRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf, backend)
        

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return 0

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        assert prompt.shape[0] == 1, "We currently only support batch size = 1."
        # recalculate gen length and init iteratory
        # TODO(dulun): the implementation align with original bd decoder implementation.
        # We may need to refine to let users control the gen_length.
        prompt_length=prompt.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        new_gen_length=total_length-prompt_length
        
        
        # prepare block_mask and position IDs
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.model.device))
        bd_attn_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                        .repeat_interleave(block_length, dim=1).unsqueeze(0)
        pos_ids = torch.arange(total_length, device=self.model.device).unsqueeze(0)


        x = TokenArray(prompt, new_gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        # We don't need kv_cache for the implementation of attention mask
        kv_cache = None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            decode_compl = self.block_runner.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id, 
                pos_ids, bd_attn_mask)
            if decode_compl:
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

class BlockDiffusionLLM(DiffusionLLM):
    """ Diffusion LLM inference

    This diffusion LLM inference generates tokens block by block with the implementation of KV-Cache

    Comparing to the BlockWiseDiffusionLLM, this one does not feed the subsequent blocks 
    (which consist only of mask tokens) into the transformer when generating the earlier blocks, 
    thereby reducing overhead.

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.

    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, early_stop=True, maximum_unroll=4, expected_tpf=8, backend='vllm'):
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.cache_factory = cache_factory
        self.diff_iteration = BlockDiffusionIteration()
        self.block_runner = BlockDiffusionRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf, backend)
        self.early_stop = early_stop

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        # recalculate gen length and init iteratory
        # TODO(dulun): the implementation align with original bd decoder implementation.
        # We may need to refine to let users control the gen_length.
        batch_size = prompt.shape[0]
        prompt_length=prompt.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        new_gen_length=total_length-prompt_length

        # prepare block_mask and position IDs
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.model.device))
        bd_attn_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                        .repeat_interleave(block_length, dim=1).unsqueeze(0).repeat(batch_size, 1, 1)
        pos_ids = torch.arange(total_length, device=self.model.device).unsqueeze(0).repeat(batch_size, 1)

        x = TokenArray(prompt, new_gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)
        prompt_length = it._get_first_block_start()
        kv_cache = self.cache_factory.create()

        # prefill for kv_cache
        prefill_blocks = prompt_length // block_length
        prefill_length = prefill_blocks * block_length
        prefill_length = max(prefill_length, block_length)
        self.block_runner.prefill(self.model, x[:, :prefill_length], kv_cache, pos_ids[:, :prefill_length], bd_attn_mask[:,:prefill_length,:prefill_length])
        
        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            decode_compl = self.block_runner.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id, pos_ids, bd_attn_mask)
            if torch.all(decode_compl) and self.early_stop:
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()



class BlockMCMCDiffusionLLM(BlockWiseDiffusionLLM):
    """BlockWise Diffusion LLM with MCMC refinement (Power Sampling)

    This class extends BlockWiseDiffusionLLM to add MCMC-based block refinement
    using the Power Sampling algorithm. After each block is denoised, it performs
    MCMC iterations to sample from the power distribution p^α.
    
    Architecture:
    - Uses MCMCDiffusionIteration for diffusion iterations with confidence tracking
    - Uses MCMCBlockRunner for block-level decoding
    - Uses MCMCProposalGenerator for generating MCMC proposals
    - Uses MCMCRefinementRunner for MCMC refinement

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : MCMCThresholdParallelDecoder
        The decoder that decodes tokens from logits (must support mcmc_alpha)
    iterator_factory : IteratorFactory
        Factory class that generates iterator on token array
    enable_mcmc : bool
        Whether to enable MCMC refinement (default: True)
    n_mcmc_steps : int
        Number of MCMC iterations per block (default: 5)
    mcmc_alpha : float
        Power parameter α for target distribution p^α (default: 4.0)
    mcmc_temperature : float
        Temperature for proposal distribution (default: 0.0)
    mcmc_use_kv_cache : bool
        Whether to use KV Cache acceleration in MCMC proposal generation (default: True)
        When enabled, proposal generation will reuse KV Cache from previous computations,
        and supports snapshot/rollback mechanism for accept/reject decisions.
    tokenizer : Tokenizer, optional
        Tokenizer for verbose output
    verbose : bool
        Whether to print debug information (default: False)
    """
    def __init__(self, model, decoder, iterator_factory,
                 enable_mcmc=True, n_mcmc_steps=5,
                 mcmc_alpha=4.0, mcmc_temperature=0.0,
                 mcmc_use_kv_cache=True,
                 tokenizer=None, verbose=False,
                 early_stop=True, cache_factory=None, 
                 maximum_unroll=4, expected_tpf=8, **kwargs):
        # 不调用父类的 __init__，因为我们需要使用不同的组件
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.cache_factory = cache_factory
        
        # MCMC 参数
        self.enable_mcmc = enable_mcmc
        self.n_mcmc_steps = n_mcmc_steps
        self.mcmc_alpha = mcmc_alpha
        self.mcmc_temperature = mcmc_temperature
        self.mcmc_use_kv_cache = mcmc_use_kv_cache
        self.tokenizer = tokenizer
        self.verbose = verbose
        
        # 使用 MCMCDiffusionIteration 替代 BaseDiffusionIteration
        self.diff_iteration = MCMCDiffusionIteration(
            mcmc_alpha=mcmc_alpha,
            mcmc_temperature=mcmc_temperature
        )
        
        # 使用 MCMCBlockRunner 替代 BlockRunner
        self.block_decoder = MCMCBlockRunner(
            self.diff_iteration, early_stop, maximum_unroll, expected_tpf
        )
        
        # 创建 MCMC 组件
        if enable_mcmc:
            self.proposal_generator = MCMCProposalGenerator(
                model=model,
                decoder=decoder,
                mcmc_alpha=mcmc_alpha,
                mcmc_temperature=mcmc_temperature,
                use_kv_cache=mcmc_use_kv_cache
            )
            self.mcmc_runner = MCMCRefinementRunner(
                proposal_generator=self.proposal_generator,
                n_mcmc_steps=n_mcmc_steps
            )
        else:
            self.proposal_generator = None
            self.mcmc_runner = None

    @property
    def num_forwards(self):
        """Total number of forward passes"""
        base_forwards = self.diff_iteration.num_forwards
        if self.proposal_generator is not None:
            base_forwards += self.proposal_generator.num_forwards
        return base_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

    # 调试开关
    DEBUG_MCMC_GENERATE = False
    
    @torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        """Generate tokens with diffusion iterations and MCMC refinement
        
        Two-phase decoding per block:
        Phase 1: Standard diffusion denoising (using MCMCDiffusionIteration + MCMCBlockRunner)
        Phase 2: MCMC refinement (using MCMCRefinementRunner)
        """
        # Initialize token array
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        
        # Create iterator
        it = self.iterator_factory.create(x, block_length)
        
        # Reset iteration state and initialize confidence tensors
        self.diff_iteration.reset_confidences(x.data.shape, x.device)
        
        # Reset proposal generator forward count
        if self.proposal_generator is not None:
            self.proposal_generator.num_forwards = 0
        
        # Create KV cache
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        
        prompt_length = x.prompt.shape[1]
        
        # Iterate over blocks
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            
            if self.DEBUG_MCMC_GENERATE:
                print(f"\n{'='*60}")
                print(f"[BlockMCMCDiffusionLLM] Processing block {block_id}: [{block_loc.start}, {block_loc.end})")
                print(f"{'='*60}")
            
            # Phase 1: Denoise block using MCMCBlockRunner
            # MCMCDiffusionIteration will track confidences internally
            decode_compl = self.block_decoder.decode(
                self.model, self.decoder, x, kv_cache, block, block_loc, block_id
            )
            
            # DEBUG: 检查 Phase 1 完成后的置信度状态
            if self.DEBUG_MCMC_GENERATE:
                block_conf = self.diff_iteration.confidences_norm[:, block_loc.start:block_loc.end]
                num_inf = (block_conf == -np.inf).sum().item()
                block_length_actual = block_loc.end - block_loc.start
                block_tokens = x.data[:, block_loc.start:block_loc.end]
                num_masks = (block_tokens == self.decoder.mask_id).sum().item()
                
                # 计算预期的 -inf 数量（prompt 区域）
                prompt_overlap_end = min(block_loc.end, prompt_length)
                expected_inf = max(0, prompt_overlap_end - block_loc.start)
                
                print(f"[Phase 1 DONE] block={block_id}, masks_remaining={num_masks}, "
                      f"conf_inf={num_inf}/{block_length_actual} (expected={expected_inf} from prompt)")
                
                # 只有当 -inf 数量超过预期时才警告
                unexpected_inf = num_inf - expected_inf
                if num_masks == 0 and unexpected_inf > 0:
                    print(f"  ⚠️ WARNING: Block fully decoded but {unexpected_inf} unexpected positions have -inf confidence!")
                    # 找出 -inf 的位置（排除 prompt 区域）
                    gen_start_in_block = max(0, prompt_length - block_loc.start)
                    gen_conf = block_conf[:, gen_start_in_block:]
                    inf_positions = (gen_conf == -np.inf).nonzero(as_tuple=True)
                    if len(inf_positions[1]) > 0:
                        actual_positions = [p + gen_start_in_block for p in inf_positions[1][:20].tolist()]
                        print(f"  -inf positions (relative to block, gen region only): {actual_positions}")
            
            if self.verbose and self.tokenizer is not None:
                current_output = x.data[:, prompt_length:]
                decoded_output = self.tokenizer.batch_decode(current_output, skip_special_tokens=True)
                print(f"[Phase 1] Block {block_id} denoised: {decoded_output}")
            
            # Phase 2: MCMC refinement
            if self.enable_mcmc and self.mcmc_runner is not None:
                if self.DEBUG_MCMC_GENERATE:
                    print(f"[Phase 2 START] Starting MCMC refinement for block {block_id}")
                
                x, self.diff_iteration.confidences_norm, self.diff_iteration.confidences_unnorm, acceptance_rate = \
                    self.mcmc_runner.refine(
                        x, block_loc,
                        self.diff_iteration.confidences_norm,
                        self.diff_iteration.confidences_unnorm,
                        kv_cache=kv_cache,
                        verbose=self.verbose,
                        tokenizer=self.tokenizer
                    )
                
                if self.DEBUG_MCMC_GENERATE:
                    print(f"[Phase 2 DONE] block={block_id}, acceptance_rate={acceptance_rate:.2%}")
                
                logger.info(f'Block {block_id} MCMC acceptance rate: {acceptance_rate:.2%}')
            
            # Early stop if all sequences have EOS
            if torch.all(decode_compl):
                if self.DEBUG_MCMC_GENERATE:
                    print(f"[EARLY STOP] EOS detected, stopping at block {block_id}")
                break
        
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()


class MCMCDiffusionIteration(DiffusionIteration):
    """
    MCMC-aware diffusion iteration that tracks dual confidences
    for Metropolis-Hastings acceptance ratio computation.
    
    与 MCMCThresholdParallelDecoder 配合使用，该解码器返回双重置信度。
    
    Parameters
    ----------
    mcmc_alpha : float
        Power parameter α for target distribution p^α (default: 4.0)
    mcmc_temperature : float
        Temperature for sampling (default: 0.0)
    """
    
    def __init__(self, mcmc_alpha=4.0, mcmc_temperature=0.0):
        super().__init__()
        self.mcmc_alpha = mcmc_alpha
        self.mcmc_temperature = mcmc_temperature
        self.iter_no = 0
        
        # 置信度追踪 (全局，跨迭代累积)
        self.confidences_norm = None      # log p(x)
        self.confidences_unnorm = None    # log p^α(x)
    
    def reset_confidences(self, shape, device):
        """重置置信度张量 (在 generate() 开始时调用)
        
        Parameters
        ----------
        shape : tuple
            The shape of the confidence tensors (batch_size, seq_len)
        device : torch.device
            The device to place the tensors on
        """
        self.confidences_norm = torch.full(shape, -np.inf, dtype=torch.float32, device=device)
        self.confidences_unnorm = torch.full(shape, -np.inf, dtype=torch.float32, device=device)
        self.iter_no = 0
    
    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """
        Single diffusion iteration with dual confidence tracking.
        
        与 MCMCThresholdParallelDecoder 配合使用时，解码器会返回双重置信度。
        
        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : MCMCThresholdParallelDecoder
            The decoder (must be MCMCThresholdParallelDecoder for confidence tracking)
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache : KVCache or None
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
            
        Returns
        -------
        tuple : (conf_norm_block, conf_unnorm_block) - 当前块的置信度
        """
        from .parallel_strategy import MCMCThresholdParallelDecoder
        
        block_length = block_loc.end - block_loc.start
        conf_norm_block = None
        conf_unnorm_block = None
        
        # 关键修复：始终从 x 获取最新的 block，而不是使用传入的 block 参数
        # 这是因为在 unroll 循环中，block 参数可能是旧的
        current_block = x[:, block_loc.start:block_loc.end]
        
        # 辅助函数：更新全局置信度张量
        def update_global_confidences(conf_norm, conf_unnorm):
            if self.confidences_norm is not None and conf_norm is not None:
                mask_updated = conf_norm > -np.inf
                if mask_updated.any():
                    self.confidences_norm[:, block_loc.start:block_loc.end][mask_updated] = \
                        conf_norm[mask_updated]
                    self.confidences_unnorm[:, block_loc.start:block_loc.end][mask_updated] = \
                        conf_unnorm[mask_updated]
        
        # Step 1: 判断是否需要更新 KV Cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_loc.start, block_loc.end):
            output = model(x.data, use_cache=True)
            self.num_forwards += 1
            
            # 使用生成的 output 进行解码
            if isinstance(decoder, MCMCThresholdParallelDecoder):
                conf_norm_block, conf_unnorm_block = decoder.decode(
                    output.logits[:, block_loc.start:block_loc.end],
                    block_loc.start, block_loc.end, x,
                    mcmc_alpha=self.mcmc_alpha
                )
                # 更新全局置信度（Step 1 的置信度也要保存！）
                update_global_confidences(conf_norm_block, conf_unnorm_block)
            else:
                decoder.decode(output.logits[:, block_loc.start:block_loc.end], 
                              block_loc.start, block_loc.end, x)
            
            # 更新 KV-cache
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1
            
            # 更新 current_block（因为 x 可能已经被修改）
            current_block = x[:, block_loc.start:block_loc.end]
        
        # Step 2: 前向传播获取 logits
        if kv_cache is None:
            # 无 KV cache，直接对整个输入进行前向传播
            logits = model(x.data).logits[:, block_loc.start:block_loc.end]
        elif kv_cache.cache_type == 'prefix':
            # 前缀缓存模式
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            output = model(x[:, block_loc.start:], 
                          past_key_values=past_key_values, 
                          use_cache=True,
                          replace_position=replace_position)
            logits = output.logits[:, :block_length]
        else:
            # 双向缓存模式 (dual) - 使用最新的 current_block
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            output = model(current_block, 
                          past_key_values=past_key_values, 
                          use_cache=True,
                          replace_position=replace_position)
            logits = output.logits
        
        # Step 3: 解码并获取置信度
        if isinstance(decoder, MCMCThresholdParallelDecoder):
            conf_norm_block, conf_unnorm_block = decoder.decode(
                logits, block_loc.start, block_loc.end, x,
                mcmc_alpha=self.mcmc_alpha
            )
            # 更新全局置信度张量
            update_global_confidences(conf_norm_block, conf_unnorm_block)
        else:
            # 标准解码器，不返回置信度
            decoder.decode(logits, block_loc.start, block_loc.end, x)
        
        self.num_forwards += 1
        self.iter_no += 1
        
        return conf_norm_block, conf_unnorm_block


class MCMCBlockRunner(BlockRunner):
    """
    MCMC-aware block runner that works with MCMCDiffusionIteration.
    
    与 BlockRunner 几乎相同，但正确处理 MCMCDiffusionIteration 的返回值。
    实际上 Python 会自动忽略未使用的返回值，所以这个类主要是为了语义清晰。
    """
    
    # 调试开关
    DEBUG_MCMC_BLOCK_RUNNER = False
    
    def decode(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """
        Decode all tokens in a block with MCMC confidence tracking.
        
        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder (preferably MCMCThresholdParallelDecoder)
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache : KVCache
            The KV-cache
        block : torch.Tensor
            The input tokens in the block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
            
        Returns
        -------
        torch.Tensor : a bool tensor that indicates whether the sequences have finished decoding.
        """
        orig_x = x
        seq_idx = torch.arange(x.batch_size, device=block.device)
        seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=False)
        block = x[:, block_loc.start:block_loc.end]
        batch_size = x.batch_size
        
        # 获取 prompt 长度，用于判断 -inf 是否是预期的
        prompt_length = orig_x.prompt.shape[1]
        
        # DEBUG: 记录初始状态
        initial_masks = (block == decoder.mask_id).sum().item()
        iteration_count = 0
        
        while (block == decoder.mask_id).sum() > 0:
            unroll_k = int(max(min((block == decoder.mask_id).sum() // self.expected_tpf, self.maximum_unroll), 1))
            for unroll_i in range(unroll_k):
                iteration_count += 1
                # MCMCDiffusionIteration.forward() 返回 (conf_norm, conf_unnorm)
                # 置信度已经在 forward() 中更新到全局张量，这里忽略返回值
                _ = self.diff_iteration.forward(model, decoder, x, kv_cache, block, block_loc, block_id)
            
            # 更新 block 引用
            block = x[:, block_loc.start:block_loc.end]
            
            if batch_size > 1:
                seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=True)
                block = x[:, block_loc.start:block_loc.end]
                if len(seq_idx) == 0:
                    break
            batch_size = x.batch_size
        
        # DEBUG: 检查去噪完成后的置信度状态
        final_masks = (block == decoder.mask_id).sum().item()
        
        if self.DEBUG_MCMC_BLOCK_RUNNER:
            # 检查全局置信度张量
            if hasattr(self.diff_iteration, 'confidences_norm') and self.diff_iteration.confidences_norm is not None:
                block_conf = self.diff_iteration.confidences_norm[:, block_loc.start:block_loc.end]
                num_inf = (block_conf == -np.inf).sum().item()
                block_length = block_loc.end - block_loc.start
                
                # 计算预期的 -inf 数量（prompt 区域）
                prompt_overlap_start = block_loc.start
                prompt_overlap_end = min(block_loc.end, prompt_length)
                expected_inf = max(0, prompt_overlap_end - prompt_overlap_start)
                
                print(f"[MCMCBlockRunner] block_id={block_id}, block=[{block_loc.start},{block_loc.end}), "
                      f"iterations={iteration_count}, masks: {initial_masks}->{final_masks}, "
                      f"conf_inf={num_inf}/{block_length} (expected_inf={expected_inf} from prompt)")
                
                # 只有当 -inf 数量超过预期时才警告
                unexpected_inf = num_inf - expected_inf
                if final_masks == 0 and unexpected_inf > 0:
                    logger.warning(f"[MCMCBlockRunner] PROBLEM: Block fully decoded but {unexpected_inf} unexpected positions have -inf confidence!")
                    # 找出哪些位置是 -inf（排除 prompt 区域）
                    gen_start_in_block = max(0, prompt_length - block_loc.start)
                    gen_conf = block_conf[:, gen_start_in_block:]
                    inf_positions = (gen_conf == -np.inf).nonzero(as_tuple=True)
                    if len(inf_positions[1]) > 0:
                        actual_positions = [p + gen_start_in_block for p in inf_positions[1][:10].tolist()]
                        logger.warning(f"  -inf positions (relative to block, gen region only): {actual_positions}...")
        
        eos_idx = torch.any(orig_x[:, block_loc.start:block_loc.end] == decoder.eos_id, dim=1)
        if self.early_stop:
            orig_x[eos_idx, block_loc.end:] = decoder.eos_id
        return eos_idx


class MCMCProposalGenerator:
    """
    生成 MCMC 提议序列，支持 KV Cache 加速。
    
    从指定位置开始重新掩码并去噪，生成提议序列及其置信度。
    
    Parameters
    ----------
    model : pytorch model
        The diffusion LLM
    decoder : MCMCThresholdParallelDecoder
        The decoder for token decoding
    mcmc_alpha : float
        Power parameter α for target distribution p^α
    mcmc_temperature : float
        Temperature for Gumbel noise sampling
    use_kv_cache : bool
        Whether to use KV Cache acceleration in proposal generation (default: True)
    """
    
    def __init__(self, model, decoder, mcmc_alpha=4.0, mcmc_temperature=0.0, use_kv_cache=True):
        self.model = model
        self.decoder = decoder
        self.mcmc_alpha = mcmc_alpha
        self.mcmc_temperature = mcmc_temperature
        self.use_kv_cache = use_kv_cache
        self.num_forwards = 0
    
    def generate(self, x_current, idx, block_end, confidences_norm, confidences_unnorm, 
                 kv_cache=None, verbose=False, tokenizer=None):
        """
        从 idx 位置开始重新生成序列，支持 KV Cache 加速。
        
        KV Cache 使用策略:
        1. 位置 [0, idx) 的 KV 可以复用（这部分序列没有变化）
        2. 位置 [idx, block_end) 需要重新计算
        
        Parameters
        ----------
        x_current : TokenArray
            当前序列
        idx : int
            重采样起始位置
        block_end : int
            块结束位置
        confidences_norm : torch.Tensor
            当前归一化置信度
        confidences_unnorm : torch.Tensor
            当前非归一化置信度
        kv_cache : DiffusionKVCacheManager, optional
            KV Cache 管理器，用于加速前向传播
        verbose : bool
            是否打印调试信息
        tokenizer : Tokenizer, optional
            用于打印调试信息的 tokenizer
            
        Returns
        -------
        tuple : (x_prop, conf_norm_prop, conf_unnorm_prop)
        """
        # Clone current sequence
        x_prop = TokenArray(x_current.prompt, x_current.gen_length, 
                           self.decoder.mask_id, self.decoder.eos_id, x_current.device)
        x_prop.data = x_current.data.clone()
        
        # Remask from idx to block_end
        x_prop.data[:, idx:block_end] = self.decoder.mask_id
        
        # Reset confidences for remasked region
        conf_norm_prop = confidences_norm.clone()
        conf_unnorm_prop = confidences_unnorm.clone()
        conf_norm_prop[:, idx:block_end] = -np.inf
        conf_unnorm_prop[:, idx:block_end] = -np.inf
        
        # Iterative denoising
        block = x_prop.data[:, idx:block_end]
        block_mask_index = (block == self.decoder.mask_id)
        steps = (block == self.decoder.mask_id).sum().item()
        
        # 检查是否可以使用 KV Cache
        # 需要同时满足：1) 启用了 KV Cache 2) kv_cache 不为 None 3) past_key_values 已初始化
        use_kv_cache = (self.use_kv_cache and 
                        kv_cache is not None and 
                        kv_cache.past_key_values is not None)
        
        if verbose and tokenizer is not None:
            print(f"[MCMCProposalGenerator] Starting denoising from idx={idx} to {block_end}, "
                  f"steps={steps}, use_kv_cache={use_kv_cache} (enabled={self.use_kv_cache})")
        
        if steps > 0:
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            block_length = block_end - idx
            
            for i in range(steps):
                # Forward pass with optional KV Cache
                if use_kv_cache:
                    # 使用 KV Cache 加速
                    past_key_values, replace_position = kv_cache.get_key_values(idx, block_end)
                    
                    if kv_cache.cache_type == 'prefix':
                        # 前缀缓存模式：从 idx 开始的所有 token
                        output = self.model(
                            x_prop.data[:, idx:],
                            past_key_values=past_key_values,
                            use_cache=True,
                            replace_position=replace_position
                        )
                        logits = output.logits[:, :block_length]
                    else:
                        # 双向缓存模式：只处理当前块
                        output = self.model(
                            x_prop.data[:, idx:block_end],
                            past_key_values=past_key_values,
                            use_cache=True,
                            replace_position=replace_position
                        )
                        logits = output.logits
                else:
                    # 无 KV Cache，完整前向传播
                    logits = self.model(x_prop.data).logits[:, idx:block_end]
                
                self.num_forwards += 1
                
                # Compute dual log probabilities
                log_p_norm = torch.nn.functional.log_softmax(logits, dim=-1)
                log_p_unnorm = torch.nn.functional.log_softmax(self.mcmc_alpha * logits, dim=-1)
                
                # Sample with Gumbel noise
                logits_with_noise = add_gumbel_noise_power(logits, alpha=1.0, temperature=self.mcmc_temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                
                # Get log probs for selected tokens
                x0_logp_norm = torch.gather(log_p_norm, -1, x0.unsqueeze(-1)).squeeze(-1)
                x0_logp_unnorm = torch.gather(log_p_unnorm, -1, x0.unsqueeze(-1)).squeeze(-1)
                
                # Compute confidence for remasking
                p = torch.nn.functional.softmax(logits, dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                
                # Select tokens to transfer
                mask_index = (block == self.decoder.mask_id)
                x0 = torch.where(mask_index, x0, block)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                
                # Update proposal sequence and confidences
                if transfer_index.any():
                    x_prop.data[:, idx:block_end][transfer_index] = x0[transfer_index]
                    conf_norm_prop[:, idx:block_end][transfer_index] = x0_logp_norm[transfer_index].float()
                    conf_unnorm_prop[:, idx:block_end][transfer_index] = x0_logp_unnorm[transfer_index].float()
                
                block = x_prop.data[:, idx:block_end]
                
                if verbose and tokenizer is not None:
                    prompt_length = x_prop.prompt.shape[1]
                    current_output = x_prop.data[:, prompt_length:block_end]
                    decoded_proposal = tokenizer.batch_decode(current_output, skip_special_tokens=True)
                    print(f"[MCMCProposalGenerator] Step {i+1}/{steps}: {decoded_proposal}")
        
        return x_prop, conf_norm_prop, conf_unnorm_prop


class MCMCRefinementRunner:
    """
    MCMC 精炼运行器，负责单个块的 MCMC 迭代。
    
    使用 Metropolis-Hastings 算法对已去噪的块进行精炼，
    从 power distribution p^α 中采样。
    
    Parameters
    ----------
    proposal_generator : MCMCProposalGenerator
        提议序列生成器
    n_mcmc_steps : int
        每个块的 MCMC 迭代次数
    """
    
    def __init__(self, proposal_generator, n_mcmc_steps=5):
        self.proposal_generator = proposal_generator
        self.n_mcmc_steps = n_mcmc_steps
    
    def refine(self, x, block_loc, confidences_norm, confidences_unnorm, 
               kv_cache=None, verbose=False, tokenizer=None):
        """
        对块进行 MCMC 精炼。
        
        Parameters
        ----------
        x : TokenArray
            当前序列
        block_loc : BlockLoc
            块位置信息
        confidences_norm : torch.Tensor
            归一化置信度张量
        confidences_unnorm : torch.Tensor
            非归一化置信度张量
        kv_cache : KVCache, optional
            KV Cache
        verbose : bool
            是否打印调试信息
        tokenizer : Tokenizer, optional
            用于打印调试信息的 tokenizer
            
        Returns
        -------
        tuple : (x_refined, confidences_norm, confidences_unnorm, acceptance_rate)
        """
        acceptances = 0
        prompt_length = x.prompt.shape[1]
        eos_id = self.proposal_generator.decoder.eos_id
        
        # 计算有效的 MCMC 起始位置（必须在生成区域内）
        effective_block_start = max(block_loc.start, prompt_length)
        
        # 如果整个块都在 prompt 区域内，跳过 MCMC
        if effective_block_start >= block_loc.end:
            if verbose:
                print(f"[MCMC] Block [{block_loc.start}, {block_loc.end}) is entirely in prompt region, skipping")
            return x, confidences_norm, confidences_unnorm, 1.0
        
        # 导入 KVCacheSnapshot（用于快照/回滚）
        from .utils import KVCacheSnapshot
        
        for mcmc_step in range(self.n_mcmc_steps):
            # Step 1: 随机选择重采样位置 (必须在生成区域内)
            idx = random.randint(effective_block_start, block_loc.end - 1)
            
            # Step 2: 创建 KV Cache 快照（如果使用 KV Cache）
            snapshot = None
            if kv_cache is not None:
                snapshot = KVCacheSnapshot(idx, block_loc.end)
                snapshot.save(kv_cache)
            
            # Step 3: 生成提议序列（可能会更新 KV Cache）
            x_prop, conf_norm_prop, conf_unnorm_prop = self.proposal_generator.generate(
                x, idx, block_loc.end, 
                confidences_norm, confidences_unnorm,
                kv_cache=kv_cache,
                verbose=verbose, tokenizer=tokenizer
            )
            
            # Step 4: 计算 MH 接受率（传入 prompt_length 以正确处理边界情况）
            log_r = self._compute_log_acceptance_ratio(
                confidences_norm, confidences_unnorm,
                conf_norm_prop, conf_unnorm_prop,
                idx, block_loc.end,
                prompt_length=prompt_length,
                verbose=verbose
            )
            
            # Step 5: 接受/拒绝决策
            accept_prob = min(1.0, np.exp(min(log_r, 0.0)))
            accepted = np.random.rand() < accept_prob
            
            if verbose and tokenizer is not None:
                print(f"\n[MCMC Step {mcmc_step+1}/{self.n_mcmc_steps}] idx={idx}, "
                      f"log_r={log_r:.4f}, accept_prob={accept_prob:.4f}, accepted={accepted}")
            
            if accepted:
                # 接受提议：保留新的 x 和 KV Cache（不需要回滚）
                acceptances += 1
                x = x_prop
                # 只更新重新采样区域的置信度
                confidences_norm[:, idx:block_loc.end] = conf_norm_prop[:, idx:block_loc.end]
                confidences_unnorm[:, idx:block_loc.end] = conf_unnorm_prop[:, idx:block_loc.end]
                
                if verbose and tokenizer is not None:
                    current_output = x.data[:, prompt_length:]
                    decoded = tokenizer.batch_decode(current_output, skip_special_tokens=True)
                    print(f"[ACCEPTED] New sequence: {decoded}")
            else:
                # 拒绝提议：回滚 KV Cache 到快照状态
                if snapshot is not None and snapshot.is_saved:
                    snapshot.restore(kv_cache)
                    if verbose:
                        print(f"[REJECTED] KV Cache restored to snapshot")
                
                if verbose and tokenizer is not None:
                    current_output = x.data[:, prompt_length:]
                    decoded = tokenizer.batch_decode(current_output, skip_special_tokens=True)
                    print(f"[REJECTED] Keep sequence: {decoded}")
        
        # 检查 EOS token
        if eos_id in x.data[0, block_loc.start:block_loc.end]:
            block_tokens = x.data[0, block_loc.start:block_loc.end].tolist()
            eos_idx = block_tokens.index(eos_id) + block_loc.start
            confidences_norm[:, eos_idx+1:] = -np.inf
            confidences_unnorm[:, eos_idx+1:] = -np.inf
            if verbose:
                print(f"[MCMC] EOS token detected at position {eos_idx}, truncating confidences")
        
        acceptance_rate = acceptances / self.n_mcmc_steps if self.n_mcmc_steps > 0 else 0.0
        return x, confidences_norm, confidences_unnorm, acceptance_rate
    
    def _compute_log_acceptance_ratio(self, confidences_norm_cur, confidences_unnorm_cur,
                                       confidences_norm_prop, confidences_unnorm_prop,
                                       idx, block_end, prompt_length=None, verbose=False):
        """计算 Metropolis-Hastings 接受率的对数
        
        Parameters
        ----------
        confidences_norm_cur : torch.Tensor
            当前序列的归一化置信度
        confidences_unnorm_cur : torch.Tensor
            当前序列的非归一化置信度
        confidences_norm_prop : torch.Tensor
            提议序列的归一化置信度
        confidences_unnorm_prop : torch.Tensor
            提议序列的非归一化置信度
        idx : int
            重采样起始位置
        block_end : int
            块结束位置
        prompt_length : int, optional
            Prompt 长度，用于跳过 prompt 区域
        verbose : bool
            是否打印调试信息
        """
        # 计算有效区域（跳过 prompt 区域）
        effective_start = idx
        if prompt_length is not None and idx < prompt_length:
            effective_start = prompt_length
        
        # 如果有效区域为空，直接返回 0（接受）
        if effective_start >= block_end:
            if verbose:
                print(f"[Acceptance Ratio] No generation region in [{idx}, {block_end}), prompt_length={prompt_length}")
            return 0.0
        
        # 提取有效区域的对数概率
        log_prob_cur_norm_raw = confidences_norm_cur[:, effective_start:block_end].view(-1).tolist()
        log_prob_cur_unnorm_raw = confidences_unnorm_cur[:, effective_start:block_end].view(-1).tolist()
        log_prob_prop_norm_raw = confidences_norm_prop[:, effective_start:block_end].view(-1).tolist()
        log_prob_prop_unnorm_raw = confidences_unnorm_prop[:, effective_start:block_end].view(-1).tolist()
        
        # 调试：统计 -inf 的数量
        region_length = block_end - effective_start
        cur_norm_inf_count = sum(1 for x in log_prob_cur_norm_raw if x == -np.inf)
        prop_norm_inf_count = sum(1 for x in log_prob_prop_norm_raw if x == -np.inf)
        
        if verbose:
            print(f"[DEBUG] Effective region: [{effective_start}, {block_end}), length={region_length}")
            print(f"[DEBUG] cur_norm: {region_length - cur_norm_inf_count}/{region_length} valid")
            print(f"[DEBUG] prop_norm: {region_length - prop_norm_inf_count}/{region_length} valid")
        
        # 过滤 -inf 值
        log_prob_cur_norm = [x for x in log_prob_cur_norm_raw if x > -np.inf]
        log_prob_cur_unnorm = [x for x in log_prob_cur_unnorm_raw if x > -np.inf]
        log_prob_prop_norm = [x for x in log_prob_prop_norm_raw if x > -np.inf]
        log_prob_prop_unnorm = [x for x in log_prob_prop_unnorm_raw if x > -np.inf]
        
        if verbose:
            print(f"[Acceptance Ratio] After filtering:")
            print(f"  cur_norm len={len(log_prob_cur_norm)}, sum={sum(log_prob_cur_norm) if log_prob_cur_norm else 0:.4f}")
            print(f"  prop_norm len={len(log_prob_prop_norm)}, sum={sum(log_prob_prop_norm) if log_prob_prop_norm else 0:.4f}")
        
        # 检查长度匹配
        if len(log_prob_cur_norm) != len(log_prob_prop_norm):
            logger.warning(f"[LENGTH MISMATCH] cur_norm={len(log_prob_cur_norm)}, prop_norm={len(log_prob_prop_norm)}")
            logger.warning(f"  Region: [{effective_start}, {block_end}), cur_inf={cur_norm_inf_count}, prop_inf={prop_norm_inf_count}")
            return -np.inf  # 拒绝提议
        
        # 如果没有有效的置信度，返回 0（接受）
        if len(log_prob_cur_norm) == 0:
            return 0.0
        
        # MH 接受率: log r = log[p^α(x') * q(x|x')] - log[p^α(x) * q(x'|x)]
        log_r = (sum(log_prob_prop_unnorm) + sum(log_prob_cur_norm)
                - sum(log_prob_cur_unnorm) - sum(log_prob_prop_norm))
        
        return log_r