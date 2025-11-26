import math
import copy
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if math.isclose(temperature, 0.0):
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def add_gumbel_noise_power(logits, alpha=4.0, temperature=0.0):
    """
    Power-scaled Gumbel noise for MCMC sampling.
    Applies power scaling (alpha) before adding Gumbel noise.
    """
    scaled_logits = alpha * logits
    if temperature == 0:
        return scaled_logits
    scaled_logits = scaled_logits.to(torch.float64)
    noise = torch.rand_like(scaled_logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return scaled_logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def calculate_op_num(x, hidden_size=4096, mlp_hidden_size = 12288, vocab_size = 126464, num_hidden_layers=32, cache_length=0):
    cfg_factor = 1
    qkv_ops = 4*x.shape[0]*hidden_size*hidden_size*x.shape[1]*2
    attn_ops = x.shape[0]*(cache_length)*x.shape[1]*hidden_size*2
    ffn_ops = 3*x.shape[0]*hidden_size*mlp_hidden_size*x.shape[1]*2
    layer_ops = qkv_ops + attn_ops + ffn_ops
    op_num = cfg_factor * (num_hidden_layers*layer_ops + x.shape[0]*hidden_size*vocab_size*x.shape[1]*2)
    return op_num/1e12 

class TokenArray:
    """ A token array to support read, update and expansion.

    We need to access the tokens that have been generated and write new tokens to the array.
    Some algorithms require to expand the token array.

    Parameters
    ----------
    prompt : Torch.Tensor
        The array that contains the input prompt.
    gen_length : int
        The number of tokens to be generated.
    mask_id : int
        the mask id of the masked tokens
    device : Torch.Device
        The device where the token array is placed on.
    """
    def __init__(self, prompt, gen_length, mask_id, eos_id, device):
        self.prompt = prompt.to(device)
        self.data = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
        self.data[:, :prompt.shape[1]] = prompt.clone()
        self.gen_length = gen_length
        self.eos_id = eos_id
        self.mask_id = mask_id

    @property
    def total_length(self):
        return self.prompt.shape[1] + self.gen_length

    @property
    def batch_size(self):
        return self.prompt.shape[0]

    @property
    def device(self):
        return self.data.device

    def expand(self, new_len):
        pass

    def get_generated_tokens(self):
        if self.batch_size == 1:
            return self.data[self.data != self.eos_id].unsqueeze(0)
        else:
            return self.data

    def select_seqs(self, idx):
        arr = copy.copy(self)
        arr.prompt = self.prompt[idx]
        arr.data = self.data[idx]
        return arr

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, vals):
        self.data[idx] = vals

class DistAlignedTokenArray:
    """ A token array to support read, update and expansion in the distributed setting.

    In this setting, each process still contains the full copy of the token array.
    The main difference from TokenArray is that this class makes sure that the length of the token array
    is rounded to the world size.

    Parameters
    ----------
    prompt : Torch.Tensor
        The array that contains the input prompt.
    gen_length : int
        The number of tokens to be generated.
    mask_id : int
        the mask id of the masked tokens
    device : Torch.Device
        The device where the token array is placed on.
    rank : int
        The rank of the process
    world_size : int
        The number of processes.
    """
    def __init__(self, prompt, gen_length, mask_id, eos_id, device, rank, world_size):
        total_length = prompt.shape[1] + gen_length
        if total_length % world_size != 0:
            total_length = (total_length // world_size + 1) * world_size
        self.data = torch.full((prompt.shape[0], total_length), mask_id, dtype=torch.long).to(device)
        self.data[:, :prompt.shape[1]] = prompt.clone()
        self.orig_gen_length = gen_length
        self.gen_length = total_length - prompt.shape[1]
        self.prompt = prompt
        self.eos_id = eos_id
        self.mask_id = mask_id

    @property
    def total_length(self):
        return self.prompt.shape[1] + self.gen_length

    @property
    def device(self):
        return self.data.device

    def get_generated_tokens(self):
        return self.data[self.data != self.eos_id].unsqueeze(0)

    def expand(self, new_len):
        pass

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, vals):
        self.data[idx] = vals

class BlockLoc:
    """ The location of the block in the token array.
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end



class BlockIterator:
    """ Block iterator

    This performs block-wise iteration on the input token array for diffusion decoding.

    Parameters
    ----------
    x : TokenArray
        The token array that contains decoded tokens and stores the new generated tokens
    block_length : int
        The length of the block
    start_block_align : bool
        将第一个解码块与块大小对齐。第一个块可能与提示重叠。
        举例说明:
        假设 prompt 的有效长度 (non_mask_number) 是 15, block_length 是 8。

        - 如果 `start_block_align = False` (默认):
            解码将从 prompt 结束的位置 (15) 开始。第一个解码块的范围是 `[15, 23)`。

        - 如果 `start_block_align = True`:
            `_get_first_block_start` 会被调用来计算起始位置:
            `start = (15 // 8) * 8 = 1 * 8 = 8`。
            解码将从位置 8 开始。第一个解码块的范围是 `[8, 16)`。
            这意味着第一个解码块 `[8, 16)` 会与 prompt 的一部分 `[8, 15)` 重叠。

        这样做的目的是为了让解码过程严格按照块的网格进行，即使这意味着需要重新生成或处理一部分 prompt 内容。这在某些扩散模型中可能有助于生成更连贯的文本。
    """
    def __init__(self, x, block_length, start_block_align=False):
        self.x = x
        self.iter = 0
        self.block_length = block_length
        self.start_block_align = start_block_align
        if start_block_align:
            self.first_block_start = self._get_first_block_start()
        else:
            self.first_block_start = self.x.prompt.shape[1]

    def _get_first_block_start(self):
        prompt = self.x.prompt
        non_mask_number = (prompt != self.x.mask_id).sum(dim=-1).min().item()
        start = ((non_mask_number) // self.block_length) * self.block_length
        return start

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        current_block_start = self.first_block_start + self.iter * self.block_length
        if current_block_start >= self.x.total_length:
            raise StopIteration
        current_block_end = min(current_block_start + self.block_length, self.x.total_length)
        assert current_block_end <= self.x.total_length
        self.iter += 1
        return BlockLoc(current_block_start, current_block_end), self.x[:, current_block_start:current_block_end]

class BlockDiffusionIterator():
    """ Block iterator

    This performs block-wise iteration on the input token array for diffusion decoding.

    Parameters
    ----------
    x : TokenArray
        The token array that contains decoded tokens and stores the new generated tokens
    block_length : int
        The length of the block
    """
    def __init__(self, x, block_length):
        self.x = x
        self.iter = 0
        self.block_length = block_length
        self.first_block_start = self._get_first_block_start()
    
    def _get_first_block_start(self):
        prompt = self.x.prompt
        non_mask_number = (prompt != self.x.mask_id).sum(dim=-1).min().item()
        start = ((non_mask_number) // self.block_length) * self.block_length
        return start

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        current_block_start = self.first_block_start + self.iter * self.block_length
        if current_block_start >= self.x.total_length:
            raise StopIteration
        current_block_end = min(current_block_start + self.block_length, self.x.total_length)
        assert current_block_end <= self.x.total_length
        self.iter += 1
        return BlockLoc(current_block_start, current_block_end), self.x[current_block_start:current_block_end]


class BlockIteratorFactory:
    """ Iterator factory

    This generates iterators for DiffusionLLM to iterate over a sequence.

    Parameters
    ----------
    start_block_align : bool
        Align the first decoding block to the block size. The first block may overlap with the prompt.
    
    use_block_diffusion: bool
        If this flag set to True, the block diffusion iteration will be used and start_block_algin will be ignored.

    Returns
    -------
    BlockIterator : the block iterator.
    """
    def __init__(self, start_block_align=False, use_block_diffusion=False):
        self._start_block_align = start_block_align
        self._use_bd = use_block_diffusion

    def create(self, x, block_length):
        if self._use_bd:
            return BlockDiffusionIterator(x, block_length)
        else:
            return BlockIterator(x, block_length, start_block_align=self._start_block_align)


class KVCache:
    """
    The KV-cache
    封装了Transformer模型中的Key-Value缓存。将分散的K/V张量整合成一个连续的块以提高效率

    存储结构与维度说明
    ----------------
    内部数据 `self._data` 是一个 6D 张量，形状为:
    `[num_layers, 2, batch_size, num_heads, seq_len, hidden_dim]`

    其中各维度含义如下：
    - `num_layers`: Transformer 的层数。
    - `2`: 第2维大小为2，分别存储 Key (索引0) 和 Value (索引1)。
    - `batch_size`: 批次大小 (dim 0 of get_keys)。
    - `num_heads`: 注意力头的数量 (dim 1 of get_keys)。
    - `seq_len`: 序列长度 (dim 2 of get_keys)。**这就是为什么 update 操作使用 dim=2**。
    - `hidden_dim`: 每个注意力头的维度大小 (dim 3 of get_keys)。

    当调用 `get_keys(layer_idx)` 时，返回的张量形状为 `[batch_size, num_heads, seq_len, hidden_dim]`。
    因此，在 `update` 函数中，`dim=2` 指向的是 **序列长度 (seq_len)** 维度。无论是拼接 (`cat`) 还是切片替换 (`slice_scatter`)，我们都是在时间步（序列长度）方向上进行操作。

    Parameters
    ----------
    past_key_values : List[torch.Tensor]
        The keys and values of each transformer layer.
    """
    def __init__(self, past_key_values, backend='vllm', length=2048, cache_align_size=256):
        if backend == 'vllm':
            assert len(past_key_values) % 2 == 0
            self._data = past_key_values
        else:
            self.cache_align_size = cache_align_size
            assert len(past_key_values) % 2 == 0
            self._raw_data = past_key_values
            # 首先将输入的列表形式的KV缓存整合成一个大的张量
            self._consolidate_raw()
            self.length = max((self._raw_data.shape[4]+64+self.cache_align_size-1)//self.cache_align_size*self.cache_align_size, length)
            device = self._raw_data.device
            num_layer, _, batch_size, num_heads, seq_len, hidden_dim = self._raw_data.shape
            # 创建一个预分配好内存的、全为零的巨大张量来存储KV缓存
            self._data = torch.zeros(num_layer, 2, batch_size, num_heads, self.length, hidden_dim, device=device, dtype=torch.bfloat16)
            # 将输入的初始KV值复制到这个巨大张量的开头部分
            self._data[:, :, :, :, :seq_len] = self._raw_data
    def consolidate(self):
        if isinstance(self._data, torch.Tensor):
            return

        num_layers = len(self._data) // 2
        inner_shape = self._data[0].shape
        # The shape is [num_layers, 2, batch_size, num_heads, seq_len, hidden_dim]
        self._data = torch.stack(self._data, dim=0).reshape(num_layers, 2, *inner_shape)

    def _consolidate_raw(self):
        if isinstance(self._raw_data, torch.Tensor):
            return

        num_layers = len(self._raw_data) // 2
        inner_shape = self._raw_data[0].shape
        # The shape is [num_layers, 2, batch_size, num_heads, seq_len, hidden_dim]
        self._raw_data = torch.stack(self._raw_data, dim=0).reshape(num_layers, 2, *inner_shape)

    @property
    def num_layers(self):
        assert isinstance(self._data, torch.Tensor)
        return self._data.shape[0]

    @property
    def seq_len(self):
        assert isinstance(self._data, torch.Tensor)
        return self._data.shape[4]

    def get_keys(self, layer_idx):
        """ Get the keys of a transformer layer.
        """
        assert isinstance(self._data, torch.Tensor)
        return self._data[layer_idx][0]

    def get_values(self, layer_idx):
        """ Get the values of a transformer layer.
        """
        assert isinstance(self._data, torch.Tensor)
        return self._data[layer_idx][1]

    def update(self, key_states, val_states, layer_idx, replace_position=None, backend='vllm'):
        """ Update the keys and values of a transformer layer.

        Parameters
        ----------
        key_states : torch.Tensor
            The keys in a block of tokens. The shape is [batch_size, num_heads, seq_len, hidden_dim]
        val_states : torch.Tensor
            The values in a block of tokens. The shape is [batch_size, num_heads, seq_len, hidden_dim]
        layer_idx : int
            The index of the transformer layer
        replace_position : Tuple[int]
            The start and the end position where keys and values should be updated.

        Returns
        -------
        torch.Tensor: the new keys for the entire sequence of the transformer layer.
        torch.Tensor: the new values for the entire sequence of the transformer layer.
        """
        if backend ==  'vllm':
            # 当提供了 `replace_position` 时，执行的是“就地更新”。
            # 这意味着我们用新的 `key_states` 和 `val_states` 替换掉缓存中从 `start` 到 `end` 的部分。
            if replace_position is not None:
                # slice_scatter(src, dim, start, end) 的作用是将 src 张量嵌入到目标张量的指定维度 dim 的 [start, end) 范围内。
                keys = self.get_keys(layer_idx).slice_scatter(key_states, dim=2, start=replace_position[0], end=replace_position[1])
                values = self.get_values(layer_idx).slice_scatter(val_states, dim=2, start=replace_position[0], end=replace_position[1])
            else:
                # 当没有提供 `replace_position` 时，执行的是“前缀缓存”更新。
                # 这意味着我们将新的 `key_states` 和 `val_states` 附加到现有缓存的末尾。
                # torch.cat([t1, t2], dim=2) 将 t2 拼接在 t1 的第 2 维度后面，从而延长了序列长度。
                keys = torch.cat([self.get_keys(layer_idx), key_states], dim=2)
                values = torch.cat([self.get_values(layer_idx), val_states], dim=2)
        else:
            # 对于非vllm后端，这里的逻辑是替换掉缓存的最后一部分。
            cache_length = self.get_keys(layer_idx).shape[2]
            block_length = key_states.shape[2]
            # 计算替换的起始位置，即 (当前缓存长度 - 新块的长度)
            start_pos = cache_length - block_length
            keys = self.get_keys(layer_idx).slice_scatter(key_states, dim=2, start=start_pos, end=cache_length)
            values = self.get_values(layer_idx).slice_scatter(val_states, dim=2, start=start_pos, end=cache_length)
        return keys, values


class KVCacheSnapshot:
    """KV Cache 快照，用于 MCMC 回滚机制
    
    在 MCMC 精炼过程中，如果提议被拒绝，需要回滚 KV Cache 到原始状态。
    此类保存指定区域的 KV Cache 副本，支持快速恢复。
    
    Parameters
    ----------
    block_start : int
        需要快照的区域起始位置
    block_end : int
        需要快照的区域结束位置
    """
    
    def __init__(self, block_start, block_end):
        self.block_start = block_start
        self.block_end = block_end
        self.snapshot_data = None  # 保存的 KV 数据
        self._saved = False
    
    def save(self, kv_cache_manager):
        """保存当前 KV Cache 状态
        
        Parameters
        ----------
        kv_cache_manager : DiffusionKVCacheManager
            KV Cache 管理器
        """
        if kv_cache_manager is None or kv_cache_manager.past_key_values is None:
            self.snapshot_data = None
            self._saved = False
            return
        
        # 确保 KV Cache 已经整合为张量格式
        kv_cache_manager.past_key_values.consolidate()
        
        # 只保存需要回滚的区域 [block_start, block_end)
        # KVCache._data 形状: [num_layers, 2, batch_size, num_heads, seq_len, hidden_dim]
        # 使用 clone() 确保是深拷贝
        kv_data = kv_cache_manager.past_key_values._data
        
        # 检查边界
        seq_len = kv_data.shape[4]
        actual_end = min(self.block_end, seq_len)
        actual_start = min(self.block_start, seq_len)
        
        if actual_start >= actual_end:
            self.snapshot_data = None
            self._saved = False
            return
        
        self.snapshot_data = kv_data[:, :, :, :, actual_start:actual_end, :].clone()
        self._saved = True
    
    def restore(self, kv_cache_manager):
        """恢复 KV Cache 到快照状态
        
        Parameters
        ----------
        kv_cache_manager : DiffusionKVCacheManager
            KV Cache 管理器
        """
        if not self._saved or self.snapshot_data is None:
            return
        
        if kv_cache_manager is None or kv_cache_manager.past_key_values is None:
            return
        
        # 将快照数据写回 KV Cache
        kv_data = kv_cache_manager.past_key_values._data
        seq_len = kv_data.shape[4]
        actual_end = min(self.block_end, seq_len)
        actual_start = min(self.block_start, seq_len)
        
        if actual_start >= actual_end:
            return
        
        # 恢复快照数据
        kv_cache_manager.past_key_values._data[:, :, :, :, actual_start:actual_end, :] = self.snapshot_data
    
    @property
    def is_saved(self):
        """检查是否已保存快照"""
        return self._saved


class DiffusionKVCacheManager:
    """ KV-cache for diffusion LLM.

    The KV-cache caches the KV of the tokens before and after the block that is being decoded.
    Because diffusion LLM uses bidirectional attention, the KV-cache has to be updated frequently in the diffusion iterations.
    This class basically defines the KV-cache update policy in the diffusion iterations. This includes the locations where
    keys and values can be updated and the frequency of the keys and values can be updated.

    """
    def __init__(self, cache_update_freq=None, cache_type='prefix', backend='vllm', max_length=2048):
        self.past_key_values = None
        self.block_start = None
        self.block_end = None
        self.cache_update_freq = cache_update_freq
        assert cache_type in ['prefix', 'dual']
        self.cache_type = cache_type
        self.backend=backend
        self.max_length = max_length

    def require_update(self, iter_no, block_start, block_end):
        """ 判断是否需要更新KV缓存。

        在扩散过程中，我们可能不需要在每一步都更新KV缓存。这个函数根据当前的迭代次数和正在处理的块来决定是否需要更新。

        Parameters
        ----------
        iter_no : int
            The diffusion iteration number
        block_start : int
            The start of the block that is being decoded.
        block_end : int
            The end of the block that is being decoded.
        """
        # 如果缓存尚未初始化，则必须更新
        if self.past_key_values is None:
            _require_update = True
        # 如果未指定更新频率 (cache_update_freq is None)，则仅当进入一个新的块时才更新缓存。
        # 这意味着在同一个块内的多次扩散迭代中，缓存保持不变。
        if self.cache_update_freq is None:
            _require_update = self.block_start != block_start or self.block_end != block_end
        else:
            # 如果指定了更新频率，则满足以下任一条件时更新：
            # 1. 达到了指定的迭代间隔 (iter_no % self.cache_update_freq == 0)
            # 2. 进入了一个新的块 (block_start 或 block_end 发生变化)
            _require_update = iter_no % self.cache_update_freq == 0 \
                    or (self.block_start != block_start or self.block_end != block_end)
        # TODO(zhengda) change update logic to block idx
        self.block_start = block_start
        self.block_end = block_end
        return _require_update

    def update(self, past_key_values, range_start=None, range_end=None):
        """ update the KV-cache

        Parameters
        ----------
        past_key_values : List[torch.Tensor] or KVCache
            新的KV缓存数据。可以是原始的张量列表，也可以是已经封装好的 KVCache 对象。
        range_start : int
            The start of the range that is being updated.
        range_end : int
            The end of the range that is being updated.
        """
        if isinstance(past_key_values, KVCache):
            self.past_key_values = past_key_values
        else:
            # 如果传入的是原始列表，则将其封装为 KVCache 对象
            self.past_key_values = KVCache(past_key_values, self.backend, self.max_length)
        # 确保所有层的KV缓存都被整合成高效的张量格式
        self.past_key_values.consolidate()
        

    def range_update(self, past_key_values, range_start=0, range_end=0, block_length=32):
        """ 更新指定范围内的KV缓存。

        这个方法用于部分更新缓存，通常用于滑动窗口或特定块的更新。

        Parameters
        ----------
        past_key_values : List[torch.Tensor]
            The key values in all transformer layers.
        range_start : int
            The start of the range that is being updated.
        range_end : int
            The end of the range that is being updated.
        """
        if isinstance(past_key_values, KVCache):
            # raise ValueError("past_key_values should be a list of tensors")
            self.past_key_values = past_key_values
        else:
            if block_length > 0:
                # 如果指定了 block_length，我们需要构造一个新的缓存。
                # 新缓存由两部分组成：
                # 1. 原始缓存中 [range_start, range_end - block_length] 的部分（保留的历史信息）
                # 2. 新传入的 past_key_values 的最后 block_length 个 token（最新生成的块信息）
                # 这种逻辑通常用于在生成新块后，丢弃旧块并追加新块。
                self.past_key_values = KVCache([torch.cat((kv[:, :, range_start:range_end-block_length], kv[:, :, -block_length:]), dim=2) for kv in past_key_values], self.backend)
            else:
                # 如果 block_length <= 0，则直接截取 past_key_values 中 [range_start, range_end] 的部分作为新缓存。
                self.past_key_values = KVCache([kv[:, :, range_start:range_end] for kv in past_key_values], self.backend)
        # We should make sure the kv-cache in all layers are converted into a tensor.
        self.past_key_values.consolidate()
        # print(self.past_key_values._data.shape)

    def get_key_values(self, block_start, block_end):
        """ Get the key-values given the block that is being decoded.

        Parameters
        ----------
        block_start : int
            The start of the block that is being decoded.
        block_end : int
            The end of the block that is being decoded.

        Returns
        -------
        List[List[torch.Tensor]] : the key-values required to decode the specified block.
        torch.Tensor : the tensor indicates the valid locations in the returned key-values.
        """
        # The key-value cache cannot be empty.
        assert self.past_key_values is not None
        if self.cache_type == 'prefix':
            # 如果是前缀缓存模式，替换位置是从当前块的开始一直到缓存的末尾。
            # 这意味着我们可能会追加或覆盖从 block_start 开始的所有内容。
            replace_position = (int(block_start), int(self.past_key_values.seq_len))
        else:
            # 如果是其他模式（如双向缓存），替换位置严格限制在当前块的范围内 [block_start, block_end]。
            replace_position = (int(block_start), int(block_end))
        return self.past_key_values, replace_position

class BlockDiffusionPrefixCacheManager(DiffusionKVCacheManager):
    """
     KVcache manager of block diffusion.
    """

    def get_key_values(self, block_start, block_end):
        # use prefix cache for block diffusion.
        return self.past_key_values, (int(block_start), int(block_end))

    def extend_cache(self, end):
        """
        When move to new block, extend the kvcache length from previous block end to new block end location.

        Parameters
        ----------
        length : int
            The extended length (equvelent to block length)
        
        """
        if self.backend == 'vllm':
            cur_kv_length = self.past_key_values._data.shape[-2]
            extended_cache = F.pad(self.past_key_values._data, pad=(0, 0, 0, end-cur_kv_length), mode='constant', value=0)
            self.past_key_values._data = extended_cache
        else:
            cur_kv_length = self.past_key_values._data.shape[-2]
            aligned_end = (end+self.past_key_values.cache_align_size-1)//self.past_key_values.cache_align_size*self.past_key_values.cache_align_size
            if aligned_end <= cur_kv_length:
                return
            extended_cache = F.pad(self.past_key_values._data, pad=(0, 0, 0, aligned_end-cur_kv_length), mode='constant', value=0)
            self.past_key_values._data = extended_cache


class KVCacheFactory:
    """ KV-cache factory.

    This class generates KV-cache for the diffusion LLM when it runs diffusion iterations.
    """
    def __init__(self, cache_type, cache_update_freq=None, is_bd_model=False, backend='vllm', max_length=2048):
        self.cache_type = cache_type
        self.cache_update_freq = cache_update_freq
        self.is_bd_model = is_bd_model
        self.backend = backend
        self.max_length = max_length

    def create(self):
        if self.is_bd_model:
            return BlockDiffusionPrefixCacheManager(cache_update_freq=self.cache_update_freq, cache_type=self.cache_type, backend=self.backend, max_length=self.max_length)
        else:
            return DiffusionKVCacheManager(cache_update_freq=self.cache_update_freq, cache_type=self.cache_type)

def gather_sequence_block(partial_data, partial_start, partial_end, block_start, block_end, rank, world_size):
    """ Gather the wanted block data from the partitioned data.

    Each process contains a partition specified by `partial_start` and `partial_end`.
    The wanted block is located between `block_start` and `block_end`.

    We want to gather the data within the block range from the partitioned data.
    """
    if partial_start >= block_end or partial_end <= block_start:
        # there is no overlap, nothing is needed from partial_data
        arr = partial_data[:, 0:0]
    elif block_start >= partial_start and block_end <= partial_end:
        # the needed block is within partial_data.
        arr = partial_data[:, (block_start - partial_start):(block_end - partial_start)]
    elif block_start <= partial_start and block_end >= partial_end:
        # the needed partition is within the block.
        arr = partial_data
    elif partial_start >= block_start and partial_end >= block_end:
        # the needed block is overlapped in the front of partial_data
        arr = partial_data[:, 0:(block_end - partial_start)]
    else:
        # the needed block is overlapped at the end of partial_data
        arr = partial_data[:, (block_start - partial_start):(partial_end - partial_start)]
    arr = arr.contiguous()

    shape_list = [
            torch.zeros(len(arr.shape), dtype=torch.int64, device=partial_data.device) for _ in range(world_size)
    ]
    dist.all_gather(shape_list, torch.tensor(arr.shape, dtype=torch.int64, device=partial_data.device))
    part_list = [
            torch.zeros(*tuple(shape.tolist()), dtype=partial_data.dtype, device=partial_data.device) for shape in shape_list
    ]
    dist.all_gather(part_list, arr)
    return torch.cat(part_list, dim=1)
