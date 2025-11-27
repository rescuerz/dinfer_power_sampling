from functools import partial
import math
import torch
import numpy as np
import torch.nn.functional as F

from .utils import add_gumbel_noise, get_num_transfer_tokens

@ torch.no_grad()
@ torch.compile(dynamic=True)
def get_transfer_index_hierarchy_fast_v2(logits, temperature, remasking, mask_index, x, num_transfer_tokens,  mask_id, threshold=None,  low_threshold = None):
    if not math.isclose(temperature, 0.0):
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    else:
        logits_with_noise = logits

    x0 = torch.argmax(logits_with_noise, dim=-1) # [batch, seq_len]

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float32), dim=-1)  # [batch, seq_len, vocab_size]
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # [batch, seq_len]
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    # 只在mask位置保留置信度，非mask位置设为-inf
    confidence = torch.where(mask_index, x0_p, -np.inf)
    

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

    # 模式1：固定数量解码
    if  num_transfer_tokens is not None:
        assert threshold is None
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
            transfer_index[j, select_index] = True

    # 模式2：层级解码（Hierarchy Decoding）
    else:
        for i in range (mask_index.shape[0]):
            mask_i = mask_index[i].int()  # 当前样本的mask索引
            conf_i = confidence[i]        # 当前样本的置信度

            # 特殊情况：如果最大置信度低于low_threshold，只转移最高置信度的token
            if low_threshold is not None:
                max_value, max_index = torch.max(conf_i, dim=0)
                if max_value < low_threshold:
                    transfer_index [i, max_index] = True
                    continue

            # 找出所有连续的mask区间
            # diff计算相邻位置的差值，1表示区间开始，-1表示区间结束
            diff = torch.diff(torch.cat([mask_i[:1]*0, mask_i, mask_i[-1:]*0]))
            starts = (diff == 1).nonzero(as_tuple=True)[0]  # 区间起始位置
            ends = (diff == -1).nonzero(as_tuple=True)[0]    # 区间结束位置

            # 在每个连续mask区间中选择置信度最高的token
            if len(starts) > 0:
                max_indices = [s + torch.argmax(conf_i[s:e]) for s, e in zip(starts.tolist(), ends.tolist())]
                transfer_index[i, max_indices] = True

            # 应用低阈值过滤：移除置信度低于low_threshold的token
            if low_threshold is not None:
                transfer_index [i] = torch.logical_and (transfer_index[i], conf_i > low_threshold)

        # 应用高阈值：置信度超过threshold的token直接转移（跨所有样本）
        if threshold is not None:
            transfer_index = torch.logical_or(transfer_index, confidence > threshold)

    return x0, transfer_index

@ torch.no_grad()
def get_transfer_index_hierarchy_remask(logits, temperature, mask_index, x, num_transfer_tokens,
                                         mask_id, threshold=None,  low_threshold = None, remask_threshold = 0.4):
    """
    层级解码策略（带重掩码机制）：对低置信度的已解码token重新掩码
    """
    if not math.isclose(temperature, 0.0):
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    else:
        logits_with_noise = logits

    x0 = torch.argmax(logits_with_noise, dim=-1) # [batch, seq_len]

    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # [batch, seq_len]

    # 识别需要重掩码的位置
    lower_index = x0_p < remask_threshold
    remask_index = torch.logical_and (lower_index, torch.logical_not(mask_index))  # 已解码但置信度低的位置
    mask_new = torch.logical_or (lower_index, mask_index)  # 扩展后的mask区间（原始mask + 重掩码）

    confidence = torch.where(mask_new, x0_p, float('-inf'))

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

    # 统计每个样本需要重掩码的token数量
    remask_cnt = remask_index.sum (dim = 1)

    # 模式1：固定数量解码
    if  num_transfer_tokens is not None:
        assert threshold is None
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
            transfer_index[j, select_index] = True

    # 模式2：层级解码
    else:
        for i in range (mask_new.shape[0]):

            mask_i = mask_new[i].int()
            conf_i = confidence[i]


            diff = torch.diff(torch.cat([mask_i[:1]*0, mask_i, mask_i[-1:]*0]))
            starts = (diff == 1).nonzero(as_tuple=True)[0]
            ends = (diff == -1).nonzero(as_tuple=True)[0]


            if len(starts) > 0:
                max_indices = [s + torch.argmax(conf_i[s:e]) for s, e in zip(starts.tolist(), ends.tolist())]
                transfer_index[i, max_indices] = True
            
            if low_threshold is not None:
                transfer_index [i] = torch.logical_and (transfer_index[i], conf_i > low_threshold) 
                
            if threshold is not None:
                transfer_index [i] = torch.logical_or(transfer_index [i], conf_i > threshold)

            # 关键：确保转移足够的token来填补重掩码的位置
            # gap = 需要重掩码的数量 + 1 - 已选择转移的数量
            gap = int((remask_cnt [i] + 1 - transfer_index [i].sum()).item())
            if gap > 0:
                # 从剩余位置中选择gap个置信度最高的token
                conf_i [transfer_index [i]] = float('-inf')  # 排除已选择的位置
                values, indices = torch.topk (conf_i, gap, largest=True, sorted=False)
                transfer_index [i][indices] = True

    # 处理重掩码：将需要重掩码但不转移的位置设为mask_id
    remask_index = torch.logical_and (remask_index, torch.logical_not (transfer_index))
    x0 [remask_index] = mask_id
    transfer_index [remask_index] = True  # 重掩码的位置也标记为转移（因为值被改变了）

    return x0, transfer_index


def get_transfer_index_cache (logits, mask_index, x, block_end, num_transfer_tokens, temperature, remasking, threshold=None, minimal_topk=1):

    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits[mask_index].to(torch.float32), dim=-1).to(logits.dtype)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0[mask_index], -1)), -1)  # b, l
        confidence = torch.full(x0.shape, -np.inf, device=x0.device, dtype=logits.dtype)
        confidence[mask_index] = x0_p
        confidence[:, block_end:] = -np.inf

    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        x0_p[:, block_end:] = -np.inf
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)
    else:
        raise NotImplementedError(remasking)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    # print("num_transfer_tokens, topk",num_transfer_tokens[0], minimal_topk)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(minimal_topk, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

class ParallelDecoder:
    """ This is a parallel decoder that decodes tokens in a block.
    """
    def __init__(self, temperature, remasking='low_confidence', mask_id=126336):
        self.temperature = temperature
        self.remasking = remasking
        self.mask_id = mask_id

    def block_init(self, block_x, block_id):
        pass

    def decode(self, logits, block_start, block_end, x):
        """ Decode the logits in a block.

        Parameters
        ----------
        logits : Tensor
            The logits in a block
        block_start : int
            The location of the starting token in the block
        block_end : int
            The location of the ending token in the block.
        x : Tensor
            The tensor where the decoded tokens are written to.
        """

# Parallel decoding only
@ torch.compile(dynamic=True)
def get_transfer_index_threshold(logits, temperature, mask_index, x, mask_id,
        threshold, rm_mask=True, use_float64=False, **kwargs):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if use_float64:
        p = F.softmax(logits.to(torch.float64), dim=-1)
    else:
        p = F.softmax(logits.to(torch.float32), dim=-1)
    x0_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    
    # gurantee the denoised token will not be the mask_id   
    if rm_mask:
        mask_index = mask_index & (x0 != mask_id)
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    actual_threshold = (torch.max(confidence, dim=1)[0]-1e-5).clamp(-1000, threshold).unsqueeze(-1)
    transfer_index = confidence >= actual_threshold
    return x0, transfer_index


# Power-scaled version for MCMC proposal generation
@ torch.compile(dynamic=True)
def get_transfer_index_threshold_power(logits, temperature, mask_index, x, mask_id,
        threshold, alpha=1.0, rm_mask=True, use_float64=False, **kwargs):
    """
    Power-scaled threshold decoding for MCMC proposal generation.
    
    Key difference from get_transfer_index_threshold:
    - Token selection (argmax) uses power-scaled logits: add_gumbel_noise_power(logits, alpha)
    - Confidence calculation uses ORIGINAL logits: softmax(logits)
    
    This ensures:
    - Proposal distribution q(x'|x) is more concentrated on high-probability tokens
    - But probability calculation remains consistent with the original distribution
    
    Parameters
    ----------
    logits : torch.Tensor
        Model logits [batch, seq_len, vocab_size]
    temperature : float
        Temperature for Gumbel noise
    mask_index : torch.Tensor
        Boolean mask indicating positions to decode [batch, seq_len]
    x : torch.Tensor
        Current token sequence [batch, seq_len]
    mask_id : int
        ID of mask token
    threshold : float
        Confidence threshold for transfer
    alpha : float
        Power parameter for proposal distribution (default: 1.0 means no power scaling)
    rm_mask : bool
        Whether to ensure decoded tokens are not mask_id
    use_float64 : bool
        Whether to use float64 for softmax
    
    Returns
    -------
    x0 : torch.Tensor
        Decoded tokens [batch, seq_len]
    transfer_index : torch.Tensor
        Boolean mask indicating which tokens to transfer [batch, seq_len]
    """
    from .utils import add_gumbel_noise_power
    
    # Step 1: Use power-scaled Gumbel noise for token selection
    # This makes the proposal distribution more concentrated on high-probability tokens
    logits_with_noise = add_gumbel_noise_power(logits, alpha=alpha, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    
    # Step 2: Calculate confidence using ORIGINAL logits (not power-scaled)
    # This ensures q(x'|x) and q(x|x') are computed consistently
    if use_float64:
        p = F.softmax(logits.to(torch.float64), dim=-1)
    else:
        p = F.softmax(logits.to(torch.float32), dim=-1)
    x0_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
    
    # Step 3: Ensure decoded tokens are not mask_id
    if rm_mask:
        mask_index = mask_index & (x0 != mask_id)
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)
    
    # Step 4: Apply threshold
    actual_threshold = (torch.max(confidence, dim=1)[0]-1e-5).clamp(-1000, threshold).unsqueeze(-1)
    transfer_index = confidence >= actual_threshold
    return x0, transfer_index

class ThresholdParallelDecoder(ParallelDecoder):
    """ This decoder deocdes tokens in parallel based on a threshold.

    The decoder decodes a token when its confidence score is larger than a threshold.
    """
    def __init__(self, temperature, threshold, remasking='low_confidence', mask_id=126336, eos_id=126081,
            use_float64=False):
        super().__init__(temperature, remasking, mask_id)
        self.threshold = threshold
        self.eos_id = eos_id
        self.use_float64 = use_float64

    def decode(self, logits, block_start, block_end, x, iter_threshold = None):
        """ Decode the logits in a block.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = get_transfer_index_threshold(logits, self.temperature, mask_index, curr_x,
                self.mask_id, threshold=iter_threshold, use_float64=self.use_float64)
        transfer_index = torch.logical_and(transfer_index, mask_index)
        assert transfer_index.dtype == torch.bool
        x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)

class MCMCThresholdParallelDecoder(ThresholdParallelDecoder):
    """MCMC专用的阈值解码器，返回置信度以支持Power Sampling"""
    
    # 调试开关
    DEBUG_MCMC_DECODER = False

    def decode(self, logits, block_start, block_end, x, mcmc_alpha=1.0, iter_threshold=None, proposal_alpha=1.0):
        """解码并返回双重置信度

        Parameters
        ----------
        logits : torch.Tensor
            Model logits
        block_start : int
            Block start position
        block_end : int
            Block end position
        x : torch.Tensor
            Token array
        mcmc_alpha : float
            Power parameter for target distribution p^α (default: 1.0)
        iter_threshold : float
            Iteration-specific threshold (default: None, uses self.threshold)
        proposal_alpha : float
            Power parameter for proposal distribution (default: 1.0)
            - proposal_alpha=1.0: standard decoding (original sequence)
            - proposal_alpha>1.0: power-scaled decoding (proposal sequence)

        Returns
        -------
        confidences_norm : torch.Tensor
            Normalized log probabilities (alpha=1.0)
        confidences_unnorm : torch.Tensor
            Unnormalized log probabilities (alpha=mcmc_alpha)
        """
        if iter_threshold is None:
            iter_threshold = self.threshold

        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]
        curr_x = x[:, block_start:block_end]
        
        # DEBUG: 记录 mask 数量
        num_masks_before = mask_index.sum().item()

        # 根据 proposal_alpha 选择合适的解码函数
        if proposal_alpha == 1.0:
            # 原始序列：使用标准阈值解码
            x0, transfer_index_raw = get_transfer_index_threshold(
                logits, self.temperature, mask_index, curr_x,
                self.mask_id, threshold=iter_threshold, use_float64=self.use_float64
            )
        else:
            # 提议序列：使用 power-scaled 解码
            x0, transfer_index_raw = get_transfer_index_threshold_power(
                logits, self.temperature, mask_index, curr_x,
                self.mask_id, threshold=iter_threshold, alpha=proposal_alpha,
                use_float64=self.use_float64
            )
        
        # DEBUG: 记录 transfer_index 数量（在 AND mask_index 之前）
        num_transfer_raw = transfer_index_raw.sum().item()

        # 计算双重置信度（始终使用原始 logits，不受 proposal_alpha 影响）
        log_p_norm = F.log_softmax(logits, dim=-1)
        log_p_unnorm = F.log_softmax(mcmc_alpha * logits, dim=-1)

        x0_logp_norm = torch.gather(log_p_norm, -1, x0.unsqueeze(-1)).squeeze(-1)
        x0_logp_unnorm = torch.gather(log_p_unnorm, -1, x0.unsqueeze(-1)).squeeze(-1)

        # 只在 transfer_index_raw 位置保存置信度（这是关键！）
        # 注意：这里使用 transfer_index_raw，而不是 AND mask_index 之后的结果
        confidences_norm = torch.full_like(x0, -np.inf, dtype=torch.float32)
        confidences_unnorm = torch.full_like(x0, -np.inf, dtype=torch.float32)

        confidences_norm[transfer_index_raw] = x0_logp_norm[transfer_index_raw].float()
        confidences_unnorm[transfer_index_raw] = x0_logp_unnorm[transfer_index_raw].float()
        
        # DEBUG: 记录置信度更新数量
        num_conf_updated = (confidences_norm > -np.inf).sum().item()

        # 更新 x（只在 mask 位置更新）
        transfer_index = torch.logical_and(transfer_index_raw, mask_index)
        assert transfer_index.dtype == torch.bool
        x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)
        
        # DEBUG: 记录实际转移数量
        num_transfer_final = transfer_index.sum().item()
        
        # DEBUG: 检查更新后的 mask 数量
        mask_index_after = (x[:, block_start:block_end] == self.mask_id)
        num_masks_after = mask_index_after.sum().item()
        
        if self.DEBUG_MCMC_DECODER:
            print(f"[MCMCDecoder] block=[{block_start},{block_end}), "
                  f"masks: {num_masks_before}->{num_masks_after}, "
                  f"transfer_raw={num_transfer_raw}, transfer_final={num_transfer_final}, "
                  f"conf_updated={num_conf_updated}, proposal_alpha={proposal_alpha}")

        return confidences_norm, confidences_unnorm

class CreditThresholdParallelDecoder(ThresholdParallelDecoder):
    """ This decoder deocdes tokens in parallel based on a threshold + credit.

    The decoder decodes a token when its confidence is larger than a threshold.
    """
    def __init__(self, 
                 credit_alpha=0.7, 
                 boost_gamma=0.2, 
                 decay_beta=0.8,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.credit_alpha = credit_alpha
        self.boost_gamma = boost_gamma
        self.decay_beta = decay_beta

        self._credit_mats = {}   
        self._credit_iters = {}  

    def _apply_credit_fusion(self, logits, mask_index, key):
        """
        EMA-based credit fusion (no CM, no pre-credit):
        - Maintains a per-block CreditMatrix (EMA with decay).
        - Accumulates enhanced top-1 probability only on masked positions.
        - Returns fused_logits.
        """
        B, L, V = logits.shape
        device = logits.device

        mat = self._credit_mats.get(key, None)
        if mat is None or mat.shape != (B, L, V) or mat.device != device:
            mat = torch.zeros((B, L, V), dtype=torch.float32, device=device)
            self._credit_mats[key] = mat
            self._credit_iters[key] = 0

        iter_idx = self._credit_iters[key]

        if iter_idx > 0:
            mat.mul_(self.decay_beta)

        probs = F.softmax(logits.to(torch.float32), dim=-1)
        top1_probs, top1_idx = torch.max(probs, dim=-1)         
        enhanced = top1_probs.pow(self.boost_gamma).to(mat.dtype)  
        update_vals = enhanced * mask_index.to(enhanced.dtype)     
        mat.scatter_add_(2, top1_idx.unsqueeze(-1), update_vals.unsqueeze(-1))

        fused_logits = logits + self.credit_alpha * torch.log(mat + 1)
        self._credit_iters[key] = iter_idx + 1
        return fused_logits

    def decode(self, logits, block_start, block_end, x, iter_threshold = None):
        """ Decode the logits in a block.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        key = (block_start, block_end)
        used_logits = self._apply_credit_fusion(logits, mask_index, key)

        x0, transfer_index = get_transfer_index_threshold(used_logits, self.temperature, mask_index, curr_x,
                self.mask_id, threshold=iter_threshold, use_float64=self.use_float64)

        transfer_index = torch.logical_and(transfer_index, mask_index)
        assert transfer_index.dtype == torch.bool
        x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)

        if hasattr(x, 'data'):
            has_mask = (x.data == self.mask_id).any()
        else:
            has_mask = (x == self.mask_id).any() if x.dim() > 0 else (x == self.mask_id)

        if not has_mask:
            self._credit_mats.clear()
            self._credit_iters.clear()

class FixedParallelDecoder(ParallelDecoder):
    """ This decoder decodes tokens in a fixed number of steps.
    """
    def __init__(self, temperature, steps, remasking='low_confidence', mask_id=126336):
        super().__init__(temperature, remasking, mask_id)
        self.steps = steps
        self.iter = 0

    def block_init(self, block_x, block_id):
        # TODO(zhengda) we need to handle steps correctly here when the distributed version changes the gen length.
        block_mask_index = block_x == mask_id
        self.num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        self.iter = 0

    def decode(self, logits, block_start, block_end, x, iter_threshold = None):
        """ Decode the logits in a block.
        """
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = get_transfer_index(logits, self.temperature, self.remasking, mask_index, curr_x, self.num_transfer_tokens[:, self.iter], None)
        self.iter += 1
        x[:, block_start:block_end][transfer_index] = x0[transfer_index]


class HierarchyDecoder(ParallelDecoder):
    """ This decoder decodes tokens in a hierarchy way. Forcing LLMs to decode tokens seperately.
    """
    def __init__(self, temperature, remasking='low_confidence',
                mask_id=126336,  eos_id=126081, 
                threshold=None, low_threshold=0.4):
        super().__init__(temperature, remasking, mask_id)
        self.iter = 0
        self.mask_id = mask_id
        self.eos_id=eos_id
        self.threshold=threshold
        self.low_threshold=low_threshold

    def get_transfer_index(self, logits,  mask_index, iter_threshold, **kwargs):
    
        B, L = mask_index.shape

        # TODO(DuLun): support batch size > 1
        assert B == 1

        device = logits.device
        
        if not math.isclose(self.temperature, 0.0):
            logits_with_noise = add_gumbel_noise(logits, temperature=self.temperature)
        else:
            logits_with_noise = logits

        x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
        
        x0_logp = F.log_softmax(logits, dim=-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        x0_p = x0_logp.exp()  # b, l

        neg_inf_val = torch.finfo(x0_p.dtype).min
        confidence = torch.where(mask_index, x0_p, torch.tensor(neg_inf_val, device=device, dtype=x0_p.dtype))
        
        prev = torch.cat(
            [mask_index.new_zeros((B, 1), dtype=torch.bool), mask_index[:, :-1]],
            dim=1
        )
        starts = torch.logical_and(mask_index, torch.logical_not(prev))

        seg_id = torch.cumsum(starts.to(torch.int64), dim=-1) - 1
        seg_id = torch.where(mask_index, seg_id, 0)

        seg_max = torch.full((B, L), neg_inf_val, device=device, dtype=confidence.dtype)
        seg_max = torch.scatter_reduce(seg_max, dim=1, index=seg_id, src=confidence, reduce='amax', include_self=True)

        seg_max_at_pos = seg_max.gather(dim=1, index=seg_id)
        transfer_index = (confidence == seg_max_at_pos)

        if self.low_threshold is not None:
            transfer_index = torch.logical_and(transfer_index, torch.gt(confidence, self.low_threshold))
        if iter_threshold is not None:
            transfer_index = torch.logical_or(transfer_index, torch.gt(confidence, iter_threshold))

        
        top1_idx = torch.argmax(confidence, dim=-1)
        top1 = torch.nn.functional.one_hot(top1_idx, num_classes=L).to(torch.bool)
        transfer_index = torch.logical_or(transfer_index, top1)
        

        return x0, transfer_index

    def block_init(self, block_x, block_id):
        # TODO(zhengda) we need to handle steps correctly here when the distributed version changes the gen length.
        block_mask_index = block_x == self.mask_id
        self.iter = 0

    def decode(self, logits, block_start, block_end, x, iter_threshold = None):
        """ Decode the logits in a block.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = self.get_transfer_index(logits, mask_index, iter_threshold)
        self.iter += 1
        transfer_index = torch.logical_and(transfer_index, mask_index)
        x[:, block_start:block_end][transfer_index] = x0[transfer_index]
