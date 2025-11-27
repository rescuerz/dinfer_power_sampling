import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.distributed as dist
import time
import tqdm
from vllm.config import CompilationConfig, ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config
from vllm.forward_context import set_forward_context
import json
from dinfer.model import LLaDAMoeModelLM, LLaDAModelLM, LLaDA2MoeModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import ThresholdParallelDecoder,CreditThresholdParallelDecoder, HierarchyDecoder, MCMCThresholdParallelDecoder
from dinfer import BlockWiseDiffusionLLM, IterSmoothDiffusionLLM, VicinityCacheDiffusionLLM, IterSmoothWithVicinityCacheDiffusionLLM, BlockDiffusionLLM, BlockMCMCDiffusionLLM
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    print(f'rank={rank}, world size={world_size}')
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# 桶大小，用于序列长度对齐
bucket_size = 32
# 记录已使用的桶长度
used_buckets = []

def get_bucket_length(length):
    """
    计算对齐到桶大小的序列长度
    """
    # 将长度向下取整到bucket_size的倍数
    bucket_length = bucket_size*(length//bucket_size)
    # 记录使用过的桶长度，用于后续的CUDA图预热
    if bucket_length not in used_buckets:
        used_buckets.append(bucket_length)
    return bucket_length

def load_inputs(dataset, tokenizer):
    with open(dataset, 'r') as f:
        data = json.load(f)
    prompts = []
    questions = []
    ids = []
    all_input_ids = []
    
    if isinstance(data, list):
        # 格式2: 列表格式 [{"question": "...", "answer": "..."}, ...]
        details_data = data
        for id, item in enumerate(details_data):
            ids.append(id)
            # 使用question字段作为prompt
            prompt = item.get('question', item.get('prompt', ''))
            prompts.append(prompt)
            questions.append(prompt)
            # 添加角色标签和格式化prompt
            formatted_prompt = '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>'+prompt+'<|role_end|><role>ASSISTANT</role>'
            
            input_ids = tokenizer(formatted_prompt)['input_ids']  # List[int], shape: [seq_len]
            input_ids = torch.tensor(input_ids).unsqueeze(0)  # Tensor, shape: [1, seq_len]
            all_input_ids.append(input_ids)
    else:
        # 格式1: 字典格式 {"judge_details": [...]} 或 {"details": [...]}
        if "judge_details" in data.keys():
            details_data = data['judge_details']
        else:
            details_data = data['details']
        
        for id, judge_detail in enumerate(details_data):
            ids.append(id)
            prompt = judge_detail['prompt']
            prompts.append(prompt)
            questions.append(prompt)
            # 添加角色标签和格式化prompt
            formatted_prompt = '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>'+prompt+'<|role_end|><role>ASSISTANT</role>'
            
            input_ids = tokenizer(formatted_prompt)['input_ids']  # List[int], shape: [seq_len]
            input_ids = torch.tensor(input_ids).unsqueeze(0)  # Tensor, shape: [1, seq_len]
            all_input_ids.append(input_ids)
    
    return all_input_ids, prompts, questions, ids

def cal_bucket_len(args, all_input_ids):
    """
    计算每个输入序列对齐后的生成长度
    
    对于每个序列：
    1. 计算目标总长度 = 输入长度 + gen_len
    2. 将总长度向下对齐到bucket_size(32)的倍数，得到padded_length
    3. 计算实际生成的长度 = padded_length - 输入长度
    """
    max_prompt_length = 0
    gen_len = args.gen_len  # 目标生成长度（如1024）
    padded_gen_lens = []
    for i in range(len(all_input_ids)):
        input_ids = all_input_ids[i]
        if input_ids.shape[1] > max_prompt_length:
            max_prompt_length = input_ids.shape[1]
        # 计算对齐后的总长度：将(输入长度+gen_len)向下对齐到32的倍数
        padded_length = get_bucket_length(input_ids.shape[1]+gen_len)
        padded_gen_lens.append(padded_length - input_ids.shape[1])
    return padded_gen_lens

def warmup_cudagraph(rank, device, dllm, args):
    """
    预热CUDA图以优化性能
    通过对不同长度的序列进行预生成，让CUDA图缓存这些计算图，从而在实际推理时提高性能
    """
    batch_size = args.batch_size
    if rank==0:
        print('warmup')
        print(used_buckets)
        iterator = tqdm.tqdm(used_buckets)
    else:
        iterator = used_buckets
    offset = 0
    # 对每个使用过的桶长度进行预热
    for i in iterator:
        input_ids = torch.randint(0, 140000, (batch_size, i - args.gen_len+offset), dtype=torch.long, device=device)
        dllm.generate(input_ids, gen_length=args.gen_len, block_length=args.block_length)

def cut_eos(data, eos_id=156892):
    """
    在遇到EOS（结束符）token时截断序列
    """
    # 找到所有EOS token的位置
    eos_indices = (data[0] == eos_id).nonzero(as_tuple=True)[0]
    if eos_indices.numel() > 0:
        # 在第一个EOS位置截断
        first_eos_idx = eos_indices[0].item()
        return data[:, :first_eos_idx]
    else:
        # 如果没有EOS，返回原序列
        return data

@ torch.no_grad()
def main(world_size, rank, gpu_id, args):
    print('started', world_size, rank, gpu_id, args)
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    all_input_ids, prompts, questions, ids = load_inputs(args.dataset, tokenizer)
    padded_gen_lens = cal_bucket_len(args, all_input_ids)

    block_length=args.block_length
    dataset_name = args.dataset.split('/')[-1]
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化vLLM分布式环境
    from vllm import distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(45601+args.port_offset)
    distributed.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl')
    distributed.initialize_model_parallel(args.tp_size, backend='nccl')
    print("[Loading model]")
    
    # 设置专家并行（Expert Parallel, EP）
    parallel_config = ParallelConfig(enable_expert_parallel = True)
    with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
        vllm_config = get_current_vllm_config()
        print("EP Enabled:", vllm_config.parallel_config.enable_expert_parallel)

        # 根据模型类型加载相应的模型
        model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        if args.model_type=='llada_moe':
            # LLaDA MoE模型
            model = LLaDAMoeModelLM(config=model_config).eval()
            model.load_weights(args.model_name, torch_dtype=torch.bfloat16)
            mask_id = 156895  # mask token ID
            eos_id = 156892   # 结束符token ID
        elif args.model_type=='llada2':
            # LLaDA2模型（mini或flash版本）
            model = LLaDA2MoeModelLM(config=model_config).eval()
            model.load_weights(args.model_name, torch_dtype=torch.bfloat16, device=device)
            mask_id = 156895
            eos_id = 156892
        elif args.model_type=='llada':
            # LLaDA或LLaDA1.5模型
            model = LLaDAModelLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, init_device=str(device)).eval()
            mask_id = 126336
            eos_id = 126081
        else:
            raise ValueError('model type not supported')
        
        # 启用张量并行（如果需要）
        if args.tp_size>1 and args.use_tp:
            print('enabling tp')
            model.tensor_parallel(args.tp_size)
        
        # 预热模型
        x = torch.arange(50+args.gen_len, dtype=torch.long, device=device).unsqueeze(0)
        model = model.to(device)
        out = model(x, use_cache=False)  # 不使用缓存的前向传播
        out = model(x, use_cache=True)   # 使用缓存的前向传播
        # 使用torch.compile编译模型以提高性能
        model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=False, dynamic=True)

        # 根据解码策略选择解码器
        if args.parallel_decoding == 'threshold':
            if args.use_credit:
                decoder = CreditThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=mask_id, eos_id=eos_id)
            else:
                decoder = ThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=mask_id, eos_id=eos_id)
        elif args.parallel_decoding == 'mcmc_threshold':
            decoder = MCMCThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=mask_id, eos_id=eos_id)
        else:
            decoder = HierarchyDecoder(temperature=0, threshold=args.threshold, low_threshold=args.low_threshold, mask_id=mask_id, eos_id=eos_id)
        use_sw = args.prefix_look > 0 or args.after_look > 0 or args.warmup_times > 0
            
        if args.cache == 'prefix' or args.cache == 'dual':
            cache_factory=KVCacheFactory(args.cache, is_bd_model=args.use_bd)
        else:
            cache_factory=None

        if args.parallel_decoding == 'mcmc_threshold':
            dllm = BlockMCMCDiffusionLLM(
                model=model, 
                decoder=decoder, 
                iterator_factory=BlockIteratorFactory(True),
                cache_factory=cache_factory,  # 支持 KV Cache
                enable_mcmc=True,
                n_mcmc_steps=args.n_mcmc_steps,
                mcmc_alpha=args.mcmc_alpha,
                mcmc_temperature=args.mcmc_temperature,
                mcmc_use_kv_cache=args.mcmc_use_kv_cache,
                tokenizer=tokenizer,
                verbose=False
            )
        else:
            # 根据配置选择扩散语言模型类型
            if not args.use_bd:
                # 不使用块扩散（Block Diffusion）
                if args.cont_weight>0:
                    # 使用连续性权重的迭代平滑
                    if use_sw:
                        # 带滑动窗口和邻近缓存的迭代平滑扩散模型
                        dllm = IterSmoothWithVicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True,
                            cont_weight=args.cont_weight, prefix_look=args.prefix_look, after_look=args.after_look, warmup_steps=args.warmup_times)
                    else:
                        # 标准迭代平滑扩散模型
                        dllm = IterSmoothDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, cont_weight=args.cont_weight)
                else:
                    # 不使用连续性权重
                    if use_sw:
                        # 带邻近缓存的扩散模型
                        dllm = VicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True,
                            prefix_look=args.prefix_look, after_look=args.after_look, warmup_steps=args.warmup_times)
                    else:
                        # 标准块级扩散模型
                        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, use_shift=args.use_shift)
            else:
                # 使用块扩散模型
                dllm = BlockDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True, use_block_diffusion=True), cache_factory=cache_factory, early_stop=True)
        
        # 执行CUDA图预热
        batch_size = args.batch_size
        warmup_cudagraph(rank, device, dllm, args)

        # 开始推理循环（这里只执行一次，wi从0到0）
        for wi in range(1):
            outputs = []  # 存储所有输出序列
            total_forward = 0  # 累计前向传播次数
            
            # 只在主进程（rank==0）显示进度条
            if rank==0:
                iterator = tqdm.trange(0, len(all_input_ids), batch_size)
            else:
                iterator = range(0, len(all_input_ids), batch_size)
            
            start = time.time()  # 记录开始时间
            # 初始化性能指标列表
            tpfs = []  # tokens per forward - 每次前向传播生成的平均token数
            tpss = []  # tokens per second - 每秒生成的token数
            fpss = []  # forwards per second - 每秒前向传播次数
            total_token = 0  # 累计生成的token总数
            token_numbers = []  # 每个样本生成的token数量
            
            # 遍历所有批次进行推理
            for i in iterator:
                # 获取当前批次的输入序列
                input_ids = all_input_ids[i:i+batch_size]
                max_length = 0  # 当前批次中最长输入序列的长度
                min_padded_length = 10000  # 当前批次中最小的生成长度
                
                # 找到批次中输入最长的序列，并获取其对应的生成长度
                # 为什么输入最长的序列对应的padded_gen_lens最小？
                # 因为在批次内，虽然每个序列的padded_length不同，但当输入长度增加时：
                # - padded_length增加的幅度小于输入长度的增加（因为向下对齐）
                # - 所以 padded_gen_lens = padded_length - 输入长度 会减小
                # 使用最小的生成长度可以确保批次内所有序列都不会超出其对齐长度
                for j, seq in enumerate(input_ids):
                    if seq.shape[1] > max_length:
                        max_length = seq.shape[1]  # 更新最长输入长度
                        min_padded_length = padded_gen_lens[i+j]  # 获取对应的生成长度
                
                # 创建填充后的批次输入张量，用mask_id(156895)填充
                batch_input_ids= torch.zeros((len(input_ids), max_length), dtype=torch.long, device=device).fill_(156895)
                # 将实际的输入序列从左侧（开头）开始复制到填充张量中
                # 因此mask_id(156895)会填充在右侧（序列末尾），这是左对齐填充方式
                for j in range(len(input_ids)):
                    batch_input_ids[j, :input_ids[j].shape[1]] = input_ids[j].to(device)
                input_ids = batch_input_ids
                
                padded_gen_len = padded_gen_lens[i]
                inner_start = time.time()  # 记录单个批次开始时间
                prev_forwards = dllm.num_forwards  # 记录之前的前向传播次数
                
                # 记录 MCMC 相关的前向传播次数（用于计算当前 sample 的增量）
                prev_diff_forwards = 0
                prev_prop_forwards = 0
                if args.parallel_decoding == 'mcmc_threshold' and hasattr(dllm, 'diff_iteration') and hasattr(dllm, 'proposal_generator'):
                    if dllm.proposal_generator is not None:
                        prev_diff_forwards = dllm.diff_iteration.num_forwards
                        prev_prop_forwards = dllm.proposal_generator.num_forwards
                
                out = dllm.generate(input_ids, gen_length=min_padded_length, block_length=block_length)
                
                nfe = dllm.num_forwards - prev_forwards  # 计算本批次的前向传播次数（Number of Function Evaluations）
                
                # 计算当前 sample 的 MCMC 前向传播增量
                sample_diff_forwards = 0
                sample_prop_forwards = 0
                if args.parallel_decoding == 'mcmc_threshold' and hasattr(dllm, 'diff_iteration') and hasattr(dllm, 'proposal_generator'):
                    if dllm.proposal_generator is not None:
                        sample_diff_forwards = dllm.diff_iteration.num_forwards - prev_diff_forwards
                        sample_prop_forwards = dllm.proposal_generator.num_forwards - prev_prop_forwards
                inner_stop = time.time()  # 记录单个批次结束时间
                sample_time = inner_stop - inner_start  # 计算本批次耗时
                
                # 保存每个样本的输出
                for j in range(input_ids.shape[0]):
                    outputs.append(out[j].unsqueeze(0))
                total_forward += nfe  # 累加前向传播次数
                
                # 计算本批次生成的token数量
                batch_token_number = 0
                for j in range(input_ids.shape[0]):
                    # 计算生成的token数：总token数减去输入token数，排除EOS(156892)之后的部分
                    token_number = int((out[j]!=156892).sum() - all_input_ids[i+j].shape[1])
                    batch_token_number += token_number
                    token_numbers.append(token_number)
                
                # 计算性能指标
                tpf = batch_token_number/nfe/batch_size
                tps = batch_token_number/sample_time
                fps = nfe/sample_time

                if rank == 0:
                    # 基础性能指标
                    print(f'[iter {i:4d}]nfe={nfe:4d}, token number={batch_token_number:4d}, fps={fps:4.2f},tpf={tpf:2.2f}, tps={tps:4.2f}')
                    
                    # MCMC 特有的性能指标
                    if args.parallel_decoding == 'mcmc_threshold' and hasattr(dllm, 'diff_iteration') and hasattr(dllm, 'proposal_generator'):
                        if dllm.proposal_generator is not None:
                            diff_forwards = dllm.diff_iteration.num_forwards
                            prop_forwards = dllm.proposal_generator.num_forwards
                            # 打印当前 sample 的增量和累积值
                            print(f'         [MCMC] sample: diff={sample_diff_forwards}, prop={sample_prop_forwards} | total: diff={diff_forwards}, prop={prop_forwards}')
                    
                    # 在前5个批次打印生成的文本样例
                    if wi==0 and i<5:
                        for j in range(input_ids.shape[0]):
                            # 截取生成的部分（去除输入部分）并在EOS处截断
                            answer = cut_eos(out[j, all_input_ids[i+j].shape[1]:].unsqueeze(0))[0]
                            print(f'generated text {j}: {tokenizer.decode(answer, skip_special_tokens=False)}')
                tpfs.append(tpf)
                tpss.append(tps)
                fpss.append(fps)
                total_token += token_number

            total_token = total_token
            stop = time.time()
        
        if rank==0:
            answers = []
            for i in tqdm.trange(len(outputs)):
                out = outputs[i]
                # 解码生成的部分（跳过输入部分），移除特殊token
                answer = (tokenizer.decode(out[0, all_input_ids[i].shape[1]:], skip_special_tokens=True))
                answers.append(answer)
            print(f'Forward: {total_forward}, Time: {stop-start}, FPS: {total_forward/(stop-start)}({np.mean(fpss)}), TPS: {total_token/(stop-start)}({np.mean(tpss)}), TPF: {total_token/total_forward}({np.mean(tpfs)})')
            filename = args.output_dir+'/'+'_'.join([str(item) for item in [args.exp_name, dataset_name, args.config, args.parallel_decoding, args.threshold, args.prefix_look]])+'.jsonl'
            
            # 保存详细结果到JSONL文件（每行一个JSON对象）
            with open (filename, 'w') as f:
                for i in range(len(answers)):
                    question = questions[i]
                    prompt = prompts[i]
                    answer = answers[i]
                    id = ids[i]
                    # 保存每个样本的完整信息：问题、提示词、答案、生成长度和性能指标
                    json.dump({'id':id, 'question':question, 'prompt':prompt, 'answer': answer, 'generated_length': token_numbers[i], 'tpf':tpfs[i//batch_size], 'tps':tpss[i//batch_size], 'fps':fpss[i//batch_size], }, f, indent=4)
                    f.write('\n')
            
            # 追加汇总结果到results.txt文件
            # 格式：实验名 配置ID 解码策略 阈值 前缀窗口 总前向数 总时间 平均生成长度 FPS TPS TPF 平均填充长度比 平均FPS 平均TPS 平均TPF 数据集路径
            with open('results.txt', 'a+') as f:
                print(args.exp_name, args.config, args.parallel_decoding, args.threshold, args.prefix_look, total_forward, stop-start, total_token / len(all_input_ids), total_forward/(stop-start), total_token/(stop-start), total_token/total_forward, sum(padded_gen_lens)/total_forward, np.mean(fpss), np.mean(tpss), np.mean(tpfs), args.dataset, file=f)

def process_args(args):
    import warnings
    gpus = [int(gpu) for gpu in args.gpu.split(',')]
    if len(gpus) > 1 and not args.use_tp:
        warnings.warn('Using multiple GPUs without tensor parallelism is not recommended. TP will be enabled.')
    elif len(gpus) == 1 and args.use_tp:
        warnings.warn('Using tensor parallelism with only one GPU is not accepted. TP will be disabled.')
    
    if args.model_type == 'llada2' and not args.use_bd:
        warnings.warn('Using llada2 without block diffusion is not recommended.')

    args.tp_size = len(gpus)
    args.use_tp = args.tp_size > 1
    args.port_offset = gpus[0]

    return args

from multiprocessing import Process
import argparse

if __name__ == '__main__':
    # 设置多进程启动方法为spawn（Windows和CUDA兼容）
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gen_len', type=int, default=1024)
    parser.add_argument('--prefix_look', type=int, default=0)
    parser.add_argument('--after_look', type=int, default=0)
    parser.add_argument('--block_length', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--warmup_times', type=int, default=0)
    parser.add_argument('--low_threshold', type=float, default=0.3)
    parser.add_argument('--cont_weight', type=float, default=0)
    parser.add_argument('--parallel_decoding', type=str, default='threshold')
    parser.add_argument('--use_credit', action='store_true')
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--cache', type=str, default='')
    parser.add_argument('--use_tp', action='store_true')
    parser.add_argument('--output_dir', type=str, default='/ossfs/workspace/detailed_results_0917')
    parser.add_argument('--use_shift', action='store_true')
    parser.add_argument('--use_bd', action='store_true')
    parser.add_argument('--model_type', type=str, default='llada2',
        help="llada2 (for llada2-mini or llada2-flash) | llada_moe (for llada-moe) | llada (for llada or llada-1.5)")
    parser.add_argument('--config', type=int, default=0)
    # MCMC 相关参数
    parser.add_argument('--n_mcmc_steps', type=int, default=3, help='Number of MCMC steps per block')
    parser.add_argument('--mcmc_alpha', type=float, default=4.0, help='MCMC alpha (power parameter)')
    parser.add_argument('--mcmc_temperature', type=float, default=0.9, help='MCMC temperature')
    parser.add_argument('--mcmc_use_kv_cache', action='store_true', help='Enable KV cache in MCMC proposal generation')
    args = parser.parse_args()

    if args.config == 1:
        args.cache = ''
        args.parallel_decoding = 'threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.95
        args.warmup_times = 0
    elif args.config == 2:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.95
        args.warmup_times = 0
    elif args.config == 3:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.95
        args.warmup_times = 4
    elif args.config == 4:
        args.cache = ''
        args.parallel_decoding = 'threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.8
        args.warmup_times = 0
    elif args.config == 5:
        args.cache = ''
        args.parallel_decoding = 'hierarchy_faster'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.8
        args.low_threshold = 0.5
        args.warmup_times = 0
    elif args.config == 6:
        args.cache = 'dual'
        args.parallel_decoding = 'hierarchy_faster'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.8
        args.low_threshold = 0.5
        args.warmup_times = 4
    elif args.config == 9:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.9
        args.low_threshold = 0.7
        args.warmup_times = 4

    elif args.config == 10:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.85
        args.warmup_times = 4
    elif args.config == 11:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.8
        args.low_threshold = 0.75
        args.warmup_times = 4

    elif args.config == 12:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.85
        args.low_threshold = 0.5
        args.warmup_times = 4
        
    elif args.config == 13:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.8
        args.warmup_times = 4

    elif args.config == 14:
        args.cache = 'dual'
        args.parallel_decoding = 'hierarchy_faster'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.9
        args.low_threshold = 0.7
        args.warmup_times = 4

    elif args.config == 15:
        args.cache = 'dual'
        args.parallel_decoding = 'hierarchy_faster'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.85
        args.low_threshold = 0.75
        args.warmup_times = 4
    elif args.config == 40:
        args.cache = 'prefix'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.95
        args.warmup_times = 0
        args.use_bd=True

    elif args.config == 41:
        args.cache = 'prefix'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.95
        args.warmup_times = 0
        args.use_bd=True
        args.block_length=32

    # MCMC 配置预设
    elif args.config == 20:
        # MCMC 基础配置（无 KV Cache）
        args.cache = ''
        args.parallel_decoding = 'mcmc_threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.9
        args.warmup_times = 0
        args.n_mcmc_steps = 3
        args.mcmc_alpha = 4.0
        args.mcmc_temperature = 0.9
        args.mcmc_use_kv_cache = False

    elif args.config == 21:
        # MCMC + dual KV Cache
        args.cache = 'dual'
        args.parallel_decoding = 'mcmc_threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.9
        args.warmup_times = 0
        args.n_mcmc_steps = 3
        args.mcmc_alpha = 4.0
        args.mcmc_temperature = 0.9
        args.mcmc_use_kv_cache = True

    elif args.config == 22:
        # MCMC + prefix KV Cache
        args.cache = 'prefix'
        args.parallel_decoding = 'mcmc_threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.9
        args.warmup_times = 0
        args.n_mcmc_steps = 5
        args.mcmc_alpha = 4.0
        args.mcmc_temperature = 0.9
        args.mcmc_use_kv_cache = True

    elif args.config == 23:
        # MCMC 高精度配置（更多 MCMC 步数，更高 alpha）
        args.cache = 'dual'
        args.parallel_decoding = 'mcmc_threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.95
        args.warmup_times = 0
        args.n_mcmc_steps = 5
        args.mcmc_alpha = 6.0
        args.mcmc_temperature = 0.9
        args.mcmc_use_kv_cache = True
        

    print(f"The input args are listed as follows: {args}")

    args = process_args(args)
    gpus = [int(gpu) for gpu in args.gpu.split(',')]
    procs = []
    if len(gpus) == 1:
        main(1, 0, gpus[0], args)
    else:
        for i, gpu in enumerate(gpus):
            p = Process(target=main, args=(len(gpus), i, gpu, args))
            p.daemon = True
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
