import os
import torch
import time
from transformers import AutoTokenizer, AutoConfig
from vllm import distributed
from vllm.config import ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config

from dinfer.model import LLaDAMoeModelLM 
from dinfer import BlockIteratorFactory, KVCacheFactory  
from dinfer import ThresholdParallelDecoder, BlockWiseDiffusionLLM  

print("========== 步骤1: 模型路径和Tokenizer加载 ==========")
m = "/home/zhounan/models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct-fused"
tokenizer = AutoTokenizer.from_pretrained(m, trust_remote_code=True)

print("========== 步骤2: 设备和分布式环境初始化 ==========")
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda:0')  
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12346'

# 初始化分布式环境
# 参数: (world_size, rank, init_method, local_rank, backend)
distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
# 初始化模型并行(Tensor Parallel, TP=1)
distributed.initialize_model_parallel(1, backend='nccl')

print("========== 步骤3: 配置并行策略和加载模型 ==========")
# 启用专家并行(Expert Parallel, EP)以提高MoE模型性能
parallel_config = ParallelConfig(enable_expert_parallel=True)
with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
    # 加载模型配置
    model_config = AutoConfig.from_pretrained(m, trust_remote_code=True)
    # 创建FusedMoE模型实例(评估模式)
    model = LLaDAMoeModelLM(config=model_config).eval()
    # 加载模型权重(使用bfloat16精度)
    model.load_weights(m, torch_dtype=torch.bfloat16)
    # 将模型移动到指定设备
    model = model.to(device)

print("========== 步骤4: 配置解码器和扩散LLM ==========")
# 创建阈值并行解码器
# 参数: (rank, threshold, mask_id, eos_id)
decoder = ThresholdParallelDecoder(0, threshold=0.9, mask_id=156895, eos_id=156892)

# 创建块级扩散LLM
dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(True), cache_factory=KVCacheFactory('dual'))

print("========== 步骤5: 准备输入和生成 ==========")
prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours?"
m = [{"role": "user", "content": prompt}, ]
# 应用聊天模板(添加生成提示符)
prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
# Tokenize输入文本
input_ids = tokenizer(prompt)['input_ids']
# 转换为张量并移动到设备,添加batch维度
input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

print("========== 步骤6: 执行生成 ==========")
start_time = time.time()
res = dllm.generate(input_ids, gen_length=1024, block_length=64)
end_time = time.time()
print(f"生成耗时: {end_time - start_time:.2f} 秒")

print("========== 步骤7: 解码并输出结果 ==========")
print(tokenizer.decode(res[0, input_ids.shape[1]:], skip_special_tokens=False))
