<!-- @format -->

# Benchmarks 文件说明

本文档介绍 benchmarks 文件夹中五个基准测试文件的功能、区别和使用方法。

## 文件概览

### 1. benchmark.py

**功能**: 单样本基准测试脚本

- 用于测试单个固定样本的生成性能
- 支持多种模型类型（llada、llada_moe、llada2）
- 支持多种解码策略（threshold、hierarchy）
- 包含 torch.compile 编译优化和 CUDA Graph 预热
- 测试指标：NFE（前向次数）、TPS（每秒 token 数）、TPF（每次前向的 token 数）、FPS（每秒前向次数）

**特点**:

- 使用内置固定测试样本（数学问题）
- 适合快速验证模型功能和性能
- 不需要准备数据集

**运行示例**:

```bash
# LLaDA-8B-Instruct 单样本测试
python benchmarks/benchmark.py \
  --model_name GSAI-ML/LLaDA-8B-Instruct \
  --model_type llada \
  --gen_len 2048 \
  --block_length 32 \
  --gpu 0,1,2,3 \
  --use_tp \
  --parallel_decoding threshold \
  --threshold 0.9 \
  --cache prefix

# LLaDA-MoE 单样本测试
python benchmarks/benchmark.py \
    --model_name /home/zhounan/models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct-fused \
    --model_type llada_moe \
    --gen_len 2048 \
    --block_length 64 \
    --gpu 0,1 \
    --use_tp \
    --parallel_decoding threshold \
    --threshold 0.9 \
    --low_threshold 0.7 \
    --cache dual \
    --prefix_look 16 \
    --after_look 16

# LLaDA2-mini 单样本测试
python benchmarks/benchmark.py \
  --model_name inclusionAI/LLaDA2.0-mini-preview \
  --model_type llada2 \
  --gen_len 2048 \
  --block_length 32 \
  --gpu 0,1,2,3 \
  --use_tp \
  --parallel_decoding threshold \
  --threshold 0.9 \
  --cache prefix \
  --use_bd
```

---

### 2. benchmark_dataset.py

**功能**: 数据集批量基准测试脚本（标准版本）

- 从 JSON 数据集加载多个样本进行批量测试
- 支持批处理（batch processing）
- 使用 bucket 机制（bucket_size=32）对输入长度进行分组优化
- 支持多种缓存策略（prefix、dual）
- 支持多种扩散模型变体（BlockWise、IterSmooth、VicinityCache 等）
- 输出详细的结果到 JSONL 文件和 results.txt

**特点**:

- 支持预定义配置（config 1-15, 40-41）
- 使用 vLLM 后端
- 包含 CUDA Graph 预热
- 适合标准的批量性能测试

**运行示例**:

```bash
# LLaDA-MoE 数据集测试（使用预定义配置）
python benchmarks/benchmark_dataset.py \
  --model_name /home/zhounan/models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct-fused \
  --model_type llada_moe \
  --dataset datasets/gsm8k/compare_data.json \
  --gen_len 1024 \
  --block_length 64 \
  --gpu 0,1 \
  --output_dir runs/llada_moe_threshold \
  --use_tp \
  --parallel_decoding threshold \
  --threshold 0.8 \
  --cache dual \
  --prefix_look 16 \
  --after_look 16 \
  --warmup_times 4 \
  --cont_weight 0.3

# LLaDA2-flash 数据集测试（手动配置参数）
python benchmarks/benchmark_dataset.py \
  --model_name inclusionAI/LLaDA2.0-flash-preview \
  --model_type llada2 \
  --dataset dataset_path \
  --gen_len 2048 \
  --block_length 32 \
  --gpu 0,1,2,3 \
  --output_dir runs/llada2_flash \
  --use_tp \
  --parallel_decoding threshold \
  --threshold 0.9 \
  --cache prefix \
  --use_bd
```

---

### 2.1 benchmark_dataset_mcmc.py

**功能**: 支持 MCMC Power Sampling 精炼的数据集批量基准测试脚本

- 基于 benchmark_dataset.py，增加了 BlockMCMCDiffusionLLM 支持
- 支持 MCMC Power Sampling 算法对生成结果进行精炼
- 支持 KV Cache 加速（包括 MCMC 提议生成阶段）
- 输出 MCMC 特有的性能指标（diffusion_forwards、proposal_forwards）

**特点**:

- 支持预定义 MCMC 配置（config 20-23）
- 可配置 MCMC 参数：n_mcmc_steps、mcmc_alpha、mcmc_temperature
- 支持 MCMC 提议生成的 KV Cache 加速（--mcmc_use_kv_cache）
- 适合评估 MCMC 精炼对生成质量的提升

**MCMC 配置预设**:

- `config=20`: MCMC 基础配置（无 KV Cache）
- `config=21`: MCMC + dual KV Cache
- `config=22`: MCMC + prefix KV Cache
- `config=23`: MCMC 高精度配置（更多步数，更高 alpha）

**运行示例**:

```bash
# MCMC 基础测试（无 KV Cache）
python benchmarks/benchmark_dataset_mcmc.py \
  --model_name /home/zhounan/models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct-fused \
  --model_type llada_moe \
  --dataset datasets/gsm8k/compare_data.json \
  --gen_len 1024 \
  --block_length 64 \
  --gpu 0 \
  --output_dir runs/llada_moe_mcmc \
  --parallel_decoding mcmc_threshold \
  --threshold 0.9 \
  --n_mcmc_steps 3 \
  --mcmc_alpha 4.0 \
  --mcmc_temperature 0.9

# MCMC + dual KV Cache（推荐配置）
python benchmarks/benchmark_dataset_mcmc.py \
  --model_name /home/zhounan/models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct-fused \
  --model_type llada_moe \
  --dataset datasets/gsm8k/compare_data.json \
  --gen_len 256 \
  --block_length 64 \
  --gpu 0,1 \
  --output_dir runs/llada_moe_mcmc_kv \
  --use_tp \
  --parallel_decoding mcmc_threshold \
  --threshold 0.9 \
  --cache dual \
  --n_mcmc_steps 3 \
  --mcmc_alpha 4.0 \
  --mcmc_temperature 0.9 \
  --mcmc_use_kv_cache

# 使用预定义 MCMC 配置
python benchmarks/benchmark_dataset_mcmc.py \
  --model_name /home/zhounan/models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct-fused \
  --model_type llada_moe \
  --dataset datasets/gsm8k/compare_data.json \
  --gen_len 1024 \
  --block_length 64 \
  --gpu 0,1 \
  --output_dir runs/llada_moe_mcmc_config21 \
  --use_tp \
  --config 21

# MCMC 高精度配置（更多步数）
python benchmarks/benchmark_dataset_mcmc.py \
  --model_name /home/zhounan/models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct-fused \
  --model_type llada_moe \
  --dataset datasets/gsm8k/compare_data.json \
  --gen_len 1024 \
  --block_length 64 \
  --gpu 0,1 \
  --output_dir runs/llada_moe_mcmc_high \
  --use_tp \
  --config 23
```

**MCMC 参数说明**:

- `--n_mcmc_steps`: 每个块的 MCMC 迭代次数（默认 3）
- `--mcmc_alpha`: Power Sampling 的 α 参数，控制目标分布 p^α（默认 4.0）
- `--mcmc_temperature`: MCMC 采样温度（默认 0.9）
- `--mcmc_use_kv_cache`: 启用 MCMC 提议生成的 KV Cache 加速

---

### 3. benchmark_dataset_fastdllm.py

**功能**: 使用 FastDLLM 生成方法的数据集基准测试

- 专门用于测试 FastDLLM 生成算法
- 使用 `generate_fastdllm` 函数而非标准的扩散模型生成
- 支持 LLaDA 和 LLaDAMoe 模型
- 较小的 bucket size（8 vs 32）
- 单样本逐个处理（不支持批处理）

**特点**:

- 专注于 FastDLLM 算法性能评估
- 使用 vLLM 后端 + Expert Parallelism (EP)
- 固定使用 dual cache 优化
- 适合评估 FastDLLM 特定优化效果

**运行示例**:

```bash
# LLaDAMoe FastDLLM 测试
python benchmarks/benchmark_dataset_fastdllm.py \
  --model_name /path/to/LLaDA-MoE-7B-A1B-Instruct-fused \
  --dataset /path/to/dataset.json \
  --gen_len 1024 \
  --block_length 64 \
  --gpu 0,1,2,3 \
  --output_dir runs/fastdllm \
  --use_tp \
  --parallel_decoding fastdllm \
  --threshold 0.95
```

---

### 4. benchmark_dataset_sglang.py

**功能**: 使用 SGLang 后端的数据集基准测试

- 使用 SGLang 推理框架而非 vLLM
- 支持 LLaDA2 SGLang 模型实现（LLaDA2SGLangLM）
- 包含输入长度排序优化
- 支持 DP attention（数据并行注意力）
- 使用 ModelRunner 封装模型

**特点**:

- SGLang 后端集成（sglang.srt）
- 输入按长度排序以提高批处理效率
- 支持 MoE 专家并行配置
- 随机端口分配避免多进程冲突
- 支持预定义配置（config 1-15, 40-41）

**运行示例**:

```bash
# LLaDA2 SGLang 后端测试
python benchmarks/benchmark_dataset_sglang.py \
  --model_name /path/to/LLaDA2-model \
  --dataset /path/to/dataset.json \
  --gen_len 1024 \
  --block_length 64 \
  --batch_size 4 \
  --gpu 0,1,2,3 \
  --output_dir runs/sglang \
  --use_tp \
  --parallel_decoding threshold \
  --threshold 0.95 \
  --cache dual \
  --prefix_look 16 \
  --after_look 16 \
  --warmup_times 4 \
  --config 3
```

---

### 5. benchmark_dataset_sorted.py

**功能**: 带输入排序和动态预热的数据集基准测试

- 按输入长度对样本排序
- 动态 CUDA Graph 预热机制
- 当 prefill 长度变化时自动重新预热
- 使用两个解码器（主解码器用于推理，预热解码器用于 warmup）

**特点**:

- 智能预热策略：只在 prefill 长度改变时预热
- 输入排序优化：相似长度的样本连续处理，减少预热次数
- 更精细的性能优化
- 适合输入长度变化较大的数据集
- 支持预定义配置（config 1, 40）

**运行示例**:

```bash
# LLaDA2-mini 排序优化测试
python benchmarks/benchmark_dataset_sorted.py \
  --model_name inclusionAI/LLaDA2.0-mini-preview \
  --model_type llada2 \
  --dataset /path/to/dataset.json \
  --gen_len 1024 \
  --block_length 32 \
  --batch_size 4 \
  --gpu 0,1,2,3 \
  --output_dir runs/sorted \
  --use_tp \
  --parallel_decoding threshold \
  --threshold 0.95 \
  --cache prefix \
  --use_bd \
  --config 40

# LLaDA-MoE 排序优化测试
python benchmarks/benchmark_dataset_sorted.py \
  --model_name inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --model_type llada_moe \
  --dataset /path/to/dataset.json \
  --gen_len 1024 \
  --block_length 64 \
  --batch_size 4 \
  --gpu 0,1,2,3 \
  --output_dir runs/sorted \
  --config 1
```

---

## 主要区别对比

| 特性        | benchmark.py | benchmark_dataset.py | benchmark_dataset_mcmc.py | benchmark_dataset_fastdllm.py | benchmark_dataset_sglang.py | benchmark_dataset_sorted.py |
| ----------- | ------------ | -------------------- | ------------------------- | ----------------------------- | --------------------------- | --------------------------- |
| 测试类型    | 单样本       | 批量数据集           | 批量数据集                | 批量数据集                    | 批量数据集                  | 批量数据集                  |
| 后端框架    | vLLM         | vLLM                 | vLLM                      | vLLM                          | SGLang                      | vLLM                        |
| 生成方法    | 扩散模型     | 扩散模型             | 扩散模型+MCMC             | FastDLLM                      | 扩散模型                    | 扩散模型                    |
| MCMC 精炼   | 否           | 否                   | 是                        | 否                            | 否                          | 否                          |
| 输入排序    | 否           | 否                   | 否                        | 否                            | 是                          | 是                          |
| 动态预热    | 否           | 否                   | 否                        | 否                            | 否                          | 是                          |
| Bucket 大小 | N/A          | 32                   | 32                        | 8                             | 32                          | 32                          |
| 批处理      | 否           | 是                   | 是                        | 否（固定为 1）                | 是                          | 是                          |
| 配置预设    | 否           | 是（1-15, 40-41）    | 是（1-15, 20-23, 40-41）  | 否                            | 是（1-15, 40-41）           | 是（1, 40）                 |

## 使用建议

- **快速功能验证**: 使用 `benchmark.py` - 无需准备数据集，快速测试模型是否正常工作
- **标准性能评估**: 使用 `benchmark_dataset.py` - 完整数据集评估，支持批处理
- **MCMC 精炼评估**: 使用 `benchmark_dataset_mcmc.py` - 评估 MCMC Power Sampling 对生成质量的提升
- **FastDLLM 算法评估**: 使用 `benchmark_dataset_fastdllm.py` - 专门测试 FastDLLM 优化效果
- **SGLang 后端测试**: 使用 `benchmark_dataset_sglang.py` - 使用 SGLang 推理框架
- **长度变化大的数据集**: 使用 `benchmark_dataset_sorted.py` - 智能排序和动态预热，提高效率

## 常用参数说明

### 模型相关

- `--model_name`: 模型路径或 HuggingFace 模型 ID（必需）
- `--model_type`: 模型类型，可选 `llada`、`llada_moe`、`llada2`（必需）
- `--gpu`: GPU 设备列表，如 `0,1,2,3`
- `--use_tp`: 启用 Tensor Parallelism（多 GPU 时推荐）

### 生成参数

- `--gen_len`: 生成长度（默认 1024）
- `--block_length`: 块长度（默认 64，LLaDA2 推荐 32）
- `--batch_size`: 批大小（仅数据集测试脚本）
- `--dataset`: 数据集路径（仅数据集测试脚本）

### 解码策略

- `--parallel_decoding`: 解码策略，可选 `threshold`、`hierarchy_faster`、`mcmc_threshold`
- `--threshold`: 并行解码阈值（0.8-0.95）
- `--low_threshold`: 层级解码的低阈值（仅 hierarchy 模式）

### MCMC 参数（仅 benchmark_dataset_mcmc.py）

- `--n_mcmc_steps`: 每个块的 MCMC 迭代次数（默认 3）
- `--mcmc_alpha`: Power Sampling 的 α 参数，控制目标分布 p^α（默认 4.0，越大越倾向高概率序列）
- `--mcmc_temperature`: MCMC 采样温度（默认 0.9）
- `--mcmc_use_kv_cache`: 启用 MCMC 提议生成的 KV Cache 加速

### 优化选项

- `--cache`: KV cache 类型，可选 `prefix`、`dual`
- `--prefix_look`: 前缀查看窗口大小
- `--after_look`: 后缀查看窗口大小
- `--warmup_times`: 预热次数
- `--cont_weight`: 连续性权重（IterSmooth 模式）
- `--use_bd`: 启用 Block Diffusion（仅 LLaDA2）
- `--use_shift`: 启用 shift 优化

### 输出

- `--output_dir`: 结果输出目录
- `--exp_name`: 实验名称
- `--config`: 预定义配置编号（部分脚本支持）

## 性能指标说明

所有脚本输出以下性能指标：

- **TPS** (Tokens Per Second): 每秒生成的 token 数，衡量整体吞吐量
- **TPF** (Tokens Per Forward): 每次前向传播生成的 token 数，衡量并行解码效率
- **FPS** (Forwards Per Second): 每秒前向传播次数
- **NFE** (Number of Function Evaluations): 总前向传播次数

MCMC 模式额外输出：

- **diffusion_forwards**: 扩散去噪阶段的前向传播次数
- **proposal_forwards**: MCMC 提议生成阶段的前向传播次数

## 注意事项

1. **LLaDA2 模型**:

   - 必须使用 `--use_bd` 启用 Block Diffusion
   - 推荐使用 `--cache prefix`
   - 最大支持 4-way TP（因为只有 4 个 attention heads）
   - 推荐 `block_length=32`

2. **LLaDA/LLaDA-MoE 模型**:

   - 不支持 `--use_bd`
   - 支持最多 8-way TP
   - 推荐 `block_length=64`

3. **多 GPU 使用**:

   - 使用多个 GPU 时建议添加 `--use_tp` 启用 Tensor Parallelism
   - GPU 列表格式：`--gpu 0,1,2,3`

4. **预定义配置**:

   - `config=1`: 基础 threshold 解码，无 cache
   - `config=3`: dual cache + vicinity refresh
   - `config=20`: MCMC 基础配置（无 KV Cache）
   - `config=21`: MCMC + dual KV Cache（推荐）
   - `config=22`: MCMC + prefix KV Cache
   - `config=23`: MCMC 高精度配置（更多步数，更高 alpha）
   - `config=40`: Block Diffusion 模式（LLaDA2）
   - 详见各脚本源码中的配置定义

5. **MCMC 模式注意事项**:
   - MCMC 精炼会增加前向传播次数，但可以提升生成质量
   - 推荐使用 `--cache dual --mcmc_use_kv_cache` 以获得最佳性能
   - `mcmc_alpha` 越大，越倾向于高概率序列，但可能降低多样性
   - `n_mcmc_steps` 越多，精炼效果越好，但计算开销也越大
