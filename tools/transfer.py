"""
===================================================================================
transfer.py - LLaDA MoE 模型转换工具
===================================================================================
功能：将标准的 MoE 模型转换为 FusedMoE 格式，用于高效推理

核心改进：
1. 权重融合：将 64 个独立专家的权重合并为紧凑的张量
2. 内存优化：显著减少模型内存占用
3. 计算加速：利用 FusedMoE kernel 实现高效矩阵乘法
===================================================================================
"""
import sys
import os
import shutil
import argparse
import torch
from transformers import AutoConfig, AutoTokenizer

# ============================================================================
# 1. 路径设置
# ============================================================================
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

sys.path.append(current_dir)

# 导入自定义的 FusedMoE 模型类
from .modeling_fused_lladamoe import FusedLLaDAMoEModelLM
from transformers import AutoTokenizer, AutoModel

def convert_and_save(
    input_path: str,      # 原始模型路径
    output_path: str,     # 转换后保存路径
    modeling_file_name: str,  # 使用的 modeling 文件名
    device: str = "cpu"   # 执行设备（默认 CPU 以节省显存）
):
    """
    ========================================================================
    核心转换函数：OlmoeForCausalLM → FusedOlmoeForCausalLM
    ========================================================================

    转换过程：
    1. 加载原始模型 (64个独立专家)
    2. 融合专家权重 (合并为统一张量)
    3. 保存 FusedMoE 格式模型

    专家权重布局变化：
    -----------------------------------------------------------------
    原始格式:
        每个专家有3个独立矩阵:
        - experts.0.gate_proj.weight  [intermediate_size, hidden_size]
        - experts.0.up_proj.weight    [intermediate_size, hidden_size]
        - experts.0.down_proj.weight  [hidden_size, intermediate_size]
        ...
        - experts.63.gate_proj.weight
        - experts.63.up_proj.weight
        - experts.63.down_proj.weight

    FusedMoE格式:
        所有专家权重融合为2个大张量:
        - w1 [num_experts, 2*intermediate_size, hidden_size]
             ↑ 融合了 gate_proj 和 up_proj
        - w2 [num_experts, hidden_size, intermediate_size]
             ↑ 对应 down_proj
    -----------------------------------------------------------------
    """

    # ========================================================================
    # 步骤 1: 加载原始模型
    # ========================================================================
    print(f"Loading original model from {input_path}...")
    config = AutoConfig.from_pretrained(input_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        input_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    state_dict = model.state_dict()

    num_layers = config.num_hidden_layers  # 层数，如 32 层
    num_experts = config.num_experts       # 专家数，如 64 个
    print(f"Model config found: {num_layers} layers, {num_experts} experts per layer.")

    # ========================================================================
    # 步骤 2: 创建 FusedMoE 模型结构
    # ========================================================================
    print("Building fused model...")
    fused_model = FusedLLaDAMoEModelLM(config).to(device)
    fused_state_dict = fused_model.state_dict()

    # ========================================================================
    # 步骤 3: 权重融合 - 核心转换逻辑
    # ========================================================================
    print("Mapping and fusing expert weights...")

    for i in range(num_layers):  # 遍历每一层
        layer_prefix = f"model.layers.{i}.mlp."

        # --------------------------------------------------------------------
        # 3.1 收集当前层所有专家的权重
        # --------------------------------------------------------------------
        # gate_proj: SwiGLU 激活函数的门控分支
        gate_weights = [
            state_dict[f"{layer_prefix}experts.{j}.gate_proj.weight"].to(device)
            for j in range(num_experts)
        ]  # List[Tensor], 每个 shape: [intermediate_size, hidden_size]

        # up_proj: SwiGLU 激活函数的上升分支
        up_weights = [
            state_dict[f"{layer_prefix}experts.{j}.up_proj.weight"].to(device)
            for j in range(num_experts)
        ]  # List[Tensor], 每个 shape: [intermediate_size, hidden_size]

        # down_proj: 输出投影
        down_weights = [
            state_dict[f"{layer_prefix}experts.{j}.down_proj.weight"].to(device)
            for j in range(num_experts)
        ]  # List[Tensor], 每个 shape: [hidden_size, intermediate_size]

        # --------------------------------------------------------------------
        # 3.2 融合权重 - 关键操作!
        # --------------------------------------------------------------------
        """
        为什么要融合 gate_proj 和 up_proj？

        原始计算（每个专家独立）：
            gate_output = gate_proj(x)      # [batch, intermediate_size]
            up_output = up_proj(x)          # [batch, intermediate_size]
            mlp_output = SwiGLU(gate_output, up_output)
            final = down_proj(mlp_output)

        融合后计算（高效）：
            combined = w1(x)                # 一次矩阵乘法
            gate_output, up_output = split(combined)
            mlp_output = SwiGLU(gate_output, up_output)
            final = w2(mlp_output)

        优势：
        1. 减少矩阵乘法次数 (2次 → 1次)
        2. 更好的内存访问模式
        3. 利用 Triton FusedMoE kernel 并行计算
        """

        # 将 gate 和 up 沿着输出维度拼接
        combined_w1 = torch.stack([
            torch.cat([g, u], dim=0)  # 拼接后 shape: [2*intermediate_size, hidden_size]
            for g, u in zip(gate_weights, up_weights)
        ])
        # combined_w1 shape: [num_experts, 2*intermediate_size, hidden_size]

        # down_proj 直接堆叠
        combined_w2 = torch.stack(down_weights)
        # combined_w2 shape: [num_experts, hidden_size, intermediate_size]

        # 保存到 fused_state_dict
        fused_state_dict[f"{layer_prefix}w1"] = combined_w1
        fused_state_dict[f"{layer_prefix}w2"] = combined_w2

    # ========================================================================
    # 步骤 4: 复制非专家参数（注意力层、LayerNorm等）
    # ========================================================================
    print("Copying non-expert parameters...")
    for key in state_dict:
        if 'experts' not in key:  # 不是专家权重的参数直接复制
            fused_state_dict[key] = state_dict[key]

    # 加载融合后的权重
    fused_model.load_state_dict(fused_state_dict)

    # ========================================================================
    # 步骤 5: 更新模型配置，使其能被 AutoModel 识别
    # ========================================================================
    print("Updating model configuration for fused model...")
    if not hasattr(fused_model.config, "auto_map"):
        fused_model.config.auto_map = {}

    fused_model_class_name = FusedLLaDAMoEModelLM.__name__
    full_module_class_path = f"{modeling_file_name}.{fused_model_class_name}"

    # 注册到 auto_map，使 AutoModelForCausalLM 能自动加载
    fused_model.config.auto_map["AutoModelForCausalLM"] = full_module_class_path
    fused_model.config.auto_map.pop("AutoModel", None)  # 移除旧的 AutoModel 映射
    fused_model.config.architectures = [fused_model_class_name]

    # ========================================================================
    # 步骤 6: 保存模型和tokenizer
    # ========================================================================
    print(f"Saving fused model to {output_path}")
    fused_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    # ========================================================================
    # 步骤 7: 复制自定义建模文件（包含 FusedMoE kernel）
    # ========================================================================
    print("Copying custom modeling files to output directory...")
    source_files = [
        os.path.join(current_dir, f"{modeling_file_name}.py"),
        os.path.join(current_dir, "fuse_moe.py")  # Triton FusedMoE kernel
    ]

    os.makedirs(output_path, exist_ok=True)

    for src_file in source_files:
        if os.path.exists(src_file):
            dest_file = os.path.join(output_path, os.path.basename(src_file))
            try:
                shutil.copy2(src_file, dest_file)
                print(f"Copied {os.path.basename(src_file)} to {output_path}")
            except Exception as e:
                print(f"Error copying {os.path.basename(src_file)}: {e}")
        else:
            print(f"Warning: Source file not found, skipping copy: {src_file}")

    print("✅ Conversion completed!")


# ============================================================================
# 命令行入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将标准 LLaDA-MoE 模型转换为 FusedMoE 格式以提升推理性能"
    )
    parser.add_argument('--input',  type=str, required=True,
                       help="原始模型路径")
    parser.add_argument('--output', type=str, required=True,
                       help="转换后模型保存路径")
    parser.add_argument('--modeling', type=str, default='modeling_fused_olmoe',
                       help="使用的 modeling 文件名（默认: modeling_fused_olmoe）")
    args = parser.parse_args()

    input_path = args.input.rstrip('/')
    output_path = args.output

    print(f"\n----- Starting CPU conversion for {input_path} -----")
    convert_and_save(
        input_path=input_path,
        output_path=output_path,
        modeling_file_name=args.modeling,
    )
    print(f"----- Finished conversion for {input_path} -> {output_path} -----\n")
