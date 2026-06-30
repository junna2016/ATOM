# ATOM DeepSeek-V4 RTP-LLM Plugin 开发状态

> 最后更新: 2026-06-30

## 概述

ATOM 通过 rtp-llm 的 external plugin 机制为 DeepSeek-V4 提供 ROCm (MI308X/MI355X) 推理支持。Plugin 复用 rtp-llm 的服务框架（API server、KV cache 管理、batch 调度），同时使用 ATOM 的 ROCm 算子内核（aiter Triton kernels）替代 rtp-llm 的 CUDA-only 算子。

**当前状态：TP（张量并行）功能已全部打通。** BS=1 和 BS>1、Eager 和 CUDA Graph 模式均已验证通过，精度与 ATOM standalone 对齐。

## 架构

```
rtp-llm (C++ engine)
  ├── API Server (OpenAI-compatible)
  ├── KV Cache Manager (multi-region paged pools)
  ├── CUDA Graph Runner
  └── PyWrappedModel → ATOM Plugin
        ├── ATOMDeepSeekV4 (model loader, weight mapping)
        ├── _ATOMDeepSeekV4Runtime (forward, CUDA graph hooks)
        ├── rtp_v4_attention.py (attention adapter)
        └── v4_kv_cache_bridge.py (pool mapping)
```

### 核心文件

| 文件 | 功能 |
|------|------|
| `atom/plugin/rtpllm/models/deepseek_v4.py` | 模型加载、CUDA graph 适配、forward 入口 |
| `atom/plugin/rtpllm/attention_backend/rtp_v4_attention.py` | V4 attention 的 prefill/decode 适配 |
| `atom/plugin/rtpllm/utils/v4_kv_cache_bridge.py` | rtp-llm 多区域 KV cache → ATOM pool 映射 |
| `atom/plugin/rtpllm/utils/forward_context.py` | RTP forward context 管理 |
| `atom/model_ops/v4_kernels/paged_decode.py` | Triton paged decode 内核（dual-pointer SPLIT_KV） |

## 功能状态

### ✅ 已完成（TP 全部打通）

| 功能 | Eager | CUDA Graph | 说明 |
|------|-------|-----------|------|
| BS=1 推理 | ✅ | ✅ | 短/长 prompt 均通过 |
| BS>1 并发推理 | ✅ | ✅ | 3 路短请求 + 2 路长请求验证 |
| Think Mode (think_mode=1) | ✅ | ✅ | reasoning_content + content 分离输出 |
| SWA 滑动窗口注意力 | ✅ | ✅ | Ring buffer + compact buffer + pool gather |
| CSA 压缩注意力 (ratio=4) | ✅ | ✅ | Compressor + Indexer + FP8 shadow buffer |
| HCA 压缩注意力 (ratio=128) | ✅ | ✅ | Compressor + 128:1 压缩 |
| Dual-Pointer SPLIT_KV | ✅ | ✅ | SWA 和 Compress 从不同 pool 读取 |
| TP=4 张量并行 | ✅ | ✅ | 4 卡 MI308X 验证 |
| FP8 (e4m3fnuz) 权重 | ✅ | ✅ | DeepSeek-V4-Flash-FP8 模型 |
| Multi-region KV Cache | ✅ | ✅ | SWA/CSA/HCA/INDEXER/STATE pools |
| 多 block SWA gather | ✅ | ✅ | prompt > 128 tokens 跨 block 正确拼接 |

### ⚠️ 已知限制

| 项目 | 说明 |
|------|------|
| think_mode=0 | 关闭思考模式后生成质量下降（模型特性，非精度 bug）。建议生产环境使用 think_mode=1 |
| MTP (Multi-Token Prediction) | 未实现 (`deepseek_v4_mtp.py` 为 placeholder) |
| DP+EP (Data Parallel + Expert Parallel) | Plugin 模式下未适配，需 ATOM standalone 使用 |
| EPLB (Expert-Level Load Balancing) | Plugin 模式下 `_NoopModelWeightsLoader` 跳过 |
| lm_eval (GSM8K 等) | rtp-llm 仅支持 chat completions API，lm_eval 需适配 |

## CUDA Graph 精度修复记录

开启 CUDA Graph 后遇到精度问题，经排查共修复 7 个 bug：

### Bug 1: Compressor state reset 破坏 prefill 状态
- **现象**: 第一个 decode token 后输出乱码
- **根因**: `prepare_cuda_graph()` 检测到"新请求"后清零 compressor 的 `kv_state`/`score_state`，破坏了 prefill 积累的未压缩尾部状态
- **修复**: 移除 `prepare_cuda_graph` 中的 state reset。Prefill 的 `_reset_v4_state_all()` 已负责初始化

### Bug 2: Graph replay 的 positions 值不对
- **现象**: CUDA graph replay 时 RoPE 计算错误
- **根因**: `_forward_impl()` 中 `pos_i64.copy_(positions)` 从 `.to(dtype=int32)` 产生的临时 tensor 复制，`prepare_cuda_graph` 更新的是 `bufs["positions"]`（另一个 buffer）
- **修复**: CUDA graph 模式下改为 `pos_i64.copy_(v4_bufs["positions"])`

### Bug 3: Compact SWA buffer 缺少 prefill 数据
- **现象**: 输出完全乱码（读到全零 KV）
- **根因**: Prefill 写 KV 到 pool，但 graph 读 KV 从 compact buffer。Graph capture 时 `_block_ids` 不存在 → gather 未被 capture → replay 时 compact buffer 全零
- **修复**: 在 `prepare_cuda_graph`（graph 外部）做 SWA + STATE pool gather，仅在新请求首步触发（block_ids 变化检测）

### Bug 4: block_ids=-1 导致越界 crash
- **现象**: 长序列 (>256 tokens) 时 GPU page fault
- **根因**: SWA block table 的 column 0 在序列超过 128 tokens 后可能被释放，`swa_bt[:, 0]` 返回 -1
- **修复**: `np.maximum(block_ids_np, 0)` 防护，与 eager 路径一致

### Bug 5: STATE 区域缺失
- **现象**: BS>1 时 compressor state 不正确
- **根因**: `build_v4_block_tables()` 只包含 `SWA_KV/CSA_KV/HCA_KV/INDEXER_KV`，缺少 `CSA_STATE/HCA_STATE/INDEXER_STATE`，导致 prefill 的 STATE scatter 和 decode 的 STATE gather 被跳过
- **修复**: 在 `build_v4_block_tables` 中加入 STATE 区域

### Bug 6: BS>1 prefill positions 累积
- **现象**: BS>1 时第 2+ 个请求输出乱码（Eager + CUDA Graph 均受影响）
- **根因**: Prefill 的 positions 使用 `torch.arange(total_tokens)` 生成累积位置 `[0,..,L1+L2-1]`，第二个 sequence 的 SWA 写入到错误的 ring positions，decode 时 attention 读到空数据
- **修复**: 使用 `input_lengths` 构造 per-sequence positions `[0,..,L1-1, 0,..,L2-1,...]`

### Bug 7: 多 block prompt 的 SWA gather 不完整
- **现象**: prompt > 128 tokens 时 CUDA Graph 模式输出乱码
- **根因**: SWA gather 只从 `swa_bt[:, 0]`（block column 0）复制，prompt 跨多 block 时当前 block 的 ring 数据未被正确收集
- **修复**: 根据 position 计算 ring 分割点，从当前和前一个 block 分别 gather 对应的 ring 段拼接到 compact buffer

## 验证矩阵

| 测试场景 | Eager | CUDA Graph |
|----------|-------|-----------|
| BS=1 短 prompt (11 tokens) | ✅ | ✅ |
| BS=1 长 prompt (128 tokens) | ✅ | ✅ |
| BS=1 超长 prompt (150 tokens) | ✅ | ✅ |
| BS=1 超长生成 (800+ tokens) | ✅ | ✅ |
| BS>1 并发短请求 (3 路) | ✅ | ✅ |
| BS>1 并发长请求 (2 路) | ✅ | ✅ |
| BS>1 混合短长请求 (5 路) | ✅ | ✅ |
| 多次重复测试稳定性 | ✅ | ✅ |
| 与 ATOM standalone 精度对齐 | ✅ | ✅ |

## 启动配置

```bash
# start_v4.sh
export RTP_LLM_EXTERNAL_MODEL_PACKAGES=atom.plugin.rtpllm.models
export ENABLE_CUDA_GRAPH=1
export ATOM_FORCE_ATTN_TRITON=1
export ATOM_V4_USE_TRITON_FUSION=1

python3 -m rtp_llm.start_server \
    --checkpoint_path <model_path> \
    --model_type deepseek_v4 \
    --tp_size 4 --world_size 4 \
    --max_seq_len 4096 \
    --think_mode 1 \
    --kernel_seq_size_per_block 128 \
    --seq_size_per_block 128
```

## 后续计划

- [ ] MTP (Multi-Token Prediction) 支持
- [ ] DP+EP plugin 模式适配
- [ ] GSM8K / lm_eval 精度评测（需适配 chat completions API）
- [ ] 性能优化：减少 `prepare_cuda_graph` 中的 per-layer 遍历开销
- [ ] 性能基准测试：对比 ATOM standalone vs rtp-llm plugin 的吞吐量
