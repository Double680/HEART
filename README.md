# HEART

HEART 是面向 MultiHiertt 类混合文本-表格文档问答任务的证据检索框架。当前主流程支持端到端问答、稠密检索、BM25 检索、BM25+稠密混合检索、标注证据上界检索、查询增强和子表证据组织。

## 环境配置

创建 Python 环境：

```bash
conda create -n hybtqa python=3.12
conda activate hybtqa
pip install torch==2.7.1
pip install -r requirements.txt
```

从 flash-attn 官方 release 页面下载与你的 CUDA、PyTorch、Python 版本匹配的 wheel 并安装。示例：

```bash
pip install flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

在仓库根目录创建 `.env`，用于配置问答阶段调用的 OpenAI 兼容 LLM API：

```bash
LLM_MODEL=your-chat-model
API_KEY=your-api-key
BASE_URL=https://your-openai-compatible-endpoint/v1
```

## 数据准备

下载 MultiHiertt 数据集，并放到：

```text
datasets/multihiertt/
  train.json
  dev.json
  test.json
```

检索代码默认每个样本包含以下字段：

- `paragraphs`：文本检索单元，以及 `## Table 0 ##` 这类表格占位符。
- `tables`：HTML 格式表格。
- `table_description`：以 `table-row-col` 为 key 的 cell-level 表格描述。
- `qa.text_evidence` 和 `qa.table_evidence`：用于 dev 集检索评测的文本/表格证据标注。

GRPO 训练脚本需要预处理后的训练文件：

```text
datasets/multihiertt/train_new.json
```

该文件除上述字段外，还应包含 `qa.table_evidence_id`。其中表格证据应表示为与 `table_description` 有序取值对齐的索引。

注意：当前代码直接在 `paragraphs` 已有单元上进行文本检索。如果要严格复现论文中的 sentence-chunk 设定，需要在生成 embedding 和运行检索前，先把 `paragraphs` 及对应 evidence id 预处理到句子级粒度。

## 训练 Query Augmentor

论文主设定的 HEART 训练目标是：文本证据使用 hard recall reward，表格证据使用 soft relevance reward。对应命令如下：

```bash
python augment/grpo_aug.py \
  --aug_type hybrid \
  --reward_retrieve_type dense \
  --train_data_path datasets/multihiertt/train_new.json \
  --retriever_model_path models/Qwen3-Embedding-0.6B \
  --augment_model_path models/Qwen3-1.7B \
  --metric_log_steps 10
```

多卡训练使用 `accelerate launch` 启动。训练模型由 TRL/Accelerate 按 DDP 方式分发；reward 中的检索模型默认加载到当前进程对应的本地 GPU，避免每个进程都通过 `device_map=auto` 占用全部显卡。

```bash
accelerate launch --num_processes 4 augment/grpo_aug.py \
  --aug_type hybrid \
  --reward_retrieve_type dense \
  --train_data_path datasets/multihiertt/train_new.json \
  --retriever_model_path models/Qwen3-Embedding-0.6B \
  --augment_model_path models/Qwen3-1.7B \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_generations 8 \
  --metric_log_steps 10
```

也可以使用 `torchrun`：

```bash
torchrun --nproc_per_node=4 augment/grpo_aug.py \
  --aug_type hybrid \
  --reward_retrieve_type dense \
  --train_data_path datasets/multihiertt/train_new.json \
  --retriever_model_path models/Qwen3-Embedding-0.6B \
  --augment_model_path models/Qwen3-1.7B \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_generations 8
```

如果显存紧张，可以把 reward 检索模型放到 CPU，但训练会变慢：

```bash
accelerate launch --num_processes 4 augment/grpo_aug.py \
  --aug_type hybrid \
  --reward_retrieve_type dense \
  --retriever_device cpu \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_generations 8
```

GRPO 通常要求全局 batch size 能被 `--num_generations` 整除。全局 batch size 等于 `per_device_train_batch_size * num_processes`；例如 4 卡、每卡 batch size 为 4 时，全局 batch size 为 16，可以被 `num_generations=8` 整除。

GRPO reward 默认使用 dense/DPR-style 检索。也可以改成 BM25 或 dense+BM25 混合检索，使训练目标和后续推理检索方式更一致：

```bash
# 使用 BM25 检索结果作为 GRPO reward。
python augment/grpo_aug.py \
  --aug_type hybrid \
  --reward_retrieve_type bm25

# 使用 dense 与 BM25 融合后的检索结果作为 GRPO reward。
python augment/grpo_aug.py \
  --aug_type hybrid \
  --reward_retrieve_type hybrid \
  --reward_bm25_weight 0.5
```

默认保存路径为：

```text
models/hybrid
```

训练过程中，脚本会每隔 `--metric_log_steps` 次 reward 调用输出当前 batch 的检索指标，包括 text/table Precision、Recall、NDCG、text/table reward 均值和 query 格式合法率。设置 `--metric_log_steps 0` 可以关闭额外检索日志。

其他 reward 变体：

```bash
# 文本和表格都使用 hard reward 的 joint 训练。
python augment/grpo_aug.py --aug_type joint

# 文本和表格都使用 soft reward 的 joint 训练。
python augment/grpo_aug.py --aug_type joint --soft

# 单独训练 text-only 或 table-only augmentor。
python augment/grpo_aug.py --aug_type text
python augment/grpo_aug.py --aug_type table --soft
```

## 生成增强查询

使用训练后的 augmentor 或 raw augmentor 生成增强查询。重新生成同名文件时建议加 `--overwrite`，避免新旧结果混在同一个 jsonl 中。

```bash
# Dev 集。
python augment/infer_aug.py \
  --dev \
  --name hybrid \
  --path models/hybrid \
  --overwrite

# Test 集。
python augment/infer_aug.py \
  --name hybrid \
  --path models/hybrid \
  --overwrite
```

上述命令会生成：

```text
datasets/multihiertt/dev_hybrid.jsonl
datasets/multihiertt/test_hybrid.jsonl
```

如果需要 raw augmentor baseline：

```bash
python augment/infer_aug.py --dev --name raw_aug --path models/Qwen3-1.7B --overwrite
python augment/infer_aug.py --name raw_aug --path models/Qwen3-1.7B --overwrite
```

## 生成和存储 Embedding

稠密检索和 BM25+稠密混合检索都需要提前存储 embedding。

```bash
# Dev 集文档 embedding 和原始问题 embedding。
python augment/store_emb.py --dev --doc --query

# Test 集文档 embedding 和原始问题 embedding。
python augment/store_emb.py --doc --query
```

如果使用增强查询，还需要存储增强 query 的 embedding：

```bash
python augment/store_emb.py --dev --query_aug hybrid
python augment/store_emb.py --query_aug hybrid
```

默认 embedding 目录结构为：

```text
stored/{uid}/
  doc_embs.json
  query_embs.json
  hybrid_query_embs.json
```

也可以显式指定路径和模型：

```bash
python augment/store_emb.py \
  --dev \
  --doc \
  --query \
  --data_root datasets/multihiertt \
  --stored_root stored \
  --model_path models/Qwen3-Embedding-0.6B
```

## 推理与评测

主问答流程统一使用：

```bash
python main.py [options]
```

加 `--dev` 表示在 dev 集运行，并输出 EM/F1 和检索指标。不加 `--dev` 表示在 test 集运行，并生成 `test_predictions.json`。

### 端到端问答

不进行检索，直接把完整文档输入问答 LLM。

```bash
python main.py --dev
python main.py
```

### 稠密检索

不使用查询增强的 training-free dense retrieval：

```bash
python main.py --dev --retrieve_type dpr --top_k 20 --top_p 0.4
python main.py --retrieve_type dpr --top_k 20 --top_p 0.4
```

使用 HEART 增强查询的 dense retrieval：

```bash
python main.py --dev --retrieve_type dpr --query_aug hybrid --top_k 20 --top_p 0.4
python main.py --retrieve_type dpr --query_aug hybrid --top_k 20 --top_p 0.4
```

使用子表形式组织检索到的表格 cell：

```bash
python main.py --dev --retrieve_type dpr --query_aug hybrid --tabform
python main.py --retrieve_type dpr --query_aug hybrid --tabform
```

### BM25 Baseline

BM25 不需要预先存储文档 embedding。

```bash
python main.py --dev --retrieve_type bm25 --top_k 20 --top_p 0.4
python main.py --retrieve_type bm25 --top_k 20 --top_p 0.4
```

使用增强查询的 BM25：

```bash
python main.py --dev --retrieve_type bm25 --query_aug hybrid --top_k 20 --top_p 0.4
python main.py --retrieve_type bm25 --query_aug hybrid --top_k 20 --top_p 0.4
```

### BM25 + 稠密混合检索

混合检索需要预先存储 dense embedding。`--bm25_weight` 用于控制 sparse 和 dense 分数的插值权重：

- `0.0`：只使用 dense score。
- `0.5`：BM25 和 dense 等权重，默认值。
- `1.0`：只使用 BM25 score。

```bash
python main.py \
  --dev \
  --retrieve_type hybrid \
  --query_aug hybrid \
  --top_k 20 \
  --top_p 0.4 \
  --bm25_weight 0.5
```

```bash
python main.py \
  --retrieve_type hybrid \
  --query_aug hybrid \
  --top_k 20 \
  --top_p 0.4 \
  --bm25_weight 0.5
```

### 标注证据上界

直接使用标注证据构造上下文，主要用于 dev 集分析。

```bash
python main.py --dev --retrieve_type gth
python main.py --dev --retrieve_type gth --tabform
```

### 只评测已有结果

跳过推理，直接评测已保存结果：

```bash
python main.py --dev --eval --retrieve_type dpr --query_aug hybrid
```

## 常用参数

- `--start`, `--end`：运行数据集切片。
- `--batch`：调度 batch 大小。
- `--max_concurrency`：最大并发 QA API 请求数，默认 `16`。
- `--save_root`：结果保存目录，默认 `./results/multihiertt`。
- `--stored_embs_path`：dense embedding 目录，默认 `./stored`。
- `--device`：dense retrieval 使用的设备，默认 `cuda`；如果 CUDA 不可用，会自动回退到 `cpu`。
- `--top_k`：文本 top-k；当 `--top_p 0` 时，也用于表格 top-k。
- `--top_p`：表格检索分数阈值。设置为 `0` 时使用 top-k 表格检索。
- `--tabform`：将检索到的表格 cell 组织为重建后的子表。

## 输出目录

结果默认保存到：

```text
results/multihiertt/{LLM_MODEL_NAME}/{setting}/
  0.json
  1.json
  ...
  test_predictions.json
```

每个样本结果包含 prompt、模型响应、解析后的 prediction，以及检索到的 text/table ids。
