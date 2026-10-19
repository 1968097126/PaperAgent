# PaperAgent

面向本地科研论文库的多轮检索问答 Agent，基于 nano-cc 改造。支持追问中的指代消解、混合检索、证据检查、带引用回答和会话记忆，并记录每轮各阶段的输入、输出和耗时。

```text
用户问题 → Guardrails → NLU → Retrieve → Rerank → Reflect → Generate → Memory
                                                 ↓
                                          证据不足时重检或拒答
```

检索使用 BGE-M3 + FAISS、BM25、RRF 和 Cross-Encoder；默认上手示例使用 DeepSeek 的 OpenAI 兼容接口。附带 [20 篇 Agent / LLM 论文清单](corpus/selected_papers.md)及评测数据。

## 快速开始

以下命令在项目根目录运行，适用于 macOS / Linux，建议使用 Python 3.11。

### 1. 安装环境

```bash
python3.11 -m venv --prompt paperagent .venvs/paperagent
source .venvs/paperagent/bin/activate
python -m pip install -r requirements.txt
```

首次下载模型需要网络和数 GB 磁盘空间。`requirements.lock.txt` 保存本次实验环境的精确依赖版本，需要复现实验环境时可用它替代 `requirements.txt`。

### 2. 配置 API

```bash
cp .env.example .env
```

编辑 `.env`，填写自己的密钥；若已有 `.env`，直接编辑即可：

```dotenv
DEEPSEEK_API_KEY=你的密钥
NANOCC_MODEL_CACHE_DIR=.cache/models
```

`deepseek-flash` 使用 `https://api.deepseek.com`。真实问答和模型评测会产生 API 费用。`.env` 已被 Git 忽略。

### 3. 下载论文并建立索引

```bash
python -m corpus.acquire_selected
python -m corpus.build_index --cache-dir .cache/models --device cpu --batch-size 8
```

脚本按固定清单下载全文并提取文本，然后生成分块、FAISS 和 BM25 索引。首次执行较慢；已下载的论文和有效索引会复用。Apple Silicon 可将 `cpu` 改为 `mps`，有 CUDA 的环境可改为 `cuda`。

论文全文、模型权重和索引不随 Git 仓库分发，通过脚本在本地生成。论文的版权和使用条件以原发布方为准。

### 4. 开始问答

```bash
# 交互式多轮对话
python nano_claude.py --model deepseek-flash

# 单次提问
python nano_claude.py --model deepseek-flash -p "Self-RAG 的核心贡献是什么？"
```

可继续追问“它与 Reflexion 有什么区别？”。交互模式中的常用命令：

| 命令 | 用途 |
| --- | --- |
| `/corpus` | 查看语料和索引状态 |
| `/trace` | 查看问答执行记录 |
| `/memory` | 查看会话记忆 |
| `/help` | 查看全部命令 |
| `/quit` | 退出 |

## 测试与评测

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q

# 索引建好后，校验附带的评测数据
python -m eval.validate_datasets

# 完整模型实验：组件、检索、Reflect 与记忆消融
python -m eval.run_rebuild --suite all --device cpu
```

完整实验会调用 API，逐项保存结果并支持断点续跑。输出位于 `eval/results/rebuild-20261019/`；独立重跑时使用 `--output-dir eval/results/my-run`。

本次基线：意图准确率 **80.0%**，混合检索加重排序 Recall@5 **68.0%**，E70 端到端成功率 **57.1%**。数据为模型辅助生成、待人工复核的开发评测集，不代表独立泛化成绩。指标口径、消融结果与限制见 [实验说明](docs/experiments.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `nano_claude.py` | 命令行入口，默认运行论文问答 |
| `pipeline/` | 七阶段编排、记忆和执行记录 |
| `retrieval/` | 分块、向量 / BM25 检索、融合与重排序 |
| `corpus/` | 论文清单、下载和索引构建 |
| `eval/` | 数据、指标、实验与标注复核工具 |
| `tests/` | 自动化测试及离线测试样例 |
| `docs/` | 最终实验说明和结果快照 |

## 项目来源

本项目沿用 [nano-cc](https://github.com/tyronnescott/nano-cc) 的 Agent Harness 思路及已有代码，保留模型接口、工具、权限、压缩与 CLI 能力，增加论文检索和七阶段问答流程。上游增强版本见 [nano-claude-code](https://github.com/SafeRL-Lab/nano-claude-code)。使用 `--coding` 可进入继承的编码 Agent 模式。
