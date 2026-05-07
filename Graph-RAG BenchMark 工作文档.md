# Graph-RAG Benchmark 工作文档

## 1. 方法调研

调研范围：2025 年之后的 Graph-RAG 相关工作，经典方法仅作为必要背景参考。

每条方法只保留以下信息：来源、代码/开源情况、核心问题、方法概括、数据与评测。

### 1.1 GraphRAG 类

#### PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths

- 来源：AAAI 2026；arXiv v1 为 2025-02-18；[arXiv](https://arxiv.org/abs/2502.14902)，[DBLP](https://dblp.org/rec/conf/aaai/ChenGYCCLSY26)
- 代码/开源情况：已开源，[GitHub](https://github.com/BUPT-GAMMA/PathRAG)
- 核心问题：现有 Graph-based RAG 往往检索过多冗余图信息，并以扁平方式拼接到 prompt 中，导致噪声、token 消耗和回答逻辑性问题。
- 方法概括：先用 LLM 从 query 中抽取关键词并检索相关节点，再通过带距离感知的 flow-based pruning 提取关键关系路径，最后把节点和边文本组织成 path-based prompt 供 LLM 生成答案。
- 数据与评测：使用 UltraDomain，覆盖 Agriculture、Legal、History、CS、Biology、Mix 六个领域；与 NaiveRAG、HyDE、GraphRAG、LightRAG 对比；用 GPT-4o-mini 做 pairwise win-rate 评测，维度包括 Comprehensiveness、Diversity、Logicality、Relevance、Coherence。

### 1.2 Agent 类

#### GraphRAG-R1: Graph Retrieval-Augmented Generation with Process-Constrained Reinforcement Learning

- 来源：WWW 2026；arXiv v1 为 2025-07-31；[arXiv](https://arxiv.org/abs/2507.23581)
- 代码/开源情况：已开源，[GitHub](https://github.com/ycygit/GraphRAG-R1)；提供 [Hugging Face 模型权重](https://huggingface.co/yuchuanyue/GraphRAG-R1)
- 核心问题：复杂多跳问题中，GraphRAG 的 query 和 retrieval 阶段常依赖预设启发式规则，容易出现 shallow retrieval；同时 RL/agentic 检索可能出现 over-thinking 和额外检索成本。
- 方法概括：基于支持 rollout-with-thinking 的 GRPO 训练 LLM 自主调用检索工具；设计 PRA 奖励鼓励必要检索并随推理深入衰减检索收益；设计 CAF 奖励在答案 F1 和检索调用成本之间做平衡；训练分为格式冷启动、PRA、CAF 三个阶段；检索侧采用 graph-textual hybrid retrieval。
- 数据与评测：使用 HotpotQA、MuSiQue、2Wiki 作为 in-domain 数据，PopQA 作为 out-of-domain 数据；HotpotQA、MuSiQue、2Wiki 的训练数据按 80%/20% 用于训练和测试；指标包括 F1、SBERT similarity、LLM-as-Judge Accuracy；对比 KGP、ToG、LightRAG、PropRAG、G-Retriever、HippoRAG2、R1-Searcher、Vanilla LLM、Naive RAG、Prompt Engineering、SFT。

### 1.3 Multi-agent 类

暂无条目，待补充。

## 2. Benchmark 调研

调研目标：整理已有 Graph-RAG Benchmark 或 testbed 的评测任务、数据集构成、指标、缺陷，以及可借鉴的设计。

每条 Benchmark 只保留以下信息：来源、评测对象、数据与任务、指标、优点、局限。

### 2.1 已有条目

#### LEGO-GraphRAG: Modularizing Graph-based Retrieval-Augmented Generation for Design Space Exploration

- 来源：PVLDB 18(10):3269-3283, 2025；arXiv v1 为 2024-11-06；[arXiv](https://arxiv.org/abs/2411.05844)，[PVLDB PDF](https://www.vldb.org/pvldb/vol18/p3269-cao.pdf)，[代码和补充材料](https://github.com/gzy02/LEGO-GraphRAG)
- 评测对象：GraphRAG 模块化框架和 testbed；将 retrieval 阶段拆成 subgraph-extraction 和 path-retrieval 两个模块，并将方法分为 structure-based 和 semantic-augmented 两类。
- 数据与任务：使用 Freebase 作为大规模多领域知识图谱，并结合 WebQSP、CWQ、GrailQA、WebQuestions 四个 query 数据集；每个数据集采样 1,000 个测试 query，one-hop 和 multi-hop 比例为 1:1；另用 MetaQA/Wiki-Movies 测试跨图谱和跨领域迁移。
- 指标：质量、效率、成本。GraphRAG instance 主要用 Hits@1，同时报告 F1 和 LLM-based evaluation；效率记录 subgraph-extraction 和 path-retrieval 运行时间；成本记录 token cost 和 peak GPU memory。
- 优点：适合分析不同 GraphRAG 模块组合的质量、效率、成本权衡；可参考其模块级消融和成本记录。
- 局限：主要围绕 KGQA 和已有知识图谱展开；不覆盖文本语料建图、图构建噪声和上下文忠实度。

### 2.2 待讨论问题

- 是否需要对比非 Graph-RAG 方法。
- Benchmark 是更偏“端到端回答质量”，还是更偏“图检索模块质量”。
- 是否要把运行时间、token cost、检索调用次数作为核心指标。

## 3. 数据集准备（最难）

工作目标：汇总调研方法使用的数据集，整理每个数据集的结构特征，并判断是否能够构建黄金子图或黄金 path。

## 4. 图上下文忠实度方法敲定（次难）

工作目标：调研 LLM 和 NLP 领域关于上下文忠实度的评价方法，并确定一个适合 Graph-RAG Benchmark 的图上下文忠实度评价方案。

### 4.1 重点问题

- 最新论文中如何判断 LLM 是否忠实于上下文。
- 需要利用 LLM 的模块或中间向量判断上下文忠实度。
- 哪类方法更容易实现和复现。
- 哪类方法更可能引起研究社区兴趣。
