# Automated Climate Science Fact-Checking System (RAG + CoT LoRA)

本项目是一个面向气候科学领域的端到端自动化事实核查系统。系统通过融合双路检索与重排以及基于思维链微调的大语言模型，从大规模科学证据库中精准检索证据，并判定声明的真实性。

---

## 🌟 STAR 原则项目亮点总结

| 维度 | 详情描述 |
| :--- | :--- |
| **Situation** | 随着气候变化议题的公众讨论日益广泛，网络上充斥着大量未经验证、甚至误导性的言论。针对气候科学陈述进行自动化核查面临两大核心挑战：一是科学语料库庞大且专业性强，单一稀疏或稠密检索难以兼顾关键词精确匹配与语义泛化；二是科学声明存在微妙的逻辑推演（支持、反驳、争议、信息不足），传统分类模型容易出现幻觉与推断错误。 |
| **Task** | 1. **证据检索**：针对给定声明，在知识库（`evidence.json` / SQLite）中召回并精确筛选出 Top-K（如 Top-16 / Top-6）相关证据。<br>2. **声明核查与分类 (Claim Verification & Classification)**：基于检索出的证据，将声明严格判定为 4 类：`SUPPORTS`（支持）、`REFUTES`（反驳）、`NOT_ENOUGH_INFO`（证据不足）、`DISPUTED`（存在争议）。<br>3. **系统性能优化**：在兼顾证据检索 F-score 与声明分类 Accuracy 的调和平均值上取得优异表现。 |
| **Action** | 1. **混合检索管道**：结合 BM25 词法检索与 Sentence-Transformers (`all-MiniLM-L6-v2` + FAISS IVF 索引) 稠密语义检索，并通过倒排融合算法聚合双路结果，大幅提高长尾专业词汇与泛语义的召回率。<br>2. **思维链数据蒸馏**：构建Few-shot科学事实核查专家提示词，引导模型自主生成“逐步逻辑推导（Let's analyze step by step）”的解释与最终结论，构建高质量推理数据集 `cot.json`。<br>3. **4-bit QLoRA 监督微调**：使用 Qwen 系列模型，采用 NF4 4-bit 量化与 LoRA 适配器技术对注意力及投影层进行 SFT 训练，强化其在气候科学领域的 CoT 逻辑判别能力。<br>4. **鲁棒批处理评估体系**：开发左填充批量推理框架与基于正则约束的状态机提取逻辑，支持全链路端到端评测。 |
| **Result** | 1. **检索召回与排序大幅提升**：RRF 融合相比单模型检索显著改善证据重合度与 Precision/Recall 平衡。<br>2. **逻辑分类准确率提升**：经过 CoT 微调的 LLM 不仅输出了结构化、可解释的审查依据，显著降低幻觉率，分类准确率较原始 Zero-shot 基础模型提升明显。<br>3. **工程化与复现性**：提供模块化的 RAG 检索、LoRA 训练、批量评估与端到端测试脚本，代码高度工程化且易于迁移。 |

---

## 📊 数据集介绍 (Dataset Description)

项目涉及训练、开发、测试以及支撑事实核查的核心科学证据库。

### 1. 数据集文件结构
```text
data/
├── evidence/
│   └── evidence.json          # 全量科学证据库，包含海量气候科学事实陈述
├── evidence.db                # SQLite格式证据库，便于按ID快速高并发读取
├── train-claims.json          # 训练集（包含 claim_text, evidences ID列表, claim_label）
├── dev-claims.json            # 验证集（用于全系统指标评估）
├── test-claims-unlabelled.json# 无标签测试集（用于比赛提交与盲测）
└── cot.json                   # 经过蒸馏生成的思维链推理微调数据集
```

### 2. 标签体系
系统对输入的每条声明判断为以下 4 种互斥标签之一：
* **`SUPPORTS`**：证据提供了充分且明确的事实支持，声明内容属实。
* **`REFUTES`**：证据与声明内容直接矛盾或明确推翻了声明中的核心结论。
* **`DISPUTED`**：证据表明该科学议题在学术界/现实数据中存在分歧，或证据中包含部分支持与部分反驳，属于有争议陈述。
* **`NOT_ENOUGH_INFO`**：证据库中缺乏足够的信息来证实或证伪该声明。

### 3. 数据样例 (Data Format Sample)
```json
{
  "claim-1024": {
    "claim_text": "The Earth’s climate sensitivity is so low that a doubling of atmospheric CO2 will result in a surface temperature change on the order of 1°C or less.",
    "claim_label": "REFUTES",
    "evidences": [
      "evidence-4821",
      "evidence-9930"
    ]
  }
}
```

---

## 🏗️ 系统架构设计 (System Architecture)

系统由 **检索层 (Retriever)**、**重排/融合层 (Ranker / Fusion)** 和 **分类判决层 (Reasoning & Verifier)** 三部分组成：

```text
┌─────────────────┐
│   Input Claim   │
└────────┬────────┘
         │
         ├──────────────────────────────────────────┐
         ▼                                          ▼
┌─────────────────────────┐              ┌─────────────────────────┐
│   BM25 Lexical Search   │              │   Dense Vector Search   │
│ (CountVectorizer/TFIDF) │              │ (MiniLM-L6-v2 + FAISS)  │
└────────┬────────────────┘              └──────────┬──────────────┘
         │                                          │
         └────────────────────┬─────────────────────┘
                              ▼
               ┌──────────────────────────────┐
               │ Reciprocal Rank Fusion (RRF) │
               │   Top-K Evidence Selection   │
               └──────────────┬───────────────┘
                              ▼
               ┌──────────────────────────────┐
               │    Prompt Template & CoT     │
               │ (System + Evidences + Claim) │
               └──────────────┬───────────────┘
                              ▼
               ┌──────────────────────────────┐
               │     Qwen LLM (4-bit QLoRA)   │
               │   "Let's analyze step by..." │
               └──────────────┬───────────────┘
                              ▼
               ┌──────────────────────────────┐
               │   Label Extraction & JSON    │
               │ {Label, Evidences, Reasoning}│
               └──────────────────────────────┘
```

---

## 🛠️ 环境依赖与安装

推荐使用 Python 3.10+ 和具有 CUDA 驱动的 GPU 环境：

```bash
# 克隆仓库
git clone <repo_url>
cd climate-fact-checker

# 安装基础与深度学习依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft bitsandbytes datasets trl sentence-transformers faiss-gpu scikit-learn scipy tqdm
```

---

## 🚀 快速使用指南

### 1. 生成 CoT 推理数据与 LoRA 模型微调
```bash
# 步骤 1: 自动生成 CoT 推理链数据集
python src/fine_tuning/generate_cot.py

# 步骤 2: 使用 QLoRA 对 Qwen 模型进行监督微调 (SFT)
python src/fine_tuning/train_lora.py
```

### 2. 验证集评估
运行评估脚本计算检索 F-score、分类 Accuracy 以及两者的调和平均数H-Mean：
```bash
python src/evaluator.py
```

### 3. 测试集生成预测文件
针对未标注的测试集生成符合比赛规范的提交文件：
```bash
python src/test_inference.py
```
生成的预测结果保存在 `results/test-output.json`，格式如下：
```json
{
  "claim_001": {
    "claim_text": "...",
    "claim_label": "REFUTES",
    "evidences": ["evidence-12", "evidence-89"]
  }
}
```

---

## 📈 评估指标 (Evaluation Metrics)

系统通过三大核心指标进行全方位综合衡量：
1. **Evidence Retrieval F-score ($F$)**：检索出的证据集合与标准答案集合的精确率与召回率的调和均值。
2. **Claim Classification Accuracy ($A$)**：声明标签判定准确率。
3. **Harmonic Mean ($H	ext{-}Mean$)**：
   $$\text{Harmonic Mean} = \frac{2 \times F \times A}{F + A}$$
