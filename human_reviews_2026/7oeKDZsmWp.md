# Multi-View Encoders for Performance Prediction in LLM-Based Agentic Workflows

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but optimizing LLM-based agentic systems remains challenging due to the vast search space of agent configurations, prompting strategies, and communication patterns. Existing approaches often rely on heuristic-based tuning or exhaustive evaluation, which can be computationally expensive and suboptimal. This paper proposes **Agentic Predictor**, a lightweight predictor for efficient agentic workflow evaluation. Agentic Predictor is equipped with a *multi-view workflow encoding* technique that leverages multi-view representation learning of agentic systems by incorporating code architecture, textual prompts, and interaction graph features. To achieve high predictive accuracy while significantly reducing the number of required workflow evaluations for training a predictor, Agentic Predictor employs *cross-domain unsupervised pretraining*. By learning to approximate task success rates, Agentic Predictor enables fast and accurate selection of optimal agentic workflow configurations for a given task, significantly reducing the need for expensive trial-and-error evaluations. Experiments on a carefully curated benchmark spanning three domains show that our predictor outperforms several strong graph-based baselines in both predictive accuracy and workflow utility, highlighting the potential of performance predictors in streamlining the design of LLM-based agentic workflows.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents Agentic Predictor, a framework for predicting the performance of LLM-based agentic workflows without costly runtime evaluations. It introduces multi-view encoders that jointly model workflow graphs, code structure, and prompt semantics, combined with cross-domain unsupervised pretraining to address label scarcity. Experiments on the FLORA-Bench benchmark show consistent improvements in both prediction accuracy and workflow utility over existing graph-based baselines.

### Strengths
1. The idea of learning to predict workflow performance for LLM agents is interesting and timely, as it could make agent design more efficient.
2. The proposed multi-view representation is intuitively reasonable and well-motivated from both structural and semantic perspectives.
3. The experiments are technically thorough within the chosen benchmark, with clear ablations and implementation details that support reproducibility.

### Weaknesses
The architectural novelty is moderate; most encoder components are adapted from prior NAS and graph-representation literature, with limited methodological innovation beyond combining them.

### Questions
Can the authors clarify how multi-view information is balanced during aggregation? Is there any learned weighting or attention over views beyond simple concatenation?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Agentic Predictor, a performance prediction framework for LLM-based agentic workflows. The core approach employs multi-view workflow encoding: graph view (topology and communication dependencies), code view (implementation, tool usage, and computational logic), and prompt view (system/instruction prompt semantics) are separately encoded and fused through an aggregation layer into a unified representation, followed by training a lightweight prediction head (MLP) for performance classification/regression on limited labeled pairs. In label-scarce scenarios, cross-domain unsupervised pretraining (reconstruction + contrastive learning) is adopted to learn generalizable representations, improving low-sample prediction performance. Experiments are conducted on FLORA-Bench subsets covering three domains: code generation, math problem solving, and commonsense reasoning, reporting consistent improvements in accuracy and utility, along with robustness across different LLM/GNN backbones and OOD scenarios. Ablations show that both multi-view and multi-graph encoding contribute gains, with pretraining being particularly effective under low label ratios.

### Strengths
1. **Novel and well-motivated approach**: Decomposes agentic workflows into three complementary views (graph, code, prompt), balancing structural dependencies and semantic signals, fitting the "heterogeneous, label-scarce" problem nature.
2. **Effective pretraining strategy**: Cross-domain unsupervised pretraining significantly improves prediction quality in low-label scenarios, offering practical sample efficiency advantages.
3. **Efficiency and practicality**: The lightweight predictor serves as a candidate ranker, substantially reducing expensive LLM calls, achieving a "search-agnostic, composable" accelerator paradigm.
4. **Robustness and generality**: Maintains good ranking consistency and accuracy across different LLM/GNN backbones and cross-system/cross-task OOD settings.
5. **Comprehensive experiments**: Main results + view/multi-graph/pretraining ablations + workflow optimization + case studies provide thorough coverage and clear argumentation.

### Weaknesses
**1. Missing Multi-graph Definition**

- The paper uses G_prompt, G_code, G_operator fused via CrossGraphAttn/ViewAttnPool, but does not clarify how the three graphs are constructed: whether they are isomorphic (sharing V, E), the source and representation of each graph's node/edge features (text embeddings, AST/CFG, operator types, etc.), how they are specifically generated from W={V,E,P,C}, and the boundary with FLORA-Bench "following". The absence of formal definitions/pseudocode/illustrations directly impacts reproducibility and contribution assessment.

**2. Insufficient Unsupervised Pretraining Data Description**

- The paper does not specify the scale of unsupervised pretraining data M, source/domain composition, sampling and deduplication strategies, whether there is overlap with test domains/frameworks, and measures to prevent potential label/semantic leakage. These factors directly relate to the credibility and transferability of +pretraining.

**3. Code Encoder Too Black-box**

- Missing details on input representation (specific usage of CodeRankEmbed, whether combined with AST/CFG/call graphs), length normalization/truncation strategies. As the code view is critical to final performance, design details should be supplemented.

**4. Incomplete Cost Analysis**

- While emphasizing zero LLM cost during search, the paper lacks detailed breakdown of one-time training/pretraining time/memory/dollar costs, and comparisons with few-shot LLM classifiers at different scales regarding cost inflection points.

### Questions
1. **Multi-graph construction**: Please formally define G_prompt/G_code/G_operator (whether isomorphic, node/edge features, construction process from W with pseudocode/illustrations).
2. **Unsupervised pretraining data details**: Scale of M, domain distribution, sampling and deduplication, strategies to prevent overlap with test domains/frameworks; if cross-framework (AFlow/G-Designer) or cross-domain mixing exists, please clarify the partitioning and anti-leakage measures.
3. **CodeRankEmbed implementation details**: How does the code view convert source code to vectors (tokenization/truncation/aggregation, whether combined with AST/CFG/call graphs), and why not choose some large-scale pretrained general-purpose embedding models?
4. **Cost clarification**:
   - Explain the basis for "search phase LLM cost is 0" in the paper (is it because candidates come from existing workflow pools rather than LLM generation?).
   - If LLM generation of candidates is needed in real systems, how would the overall cost change? You may discuss this appropriately.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors present a system to predict how well different workflows of agents built on large language models (LLMs) will perform, before actually running them. They encode workflows using three “views” (structure/graph of agents, code logic, and prompts) and pre-train unsupervised across domains, then fine-tune for prediction of success/utility. The method is inspired by recent work in Neural Architecture Search task that effectively evaluates architectures. They show that this multi-view approach improves prediction accuracy and helps guide workflow design with fewer expensive evaluations.

### Strengths
- The motivation (too many workflows, high cost to evaluate) is convincing.
- The multi-view encoding idea is intuitive and makes sense: workflows are complex, so capturing various facets (graph, code, prompt) is a logical approach.
- The pre-training across domains is a good practical touch: many real-world tasks have limited labels, so this should help generalization.

### Weaknesses
In Table 3, it is not clear how the improvement percentages are computed. When comparing Agentic Predictor to the best baseline, the gains appear much smaller. It seems that the authors calculate improvements relative to the simplest baseline (MLP), which could be misleading. Are the other rows in table 3 from the author? Is the MLP the previous state of the art baseline?
Overall, the paper tends to overstate the magnitude of the improvements. The results are positive but more modest than implied.

Minor comments:
In Table 2, please define the abbreviations GD and AF directly in the caption for clarity.

### Questions
see weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Agentic Predictor, a performance prediction framework for LLM-based agentic workflows that aims to reduce expensive trial-and-error evaluations during workflow optimization. The core technical contribution is a multi-view encoding scheme that integrates graph structures, code semantics, and prompt embeddings, combined with optional cross-domain unsupervised pretraining to address label scarcity. Experiments on FLORA-Bench show improvements of up to 12.12% in prediction accuracy and 15.16% in workflow utility over baseline methods across three domains (code generation, math, and reasoning tasks).

### Strengths
The paper addresses a practically important challenge—reducing the computational cost of evaluating agentic workflows—and clearly articulates why existing execution-based approaches are inefficient for workflow optimization. The evaluation is thorough, including ablation studies, low-label regime analysis, OOD generalization tests, and workflow optimization experiments, demonstrating the predictor's effectiveness across multiple dimensions. The method consistently outperforms strong GNN-based baselines (GCN, GAT, Graph Transformer, etc.) with notable margins, and the pretrained variant (Agentic Predictor+) shows clear advantages in label-scarce scenarios.

### Weaknesses
1. The core contribution essentially combines existing techniques (multi-graph GNN, cross-view attention, contrastive pretraining) without fundamental innovation, the multi-view encoding is a relatively straightforward ensemble of three modality-specific encoders, and the pretraining strategy follows standard contrastive + reconstruction objectives commonly used in multi-modal learning.
2. All experiments rely on a single benchmark (FLORA-Bench) with its specific workflow representation format; it remains unclear whether the design choices (three specific views, multi-graph encoding, cross-view attention) would transfer to other agentic frameworks with different architectures, and the OOD tests are still within the same benchmark ecosystem rather than truly novel workflow paradigms.

### Questions
As shown in weakness.

### Soundness
3

### Presentation
3

### Contribution
2
