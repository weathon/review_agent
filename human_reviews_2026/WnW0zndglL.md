# UNITE: Universal kNowledge Integration from Task-specific Experts

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Large language models (LLMs) with Mixture-of-Experts (MoE) architectures achieve strong performance under sparse activation. However, their expertise is often fragmented across experts and redundant across layers. Prior studies primarily diagnosed redundancy or parameter importance, revealing overlaps but lacking mechanisms to transform them into reusable knowledge. In contrast, human learning succeeds not by memorizing isolated facts but by reusing shared strategies across domains, which motivates the question: do MoE models similarly encode universal knowledge that can be systematically extracted and reused? We propose Universal kNowledge Integration from Task-specific Experts (UNITE), a framework that consolidates experts through Fisher-weighted fusion and then applies Tucker decomposition to disentangle shared low-rank input/output subspaces as universal knowledge from layer-specific variations. This universal component provides a compact basis for reconstructing target models with flexible depth, enabling lightweight yet competitive adaptation across tasks. To assess effectiveness, we evaluate data efficiency, convergence speed, and generalization across multiple MoE-based LLMs and diverse datasets. The results show that UNITE not only extracts universal knowledge, but also flexibly enabling once-for-all extraction and flexible target model construction that generalize across domains.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes UNITE, a framework that fuses MoE experts via Fisher-weighted aggregation and applies Tucker decomposition to extract shared low-rank subspaces that encode universal knowledge, from which it reconstructs a dense model training only small per-layer cores.

### Strengths
The paper is clearly written and well-organized, with fluent language that make the presentation easy to follow. The contribution sounds novel.

### Weaknesses
The method introduces Fisher estimation and Tucker decomposition. What is the efficiency of these operations? The authors should provide time-efficiency comparisons against baselines.

Does the method depend on the calibration dataset? Can it remain robust under different calibration datasets?

MoE models enjoy the advantage of activating only a small fraction of parameters at inference. After integration, how does the inference time of the resulting model compare to the original—does it lose this advantage?

Seems that the method can be viewed as a form of model distillation from an MoE teacher to a dense student. Could the authors include experimental comparisons against standard distillation baselines?

### Questions
The authors claim that universal knowledge is captured by the matrices U_o, U_i. What metrics or evidence substantiate this claim?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the challenge of fragmented and redundant expertise within Mixture-of-Experts (MoE) Large Language Models (LLMs). It argues that, similar to human learning, MoE models likely encode transferable, task-agnostic knowledge (universal knowledge) that is obscured by the layer-wise redundancy and expert sparsity. UNITE systematically extracts and reuses universal knowledge from MoE-based LLMs. The method involves: Intra-layer Consolidation (with Fisher-weighted fusion )and Cross-layer Factorization(with Tucker decomposition). Experiments on three MoE-based LLMs (Mixtral, DeepSeek-MoE, Qwen3-MoE) demonstrate that UNITE-based models consistently outperform randomly initialized baselines.

### Strengths
1. The recognized issues of fragmented and redundant knowledge in MoE models is an interesting topic.
2. The paper thoroughly evaluates UNITE across three MoE models (Mixtral, DeepSeek, Qwen3), multiple benchmarks (ARC, HellaSwag, PIQA, etc.), and three key criteria: generalization, data efficiency, and convergence speed. Ablation studies validate each component (Fisher fusion, Tucker decomposition, rank, depth).

### Weaknesses
1. The motivation is not clear to me. I did not understand what "overlap" is between LLM
2.  The definition is not clear. For example, what is the definition of knowledge?

3. "By analogy, MoE-based LLMs may also exhibit similar properties: although each expert is designed to specialize, their parameters are not entirely independent and may contain overlaps, redundancies, and recurring transformations that implicitly capture transferable patterns." I am not sure the experts are specialized in moe-structure. 

4. The computational cost of Fisher information calculation and Tucker decomposition on large MoE models is not discussed. This could be a practical barrier for very large models, and the paper does not analyze the trade-off between extraction cost and downstream benefits.

### Questions
## questions

1. What is the computational and memory overhead of the Fisher information calculation and Tucker decomposition steps, especially for larger MoE models (e.g., Qwen3-MoE-30B)? Is the extraction process scalable to models with hundreds of experts?

2. What are the advantages compared to the Tucker decomposition of the CP decomposition?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes UNITE, a framework that extracts universal knowledge from MoE-based LLMs and uses it to initialize smaller dense models for downstream tasks. The method applies Fisher-weighted fusion to consolidate experts within each layer, then uses Tucker decomposition to extract shared projection matrices and layer-specific cores across layers. Further experiment results shows the effectiveness of the proposed method.

### Strengths
1. Extracting universal knowledge from MoE models is a compelling research direction. 
2. The experiments effectively demonstrate the method's effectiveness across several MoE architectures and benchmarks.

### Weaknesses
1. **Limited technical novelty**: Both Fisher information weighting and Tucker decomposition are well-established techniques. The main contribution is their combination for MoE models, which feels incremental. The paper does not sufficiently differentiate itself from prior work on model compression, knowledge distillation.
2. **Critical architectural details missing**: The paper focuses on MoE feed-forward layers but omits key implementation details. How are embedding and attention layers initialized in target models? When constructing a model with L'=2 layers from a source with L=32 layers, which layers are selected and how? 
3. **Unfair baseline comparisons**: The paper compares UNITE models (initialized from state-of-the-art MoE models like DeepSeek-16B and Qwen3-30B) against outdated baselines such as T5-Base, BERT-Large, and GPT-2. Since the source MoE models benefit from years of architectural improvements and much larger, higher-quality pretraining data, the performance gains may reflect source model quality rather than effective knowledge extraction. Fairer comparisons would include recent similarly-sized models.

### Questions
Another point is how the Qwen3-based 2-layer model has only 360M parameters when the embedding layer alone should exceed this? Please provide a complete parameter breakdown,including embeddings, attention, and MLP components.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
### Briefly summarization of **UNITE**

UNITE introduces a novel framework for extracting and reusing universal knowledge from Mixture-of-Experts (MoE) based large language models. The authors observe that while MoE architectures enable sparse activation and scalability, they also lead to fragmented expertise and redundancy across layers. Inspired by how humans generalize domain-specific experiences into transferable reasoning skills, UNITE aims to uncover shared, task-agnostic structures embedded within expert parameters.

### Main Contributions of **UNITE**

- **Universal Knowledge Extraction**  
  First framework to *systematically extract* **task-agnostic, reusable knowledge** from MoE-based LLMs by combining **Fisher-weighted fusion** and **Tucker decomposition**.

- **Efficient Downstream Adaptation**  
  Enables **once-for-all** extraction → build **lightweight target models** of *any depth* without retraining, achieving **strong performance with fewer parameters**.

- **Superior Performance**  
  Outperforms **pretrained baselines** (e.g., BERT, GPT-2) and **compression methods** on diverse tasks (science, commonsense, NLI) with **+6–12% accuracy gains**.

- **Improved Training Efficiency**  
  Boosts **data efficiency** (no large pretraining needed) and **convergence speed** (fewer epochs to reach peak performance).

- **Theoretical Justification**  
  Provides **generalization bounds** showing Fisher-weighted fusion reduces variance and Tucker decomposition lowers model complexity.

### Strengths
1. Novel & Timely Idea
First to frame “universal knowledge” inside sparse MoE transformers and give a concrete, tensor-decomposition recipe to extract it—directly relevant as trillion-parameter MoEs proliferate.

2. Solid Empirical Sweep
Tests three different MoE back-bones (Mixtral, DeepSeek, Qwen3) on seven varied benchmarks; ablates every major design choice (fusion weighting, decomposition rank, target depth); consistently beats strong pretrained and compression baselines while using ¼–⅓ the parameters.

3. Theoretical Backing
Supplies generalization theorems that justify Fisher-weighted fusion as minimum-variance unbiased estimator and Tucker compression as Rademacher-complexity reducer—rare for a systems-oriented NLP paper.

### Weaknesses
1. Downstream Tasks Skew toward QA/CLS
Evaluation focuses on multiple-choice and sentence-pair classification; no generation, long-context, or multilingual benchmarks, leaving extrapolation to other modalities unclear.

2. Fisher Computation Overhead
Requires a full forward–backward pass on a calibration dataset to estimate expert importance; could be expensive for very large MoEs and is sensitive to the choice of that dataset (only WikiText-2 is tested).

3. Rank & Depth Chosen Empirically
Tucker ranks and target depths are grid-searched on validation sets; no principled way to predict optimal compression given a compute budget, which limits “off-the-shelf” adoption.

### Questions
1. Generation & long-context evaluation
All reported downstream tasks are either multiple-choice QA or sentence-pair classification. Could you provide results on generative objectives (e.g., summarization, open-ended QA) or long-context benchmarks to verify that the extracted universal bases remain useful when the output space is unconstrained or the input length grows beyond the calibration window?

2. Calibration cost & data sensitivity
The Fisher-information estimates require a full forward–backward pass over a calibration corpus. For trillion-parameter MoEs this can be prohibitive, and only WikiText-2 has been tested. How does the quality of the extracted bases vary with (a) calibration size, (b) domain shift, and (c) cheaper approximations such as diagonal Fisher or gradient-free importance metrics? Is there a “minimal sufficient” calibration set or an online variant that amortizes the cost?

3. Rank/depth selection without grid search
Tucker ranks and target depth are currently chosen by grid search on validation accuracy. Is there a theoretically grounded or compute-budget-aware procedure (e.g., eigen-gap heuristic, MDL principle, or a single-shot sensitivity analysis) that predicts the optimal compression ratio before any downstream fine-tuning, so that practitioners can deploy UNITE in an “off-the-shelf” fashion?

### Soundness
3

### Presentation
3

### Contribution
2
