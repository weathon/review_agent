# NIRVANA: Structured Pruning Reimagined for Large Language Models Compression

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Structured pruning of large language models (LLMs) offers substantial efficiency improvements by removing entire hidden units, yet current approaches often suffer from significant performance degradation, particularly in zero-shot settings, and necessitate costly recovery techniques such as supervised fine-tuning (SFT) or adapter insertion. To address these critical shortcomings, we introduce NIRVANA, a novel pruning method explicitly designed to balance immediate zero-shot accuracy preservation with robust fine-tuning capability. Leveraging a first-order saliency criterion derived from the Neural Tangent Kernel under Adam optimization dynamics, NIRVANA provides a theoretically grounded pruning strategy that respects essential model training behaviors. To further address the unique challenges posed by structured pruning, NIRVANA incorporates an adaptive sparsity allocation mechanism across layers and modules (attention vs. MLP), which adjusts pruning intensity between modules in a globally balanced manner. Additionally, to mitigate the high sensitivity of pruning decisions to calibration data quality, we propose a simple yet effective KL divergence-based calibration data selection strategy, ensuring more reliable and task-agnostic pruning outcomes. Comprehensive experiments conducted on Llama3, Qwen, and T5 models demonstrate that NIRVANA outperforms existing structured pruning methods under equivalent sparsity constraints, providing a theoretically sound and practical approach to LLM compression.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work suggests a theoretically organized pruning strategy for LLMs called NIRVANA (Neural Tangent Kernel-InfoRmed adaptiVe neuron & AttentioN heAd pruning), which strikes a balance between robust fine-tuning capabilities and immediate zero-shot accuracy preservation.

### Strengths
- The paper is well written, with smooth flow and clear sectioning.
- Connects pruning to NTK stability by deriving a bound $|\tilde{\Theta}-\Theta|\le O(\epsilon)$ (Eq. 3), linking pruning to training dynamics under Adam/SignGD, an uncommon contribution among structured pruning works.
- Experiments is sufficient. Tested across multiple model families and architectures.

### Weaknesses
- There is no formal explanation for why decreasing KL over calibration subsets ensures superior pruning results, but empirical results (Fig. 3) showing a linear link between KL divergence and perplexity.
- Adam and SignGD kernel equivalence ($\Theta_{\text{Adam}}\approx \Theta_{\text{Sign}}$) is assumed in the theoretical connection, there should be some experimental validation provided.
- Provide a non-LM architecture (like ViT) or talk about the changes that should be made to NIRVANA may expand it beyond text-based LLMs.
- Some relevant unstructured/semistructured baselines like SparseGPT or MaskLLM could further contextualize performance trade-offs.

### Questions
- To what extent does the optimizer selection affect the NTK-based saliency? 
- How much more work is required to compute NTK-guided saliency than normal gradient-based saliency? 
- Is it possible to use the KL-based calibration selection into dynamic or online pruning scenarios?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes NIRVANA, an innovative method for structured pruning of large language models (LLMs), aiming to balance zero-shot accuracy preservation and fine-tuning adaptability. Key contributions include:
1. a neural tangent kernel (NTK)-based saliency score derived from Adam optimization dynamics;
2. an adaptive sparsity allocation mechanism across layers and modules (attention vs. MLP); 
3. a KL-divergence-based strategy for calibration data selection. Experiments on Llama3, Qwen, and T5-base models demonstrate that the proposed method generally outperforms baseline approaches such as LLM-Pruner and SliceGPT under comparable sparsity constraints, offering a theoretically rigorous and practical solution for LLM compression.

### Strengths
1. Weight-level Saliency Scoring is essentially a single judgment that simultaneously achieves two goals: minimizing the current output perturbation while maximizing the NTK stability of the subsequent stage.
2. Calibration data selection based on KL divergence, rather than merely selecting data based on quality, quantity, or diversity by human intervention.

### Weaknesses
1. The KL-based data selection requires multiple pruning trials, which may be computationally expensive for large models or datasets.
2. The paper is overly lengthy; integrating some appendix content into the main body could improve structure and readability.

### Questions
1. In Table 3, it is counterintuitive that at 20% and 40% sparsity, nearly all traditional pruning strategies show higher latency than the full-precision model (except NIRVANA), while at 50% sparsity, all methods exhibit lower latency. Additionally, LLM-Pruner shows the highest latency at 20% and 40% sparsity, yet again ranks highest at 50%-these statistical anomalies raise concerns about data reliability.
2. In Table 6, at 40% sparsity, NIRVANA underperforms LLM-Pruner on certain metrics for Qwen2.5-7B, while achieving the best results in other cases. Does this variability indicate potential limitations in the robustness and generalizability of NIRVANA across different models and settings?
3. This paper involves the use of randomly selected data sets, and the group with the lowest KL divergence is selected as the optimal calibration data. Why can this set of random data be guaranteed to be optimal?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose NIRVANA, a Neural Tangent Kernel (NTK)-guided structured pruning framework for large language models. To ensure pruning decisions align with model optimization dynamics, NIRVANA derives a first-order saliency criterion under Adam-based NTK analysis. It further introduces an adaptive sparsity allocation mechanism that dynamically distributes pruning ratios across layers and between Attention and MLP modules, and a KL divergence–based calibration data selection strategy to enhance pruning robustness and decouple pruning quality from calibration dataset size.

### Strengths
* The paper proposes a novel NTK-guided structured pruning framework that integrates adaptive sparsity allocation and KL-divergence-based calibration data selection.

* Extensive experiments on Llama3, Qwen, and T5 models show that NIRVANA achieves good performance.

### Weaknesses
* Idea of using NTK in pruning is not new, for example [NTK-SAP](https://openreview.net/pdf?id=-5EWhW_4qWP), authors should highlight the difference in the method section and add the comparison in ablation studies.
* Tables 12–17 present qualitative samples, but the descriptions are insufficient, and no corresponding objective quantitative metrics are provided, making it difficult to assess their quality and reducing the persuasiveness of the results.
* The authors derive an optimal $\gamma \approx 3.36$ theoretically and find the empirical optimum to be $3.0$ , showing close consistency, but this validation appears to be conducted only on a single model and lacks testing across multiple models.

### Questions
* As shown in Table 1, for MATH problems like **SVAMP** and code generation tasks like **MBPP**, performance drops sharply when sparsity reaches 40% (SVAMP: 72.67 → 11.67, MBPP: 48.60 → 2.20). How can this sudden drop in these tasks be explained, and does the improvement in throughput truly justify such a significant performance degradation?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces NIRVANA, a structured pruning method for large language models that aims to balance zero-shot accuracy preservation with fine-tuning adaptability. The key contributions include: (1) a Neural Tangent Kernel (NTK)-guided saliency criterion that considers training dynamics under Adam optimization, (2) an adaptive sparsity allocation mechanism that differentially prunes MLP neurons vs. attention heads using a theoretically-derived ratio γ, and (3) a KL divergence-based calibration data selection strategy. The method is evaluated on Llama3, Qwen, and T5 models, showing improvements over existing structured pruning approaches like LLM-Pruner, SliceGPT, and FLAP across various sparsity levels (20-50%). But then I found the evaluation is limited to outdated baselines and smaller models, with significant performance degradation even at moderate sparsity levels.

### Strengths
The NTK-based saliency criterion provides a principled theoretical foundation connecting pruning decisions to training dynamics under Adam optimization, distinguishing it from purely heuristic methods.

The integration of three components (NTK-guided saliency, adaptive sparsity allocation, and KL-based calibration selection) forms a cohesive end-to-end solution that addresses multiple aspects of structured pruning.

The paper is well-written with clear motivation, consistent notation, and good visual aids (particularly Figure 1). The workflow from weight-level saliency to structured pruning is easy to follow.

The component-wise analysis effectively demonstrates the contribution of each proposed element, particularly the importance of the adaptive ratio γ and calibration data selection.

The practical insight about dimension alignment (multiples of 8) for GPU optimization provides valuable guidance for real-world deployment.

### Weaknesses
The experimental evaluation relies exclusively on methods from 2023-2024 (LLM-Pruner, FLAP, SliceGPT) while omitting numerous recent advances including SlimLLM (ICML 2025), Olica (ICML 2025), Týr-the-Pruner (NeurIPS 2025), LoRAPrune (ACL 2024), LLM-Barber (ICCAD 2025) and LoRAP (ICML 2024). This significantly undermines claims of advancing state-of-the-art.

Even when outperforming baselines, the method shows severe performance degradation at practical sparsity levels. At 20% sparsity, WikiText-2 perplexity increases from 8.5→13.4; at 50% sparsity it reaches ~49. Downstream tasks lose 20-40% accuracy and code generation collapses to near-zero, raising serious questions about real-world applicability.

Experiments are confined to relatively small models (≤8B parameters) with minimal calibration data (32 samples). No evaluation on larger models (70B+), MoE architectures, or comparison with modern quantization methods that achieve near-lossless compression.

While the NTK connection is interesting, the practical implementation remains similar to existing gradient-based importance metrics. The theoretical analysis (Proposition 4.1) relies on strong assumptions that may not hold in practice. Any analysis when it'll fail?

The ablation study focuses primarily on 50% sparsity where models already exhibit significant degradation, making it difficult to isolate component contributions. Missing analysis of pruning patterns across layers and computational overhead quantification.

Lacks verification of claimed NTK stability, distribution analysis of saliency scores across layers, and comparison of pruning decisions with simpler criteria like magnitude-based methods.

### Questions
See also weakness.

1. Given the rapid advancement in LLM compression, why were recent state-of-the-art methods (SlimLLM, Olica, Týr-the-Pruner, etc.) not included in the comparison? How would NIRVANA perform against these contemporary approaches?

2. Given the significant performance degradation observed even at 20% sparsity, what are the envisioned real-world applications? How does this compare to modern 8-bit/4-bit quantization methods that maintain near-lossless performance?

3. Have you evaluated NIRVANA on larger models (≥13B parameters) or MoE architectures? What is the computational overhead of NTK-guided scoring compared to simpler alternatives as model size increases?

4. Can you provide empirical evidence of NTK stability preservation (e.g., eigenvalue distribution, kernel alignment) before and after pruning to validate Proposition 4.1?

5. What is the quantitative computational cost of KL-based calibration selection? How do pruning patterns distribute across layers, and does the method successfully avoid layer collapse?

6. Could you provide ablation studies at moderate sparsity levels (20-30%) where models maintain reasonable functionality to better isolate individual component contributions?

7. How sensitive is performance to the adaptive ratio γ across different architectures? Does the theoretically derived value generalize beyond Llama-family models?

### Soundness
2

### Presentation
3

### Contribution
2
