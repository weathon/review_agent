# LUMINA: Detecting Hallucinations in RAG System with Context–Knowledge Signals

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Retrieval-Augmented Generation (RAG) aims to mitigate hallucinations in large language models (LLMs) by grounding responses in retrieved documents. Yet, RAG-based LLMs still hallucinate even when provided with correct and sufficient context. A growing line of work suggests that this stems from an imbalance between how models use external context and their internal knowledge, and several approaches have attempted to quantify these signals for hallucination detection. However, existing methods require extensive hyperparameter tuning, limiting their generalizability. We propose LUMINA, a novel framework that detects hallucinations in RAG systems through context--knowledge signals: external context utilization is quantified via distributional distance, while internal knowledge utilization is measured by tracking how predicted tokens evolve across transformer layers. We further introduce a framework for statistically validating these measurements. Experiments on common RAG hallucination benchmarks and four open-source LLMs show that LUMINA achieves consistently high AUROC and AUPRC scores, outperforming prior utilization-based methods by up to +13% AUROC on HalluRAG. Moreover, LUMINA remains robust under relaxed assumptions about retrieval quality and model matching, offering both effectiveness and practicality.

LUMINA: https://github.com/deeplearning-wisc/LUMINA

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes LUMINA, a framework for detecting hallucinations in Retrieval-Augmented Generation (RAG) systems by quantifying external context utilization and internal knowledge utilization. The authors measure external context utilization via Maximum Mean Discrepancy (MMD) between predictive distributions conditioned on retrieved vs. random documents, and internal knowledge utilization by tracking token prediction evolution across transformer layers (information processing rate). The method is validated through statistical hypothesis testing and demonstrates strong performance on RAGTruth and HalluRAG benchmarks across four open-source LLMs.

### Strengths
1.	Novel and principled approach: The decomposition of hallucination signals into external context and internal knowledge utilization is well-motivated. The use of MMD for context sensitivity and information processing rate for internal knowledge are creative and theoretically grounded.
2.	Statistical validation framework: The hypothesis testing approach (Section 3.3) to validate the proposed measurements is a significant contribution. This addresses a critical gap in prior work (e.g., ReDeEP) that lacked validation of whether their scores truly reflect utilization.
3.	Comprehensive experiments: Testing across 4 LLMs, 2 datasets, multiple baselines, and extensive ablations demonstrates thoroughness.

### Weaknesses
1.	Limited theoretical justification for Conjecture 1: The core assumption that hallucinations occur when I_pθ >> E_pθ is presented as a conjecture without formal proof. While empirically validated, stronger theoretical grounding (e.g., information-theoretic analysis) would strengthen the contribution.
2.	MMD approximation concerns: (1) The approximation using only top-100 tokens (Appendix D) may miss important distributional information; (2) No analysis of approximation error or sensitivity to this choice; (3) The choice seems arbitrary without justification.
3.	Random document selection: The paper states d' is obtained from "another data point" but doesn't discuss: (1) How this choice affects results; (2) Whether document similarity between d and d' matters; (3) Potential biases from this selection strategy
4.	Evaluation limitations: (1) No comparison with more recent methods (paper only compares with ReDeEP from 2025); (2) Missing evaluation on longer-form generation tasks; (3) HalluRAG doesn't include Llama3-8B results (unexplained gap)
5.	Computational cost: While claimed to be "lightweight" (Appendix G), requiring two forward passes with full hidden states may still be expensive for very long contexts. No comparison of computational cost vs. baselines.

### Questions
1.	Can you provide theoretical analysis or bounds on the MMD approximation error when using top-100 tokens?
2.	How sensitive is the method to the choice of random documents d'? Have you tried adversarial selection?
3.	Can you ablate the components of Equation (8) to show the contribution of layer weighting vs. entropy normalization?
4.	What is the computational cost comparison (wall-clock time, memory) vs. baselines like ReDeEP?
5.	Why is HalluRAG missing Llama3-8B results?
6.	Can you provide more quantitative error analysis beyond 20 sampled cases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces LUMINA, a framework for detecting hallucinations in Retrieval-Augmented Generation (RAG) systems by jointly modeling external context utilization and internal knowledge utilization. The method assumes that if an LLM effectively leverages external documents, replacing them with random ones should significantly alter the model’s output distribution. Internal knowledge usage is estimated through layerwise token evolution, measured by changes in predicted token distributions across transformer layers. LUMINA integrates these two signals into a unified hallucination score and provides a statistical validation framework. Experimental results across common RAG hallucination benchmarks demonstrate consistent improvements in AUROC and AUPRC compared to prior utilization-based detectors, with robustness under noisy or mismatched retrieval settings.

### Strengths
1.	Clear motivation addressing persistent hallucinations even with correct retrievals.
2.	Simple yet effective framework combining two measurable utilization signals without model retraining.
3.	Comprehensive experiments across multiple LLMs and benchmarks, with robust results under noise and model mismatch.
4.	Practical applicability — LUMINA is lightweight, reproducible, and generalizable to different RAG architectures.
5.	Statistical validation adds methodological rigor to empirical measurements, distinguishing the work from heuristic detectors.

### Weaknesses
1.	Assumption vulnerability — The external-context utilization metric assumes that replacing relevant documents with random ones should change generation probabilities. This fails when the model already knows the correct answer from its parametric knowledge, leading to potential false negatives.
2.	Limited generalization to reasoning tasks — The internal-knowledge utilization signal may not handle complex reasoning cases where correct responses rely on both external and internal information, potentially misclassifying reasoning-based outputs as hallucinations.
3.	Insufficient explanation for robustness — While LUMINA shows stability under noisy retrievals, the paper does not clearly justify why its metrics remain reliable when external documents are irrelevant or partially correct.
4.	Missing theoretical grounding — The framework treats token-level utilization as a direct indicator of truthfulness, but lacks analysis connecting these signals to semantic correctness or factual alignment.
5.	Incomplete discussion of failure cases — The paper could better analyze when and why LUMINA misclassifies or underperforms, especially under low retrieval relevance.

### Questions
1.	How does LUMINA distinguish between cases where the model truly ignores external context and cases where external and internal knowledge simply overlap?
2.	Could the authors provide examples where correct answers yield low context-utilization scores, and explain how such cases are handled?
3.	How might the proposed framework adapt to reasoning tasks that integrate both context and internal knowledge, such as multi-hop question answering?
4.	Why does LUMINA remain robust under noisy retrievals, given that both proposed metrics are highly dependent on the correctness of external documents?
5.	Would integrating interpretability analyses (e.g., attention visualization or neuron probing) strengthen the causal link between the measured utilization signals and factual reliability?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper propose LUMINA, a framework that detects hallucinations in RAG systems with external and internal context
utilization metrics. The external context utilization metric measure the maximum mean discrepancy between two next token probability distributions conditioned on retrieved documents and random documents. The internal knowledge utilization metric introduce the idea of information processing rate by looking at the ratio of the most probable output token’s probability across transformer layers and
use it to determine the amount of utilized internal knowledge when generating the next token. The experiments show that LUMINA outperforms both score-based and learning based methods in hallucination detection.

### Strengths
1. This paper attributes the hallucination problem in RAG systems to LLMs' preference for utilizing internal knowledge versus external retrieval information. It argues that a good RAG system should pay more attention to external retrieval information, and designs two related metrics to quantify the ability to utilize external information and the ability to utilize internal knowledge. The idea is interesting and the metric design logic is sound.

2. This paper presents a framework to statistically validate the metrics, showing the soundness of the evaluation approach.

3. The experimental results show that the proposed evaluation approach works well on relevant benchmarks.

### Weaknesses
1. Although Section 3.2 describes the relationship between internal knowledge utilization and output token probability across transformer layers, it is not clear how the internal knowledge works in black-box LLMs. Could you provide some concrete examples demonstrating that the internal knowledge utilization metric effectively captures the overuse of the internal knowledge?

2. The experiments are conducted on outdated models, with the most recent being Llama 3, which was released 1.5 years ago. Could you provide additional experimental results on newer models such as Qwen 3 or Llama 3.2 to demonstrate the consistent superiority of your approach?

3. How about the evaluation resource cost of LUMINA? There should be a comparison between LUMINA and other related works.

### Questions
1. If a model can use internal knowledge to solve many RAG queries, how should we rate this model under LUMINA?

### Soundness
3

### Presentation
4

### Contribution
3
