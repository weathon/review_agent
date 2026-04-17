# Cat-PO: Cross-modal Adaptive Token-rewards for Preference Optimization in Truthful Multimodal LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Multi-modal Large Language Models (MLLMs) have shown remarkable generative capabilities across multi-modal tasks, yet remain plagued by hallucinations where generated textual contents are semantically inconsistent with the input images. This work reveals that existing multi-modal preference optimization methods exhibit shortcomings at the preference data decoding stage. Specifically, different response tokens exhibit varying degrees of association with visual content, and consequently, their contributions to reducing hallucinations and generating high-quality responses differ. Nevertheless, most existing methods do not distinguish in their treatment, often handling them uniformly. To address this challenge, we propose a novel preference alignment method: Cross-modal Adaptive Token-rewarded Preference Optimization (Cat-PO). Building upon direct preference optimization, Cat-PO calculates hierarchical visual relevance rewards for each response token at global, local, and semantic levels. It then organically integrates these three rewards to construct a smooth reward mechanism and designs an innovative KL-based customized loss for rewarded tokens, thereby enabling fine-grained correction of hallucinatory outputs. Extensive experiments on various base models and evaluation benchmarks demonstrate that our Cat-PO can significantly reduce hallucinations and align with human preferences to enhance the truthfulness of MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes CAT-PO (Cross-modal Adaptive Token-rewards for Preference Optimization), a token-level preference optimization method for multimodal LLMs aimed at reducing visual hallucinations while improving truthfulness. Building on DPO, the authors compute hierarchical token relevance from model-internal signals: (i) global relevance as the sum of cross-modal attention over all visual patches; (ii) local relevance via patch-entropy (lower entropy ⇒ more localized visual focus); and (iii) semantic relevance using a cosine similarity between token and vision-conditioned text/visual representations. These three signals are fused (with a smooth nonlinearity) into per-token rewards that modulate DPO and a token-level KL regularizer (reward-aware KL) to discourage visually ungrounded continuations.
Experiments on AMBER (disc/gene), MM-Hal, LLaVA-Bench, and SEED primarily with LLaVA-v1.5-7B/13B and Qwen variants show consistent improvements, notably 7–15% gains on AMBER-Gen/MM-Hal and reduced hallucination rates. The approach adds modest overhead (+38% per-sample time; ~+0.07% peak memory).

### Strengths
1. **Methodological clarity**. Token-level reward design from internal signals (attention, entropy, similarity) that are cheap to obtain.
2. **Consistent empirical gains**. 7–15% improvements on AMBER-Gen/MM-Hal, and competitive on AMBER-Disc。

### Weaknesses
1. **Heuristic fusion lacks learning/theory**: The tanh-based smoothed fusion and fixed mixing coefficients are plausible but not justified.
2. **Limited evaluation**: While standard suites are covered, edge-case diagnostics are missing.
3. **Reward-gradient stability not analyzed**. No examination of gradient variance when many tokens get high rewards.

### Questions
1. **Learned fusion and fixed fusion**. Did you try a learned gate like 2-layer MLP over [S_global, S_local, S_sem] or a Dirichlet-style mixture with simplex constraints? How do gains compare and how stable is training?

2. Several highly relevant works are not covered or discussed in the current manuscript.

[1]Token-level Direct Preference Optimization.
[2]CHiP: Cross-modal Hierarchical Direct Preference Optimization for Multimodal LLMs
[3]TGDPO / T-Guided DPO: Harnessing Token-Level Reward Guidance for Preference Optimization

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
4

### Summary
This paper proposes Cat-PO, which introduces a token-based hierarchical reward mechanism on the foundation of standard DPO to achieve more refined hallucination suppression. It fully leverages the MLLM's own architecture (without requiring external tools) to compute the relevance of each response token to the image from three perspectives: global relevance, local relevance, and semantic relevance. These three scores are integrated into a unified relevance score, allowing tokens with strong visual relevance to receive higher rewards in the chosen response and higher penalties in the rejected response. Finally, the authors demonstrate the effectiveness of their approach through a series of experiments.

### Strengths
1. The authors present a clear motivation for their method, with a well-justified problem statement. It effectively addresses the issue of token-level fine-grained alignment in DPO, thereby enhancing its efficiency.

2. The experiments are sufficiently comprehensive, evaluating the method from both hallucination suppression and general capabilities. The ablation studies are also thorough.

3. The paper is clearly written and easy to understand.

### Weaknesses
**My primary concern regarding this work is the potential risk of circular reasoning in its methodological motivation.** If an MLLM has already incorrectly allocated high attention to a hallucinated object (e.g., misidentifying a chair as a dog and focusing on that region), Cat-PO may mistakenly judge the corresponding tokens as "highly relevant." This would inadvertently reinforce the incorrect generation, potentially cementing or even amplifying the erroneous attention pattern. Consequently, rather than mitigating hallucinations, this approach could exacerbate specific types of them. I would raise my score if the authors could address this issue, either through a theoretical discussion or via visualizations that demonstrate how their method remains robust in such scenarios.

### Questions
1. The current method is a black-box "automatic scoring" approach but does not provide any interpretability analysis. Could it be that high-frequency words such as "the" or "a" receive excessive rewards due to coincidental high attention scores?

2. The experiments may involve unfair comparisons. For instance, Cat-PO introduces multiple hyperparameters, which might have been meticulously fine-tuned, whereas hyperparameters of other methods may not have received the same level of adjustment.

3. How does the computational cost compare to standard DPO?

### Soundness
3

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
This paper tackles the persistent issue of hallucinations in multimodal large language models (MLLMs) by introducing Cross-modal Adaptive Token-rewarded Preference Optimization (Cat-PO). 
The authors observe that existing preference optimization methods, such as DPO, fail to account for varying visual relevance among response tokens. Cat-PO addresses this by assigning hierarchical, token-level rewards based on global, local, and semantic visual relevance, integrated into a KL-regularized DPO framework. 
Experiments on multiple benchmarks show that Cat-PO substantially reduces hallucinations and enhances factual alignment, with extensive ablation studies validating its contributions.

### Strengths
1. Fine-grained Token-level Rewarding: The core innovation lies in Cat-PO’s token-level reward granularity. By assessing each token’s global, local, and semantic relation to the input image, the model updates its preferences more precisely than aggregate reward schemes.

2. No External Dependencies: Cat-PO requires no external detectors or APIs. It fully leverages the pretrained MLLM’s own components (e.g., CLIP+ViT and the base LLM), ensuring simplicity and low overhead.

### Weaknesses
1.Generalization Claims: Although Cat-PO is presented as a general framework, experiments are confined to LLaVA, with no tests across different architectures or open-domain datasets.

### Questions
1. Learnable Weighting: Would learning the fusion coefficients in Equation 7 improve adaptability or stability?
2. Comparative Evaluation: How does Cat-PO compare empirically or theoretically with TARS[1], CHiP[2] and AMP[3]?

[1] TARS: MinMax Token-Adaptive Preference Strategy for Hallucination Reduction in MLLMs
[2] CHiP: Cross-modal Hierarchical Direct Preference Optimization for Multimodal LLMs 
[3] Automated multi-level preference for mllms

### Soundness
3

### Presentation
3

### Contribution
3
