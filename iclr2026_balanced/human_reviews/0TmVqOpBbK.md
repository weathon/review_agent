## Human Reviewer 1

### Summary
This work builds on the Chinchilla scaling laws by adding architecture-based dimensions like hidden size, MLP-to-attention ratios, and grouped-query attention, to analyse the trade-off between inference efficiency and downstream accuracy. Authors fit a conditional scaling law on >200 Llama-style models (80M-30B #params), authors identify Pareto-optimal configurations and improve Panda/Surefire models that improve downstream accuracy and inference throughput over Llama-3.2 baselines.

### Strengths
- Conditioning the Chinchilla scaling laws on hidden size and MLP-to-attention ratios is timely, useful, and interesting
- Experiments over 200+ trained models provide robust empirical results
- Architectures suggested by the new scaling laws yield sensible gains -- Panda models raise average zero-short accuracy, while Surefire significantly improves the inference throughput by up to ~25%

### Weaknesses
- Results stop at 3B parameters -- what happens at larger model scales?

### Questions
- What happens for larger models? Do the authors have architectural recommendations for these regimes?
- The paper's results are based on vllm -- do results transfer also to e.g. SGLang (https://github.com/sgl-project/sglang)?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper addresses the critical and underexplored trade-off between the accuracy and inference efficiency of Large Language Models (LLMs). While traditional scaling laws focus on training compute, parameters (N), and data (D), they largely ignore the architectural choices that significantly impact inference cost, which is a dominant expense in real-world deployment.

### Strengths
The proposed conditional scaling law is a novel extension of established Chinchilla scaling laws12. By formulating the loss as a separable, calibrated function of architectural ratios relative to a Chinchilla-optimal reference loss, the authors provide a practical tool for architecture design.

Besides, the claims are substantiated by a large-scale empirical study involving over 200 trained models, spanning a wide range of parameter counts (80M to 3B) and token budgets (8B to 100B). This comprehensive dataset provides a strong foundation for fitting the scaling laws.

### Weaknesses
1. The experiments are validated up to 3B parameters. While this is a substantial undertaking, the authors acknowledge that the evaluation does not extend to the 7B+ scale, which is a very common size for deployed open-source models.
2. The entire analysis is confined to dense decoder-only transformers. It is unclear if these findings, particularly the U-shaped loss curves and the separability assumption, would hold for other popular architectures like Mixture-of-Experts (MoE).
3. The study finds that GQA significantly impacts inference efficiency (Figure 9) but has a "highly fluctuating" and inconsistent relationship with training loss (Figure 13). Consequently, GQA is not integrated into the conditional scaling law itself. Instead, it is handled via a separate "local search" after other parameters are optimized. This feels less principled and integrated.

### Questions
The authors note the "highly fluctuating" relationship between GQA and loss (Figure 13)33. Do you have a hypothesis for why GQA behaves so erratically compared to the smooth curves for $d_{model}$ and $r_{mlp/attn}$?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper revisits scaling laws for large language models (LLMs) through the lens of model architecture and inference efficiency.
While existing works (e.g., Chinchilla) focus on training efficiency by relating model parameters, data tokens, and loss, this work introduces architecture-aware scaling laws that explicitly incorporate structural hyperparameters affecting inference cost.

Through a large empirical study (>200 models from 80M–3B parameters trained on 8–100B tokens), the paper shows that performance and inference throughput follow predictable trends governed by these architecture variables. Using these fits, the authors predict “optimal” configurations for new small models (e.g., Panda-1B, Panda-3B) that achieve ~26% higher inference throughput and ~2% higher accuracy than LLaMA-3.2-1B.

### Strengths
1. Novel perspective: Introduces an architecture-aware formulation of scaling laws, bridging the gap between training efficiency and inference efficiency.
2. Extensive empirical coverage: The study includes over 200 model configurations across hidden sizes, attention/MLP ratios, and token budgets.
3. Practical insight: Demonstrates that scaling law fits from small models generalize to larger ones, potentially guiding efficient architecture search.

### Weaknesses
1. Empirical “optimality” is descriptive: The proposed “conditional scaling law” is fitted post-hoc to data rather than derived from optimization principles.
2. Hardware dependence: Reported inference efficiency improvements rely on specific implementation details (vLLM, A100), which may not generalize to other hardware or serving stacks.

### Questions
1. How sensitive are the predicted “optimal” ratios to the hardware platform (A100 vs. H100 or TPU)? Could authors provide several experiments on another hardware?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper addresses a challenge in the efficient deployment of large language models. Whiile existing scaling laws guidfe the trade-off between model size and training data for optimal accuracy, they largely ignore inference efficiency. This work bridges the gap by investigating how architecture hyperparameters impact inference throughput and model accuracy, including hidden size, mlp-to-attention ratio, and GQA.

### Strengths
1. **Novel and practical contribution**: The focus on inference scaling law is highly relevant for the practical deployment of LLMs. The proposed conditional scaling law is a clever and effective extension to the scaling law.
2. **Rigorous and extensive experiments**: The scale of the study is impressive, with over 200 models trained across different parameter scales. This provides a solid empirical foundation for their claims.

### Weaknesses
1. **The design for the conditional scaling law is lack of theoretical support**. The paper doesn't present why the conditional scaling law is fomulated as a multiplicative or additive equation. Why these two forms act similar? Are they still similar if the candidate models are much larger (e.g., 70B+)? Which one should we take in the future research?
2. **Lack of a clear summarization**: It is encouraged to present an summarization on the conclusions about how to design an efficient LLM architecture. It is also encouraged to present an comparison on the design choices of existing open source models about which models are closer to the optimal design.
3. **Lack of analysis on GQA**: Why the relationship between loss and GQA is highly fluctuating? This conclusion is counterintuitive and requires a discussion.

### Questions
The questions are listed in the Weaknesses. I believe this direction is worth researching and this paper will be a good start if the above problems are well solved.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3