# Critical attention scaling in long-context transformers

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
As large language models scale to longer contexts, attention layers suffer from a fundamental pathology: attention scores collapse toward uniformity as context length $n$ increases, causing tokens to cluster excessively, a phenomenon known as rank-collapse. While $\text{\emph{attention scaling}}$ effectively addresses this deficiency by rescaling attention scores with a polylogarithmic factor $\beta_n$, theoretical justification for this approach remains lacking.

We analyze a simplified yet tractable model that magnifies the effect of attention scaling. In this model, attention exhibits a phase transition governed by the scaling factor $\beta_n$: insufficient scaling collapses all tokens to a single direction, while excessive scaling reduces attention to identity, thereby eliminating meaningful interactions between tokens.
Our main result identifies the critical scaling $\beta_n \asymp \log n$ and provides a rigorous justification for attention scaling in YaRN and Qwen, clarifying why logarithmic scaling maintains sparse, content-adaptive attention at large context lengths.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper attempts to justify the choice of attention scaling factor used in long-context scenario from the theoretical perspective. Specifically, the authors identified that with scaling choice $\beta_n=\gamma\log n$, the attention dynamics demonstrate a transition phase.  To delve into this question, the authors adopt a simplified self-attention model (without parameterisation) for theoretical analysis from both forward/backward perspective under two separate assumptions. The final numerical analysis justifies the derived theory.

### Strengths
This paper is well-motivated that the choice of $\beta_n$ scaling in Qwen, SSMax, SWAN-GPT ($\log n$) is not justified from the theoretical perspective. To address this question, the authors derive their theory on a simplified self-attention network, which is more tractable from the theoretical perspective. The authors consider both forward and backward two directions, which are validated through numerical studies. I believe the theoretical framework could be beneficial to the community.

### Weaknesses
Since the main paper is built on the theoretical analysis on an over-simplified self-attention network (without parameters), the generalisation of the conclusion to the realistic transformer model remains unknown. I have concerns about two main issues in the theoretical framework:

1. This paper still discusses self-attention under the scope of LLMs, which are assumed to adopt causal mask in the attention computation. However, this paper seems to completely ignore this point in the proof (from proof 1 that the denominator is independent of the choice of $i$). I am aware that the authors are trying to understand the attention scaling by simplification, but I am wondering whether the conclusion is still valid considering causal mask in the framework.

2. Another practical consideration in LLM is the existence of attention sink [1], which means the first token has large attention score while its value state is almost zero [2, 3]. Additionally, [3] claimed that this is a way to mitigate over-smoothing in long context scenarios. I am seeking for the authors' opinion towards this as this challenges some assumptions in this paper, e.g., the assumption on $a_{ii}=1$ and $a_{ij}=\rho$.


References:\
[1] Xiao et al. Efficient streaming language models with attention sinks. ICLR 2024.\
[2] Gu et al. When Attention Sink Emerges in Language Models: An Empirical View. ICLR 2025.\
[3] Barbero et al. Why Do LLMs Attend to the First Token? COLM 2025.

### Questions
See the weakness.

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
2

### Summary
The paper analyzes a long-context self-attention layer in which the attention scores are scaled by a factor that grows with the logarithm of the context length. In a simplified pre-normalization setting where the key, query, and value matrices are all equal to the identity, the authors prove that this scaling produces a clear phase transition as the context length increases. In the subcritical regime, when the scaling factor grows too slowly, attention weights become almost uniform across tokens, leading to strong contraction and rank collapse.
In the critical regime, when the scaling grows at just the right rate, attention focuses on a small but non-trivial subset of tokens, resulting in sparse and content-adaptive mixing. In the supercritical regime, when the scaling grows too quickly, attention effectively reduces to the identity and cross-token interactions vanish.

### Strengths
Although I am not too theoretically grounded myself, I very much liked this paper and I hope some of my comments/suggestions are useful to improve the paper. I have added some strong points below.

- The paper is mathematically strong and offers a well-grounded analysis of a practically important issue of rank collapse in long-context transformers.
- The key result is simple yet powerful, providing a clear and interpretable explanation of why logarithmic scaling works and what happens when the scaling constant is too small or too large.
- I particularly like that the analysis connects both forward and backward dynamics. The insight that vanishing gradients appear in the subcritical regime helps explain and even motivate architectural design choices such as residual connections and normalization layers in transformer blocks.
- The work successfully bridges theoretical and practical perspectives, with immediate implications for stable long-context training.

### Weaknesses
- As in many theoretical studies, the analysis operates in a simplified regime (pre-norm, identity key/query/value matrices, no positional encoding, no multi-head or causal masking). This is understandable but limits direct applicability.

- Experiments are synthetic and single-layer; there are no demonstrations on pretrained or small-scale real models. Even a lightweight validation could have been an interesting addition, though I recognize that this is beyond the paper’s main focus.

### Questions
- Do you view the phenomenon described here as primarily length-driven token uniformity (present even in a single layer)? How does this relate to depth-driven rank collapse? If logarithmic scaling is applied, is the length-driven component largely neutralized, leaving only depth-driven collapse? Are there diagnostics that could distinguish the two mechanisms in trained models?
- Your Theorems 2.3 and 2.4 show phase transitions in Jacobian norms, strongly reminiscent of the vanishing-gradient and over-smoothing analyses in graph neural networks (Arroyo et. al. 2025). Can your results be interpreted as the transformer analogue of these phenomena? Do you see conceptual parallels between the two settings?
- Recent work on attention sinks argues that attention sinks emerge as a way to prevent mixing. Do you agree with the interpretation that such sink tokens act as a compensatory mechanism to stabilize mixing when attention heads operate near the subcritical regime? 

Arroyo, Álvaro, et al. "On vanishing gradients, over-smoothing, and over-squashing in GNNs: Bridging recurrent and graph learning." arXiv preprint arXiv:2502.10818 (2025).

Barbero, Federico, et al. "Why do LLMs attend to the first token?." arXiv preprint arXiv:2504.02732 (2025).

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
As language models have grown, context lengths have increased while signal propagation in attention layers has degraded; phenomena such as rank collapse have been observed. Empirically, prior work tunes the inverse temperature $ \beta $ using a $(\log n)^k $ scaling (with $k>0$ varying across papers). Under a strongly simplified model (all attention weights set to the identity), this paper analyzes tractably when a phase transition at $ \beta=\gamma\log n $ occurs and characterizes its connection to rank collapse.

### Strengths
* The angle‑parameter characterization is clear, and comparing to $1/(1-\rho)$ gives a natural condition for the phase transition.
* The numerical simulations align with the theoretical predictions and provide empirical support.

### Weaknesses
* **Insufficient precision in comparisons to prior work.** The introduction distinguishes the REM setting as “dense,” but that distinction applies at the $a_{ij}$ stage; at the $A_{ij}= \exp(\beta a_{ij})$ stage, the  behavior can become sparse, so the separation is not clearly justified. The discussion is thin on why REM suggests  $(\log n)^{1/2}$ while the present model yields $\log n$.  
* **Strength of assumptions.** The score assumption,  exact one dominant entry, others equal (off‑diagonals fixed at  $\rho$ before normalization),  abstracts away near‑ties that can naturally occur in self-attention in real data: in language inputs the same token (e.g., a subject) can reasonably appear multiple times in one sequence.  This also stems from assuming the weights are identity mappings; In the REM case,  generally, the number of  near-tie entries can be more than one.
* **Gap to implementation.** Evidence is limited for how conclusions transfer to systems with learned $(K,Q,V)$. It remains unclear which architectural components would break the conclusions; in particular, the connection between the observed phase transition here and constants or regimes discussed in prior work (e.g., $C$ of $\beta=C \log n$ in Table 1) is weak.
* **Independence of numerical validation.** Experiments closely mirror the theoretical assumptions, limiting their independence as evidence for practical application. Stability of the critical behavior under slight deviations from the assumptions and reproducibility in more implementation‑like settings are underexplored. (e.g. incleasing the number of the near-tie entries)
* **Tone of claims.** While the $ \beta\asymp\log n $ phase‑transition picture is clear, the evidence is not yet sufficient to claim “clarifying  why logarithmic scaling maintains sparse, content-adaptive
attention at large context lengths.” for general practical attention scaling in my current understanding. The scope and limits of applicability should be stated more explicitly.

### Questions
* **Size of the top set.**  Because of $K=Q=I$, we get exactly one large  socre, others small scores after applying softmax with the Assumption 1.  How is Assumption 1 justified for self-attetoin with $K=Q=I$? Prior citations may support it, but a fresh statement of the authors’ view would help.   Any empirical evidence that appearance of exactly one large score in deep model (initialization/pretrain/finetuning phase) would also be helpful.

* **Consistency of $ \log n $ vs  $\sqrt{\log n}$.** Which specific assumption differences (e.g.\,correlation structure, sparsity/density/ layer norm) could determine the order of the critical scale? 
* **Robustness to assumptions.** For non‑identity weights, to what extent do conclusions (including near‑ties) preserve the phase diagram and the critical behavior? Under the stated constraints, in what sense does the work clarify the claim that “logarithmic scaling maintains sparse, content‑adaptive attention at large context lengths”? Also, how do the authors explain $ (\log n)^2 $ (YaRN) scalings?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a rigorous theoretical analysis of attention scaling in long-context transformers. By studying a simplified but informative self-attention model, the authors show that the system undergoes a sharp phase transition determined by a critical scaling factor $\beta$ proportional to $log(n)$. This result provides a clear theoretical explanation for the empirical success of scaling strategies used in recent models such as YaRN, SSMax, and SWAN-GPT. The analysis addresses both the contractive properties of the forward pass and gradient behavior in the backward pass, demonstrating that this critical scaling is necessary to preserve meaningful token interactions and maintain stable training dynamics.

### Strengths
The main strength of the paper lies in the quality of its theoretical analysis. The results are rigorous and presented in a way that remains accessible. The work provides a direct and mathematically grounded explanation for a technique currently used in state-of-the-art large language models, which is a valuable and rare contribution. Moreover, by isolating the mechanisms that lead to performance degradation in long-context tasks, the paper offers a framework that can guide the development of more targeted scaling strategies.

### Weaknesses
The model necessarily incorporates simplifying assumptions, and the experiments are correspondingly limited. However, given that the goal of the work is to isolate and explain the underlying scaling phenomenon, already well studied by practitioners, rather than to propose a new architecture or strategy, this scope seems appropriate. The limitations are acknowledged and do not undermine the core contribution.

### Questions
- The phase transition boundary depends on geometric quantities such as the inner product ρ and the norm q. The simplex case makes the nature of this dependence particularly transparent. Recent work has shown that layer normalization can systematically influence token norms over depth (e.g., “Normalization in Attention Dynamics,” arXiv:2510.22026). This raises the question of whether an adaptive scaling factor, adjusting $\beta$ based on observed per-layer statistics of token norms or pairwise similarities, could maintain the model closer to the critical regime and potentially improve performance?
- Assumption 2 requires a positive lower bound $\rho_1 > 0$ on pairwise inner products. While the upper bound $\rho_2 < 1$ is intuitive, the necessity of a positive lower bound is less obvious. Is this bound essential for the phase transition to appear, or is it primarily a technical condition to simplify the proof? Additional intuition here would be helpful.
- The analysis is conducted under the simplifying assumption $Q = K = I$, which is a standard and effective choice for achieving tractability and is consistent with related theoretical work. As a curiosity, do you observe the same phase transition behavior in numerical experiments when $Q$ and $K$ are non-identity matrices, for example in the case where $Q^TK$ is symmetric positive definite? If so, does the location of the phase transition or the strength of contraction change in a predictable way?

### Soundness
4

### Presentation
4

### Contribution
4
