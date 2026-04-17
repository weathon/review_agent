# Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Masked Diffusion Models (MDMs) offer flexible, non-autoregressive generation, but this freedom introduces a challenge: final output quality is highly sensitive to the decoding order. We are the first to formalize this issue, attributing the variability in output quality to the cumulative predictive uncertainty along a generative path. To quantify this uncertainty, we introduce Denoising Entropy, a computable metric that serves as an internal signal for evaluating generative process. Leveraging this metric, we propose two algorithms designed to optimize the decoding path: a post-hoc selection method and a real-time guidance strategy. Experiments demonstrate that our entropy-guided methods significantly improve generation quality, substantially boosting accuracy on challenging reasoning, planning, and code benchmarks. Our work establishes Denoising Entropy as a principled tool for understanding and controlling generation, effectively turning the uncertainty in MDMs from a liability into a key advantage for discovering high-quality solutions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper defines path uncertainty for MDMs and proposes Denoising Entropy, state entropy, and path entropy. Using these notions of uncertainty, it introduces two path-search methods (greedy selection method / SMC-based method). The authors also provide experimental evidence for these methods.

### Strengths
The authors propose principal metrics and simple algorithms to use them for determining the unmasking order of MDMs. These methods indeed surpass the previous heuristic-based unmasking order methods.

### Weaknesses
The paper does not present budget-fair comparisons under equal NFE/compute (e.g., matching total denoising function evaluations or wall-clock), making it hard to disentangle algorithmic gains from increased sampling/particle budgets. Although the authors include semi-AR sampling in their baseline, I believe it would've been fairer if the semi-AR + other confidence methods, e.g., semi-AR + prob margin, were also included. Also, the paper's notion of entropy isn't that different from previous papers', in the sense that they still leverage the information of per-position logits. Given all these, I don't think the paper's contribution (at least under the current manuscript) is substantial.

### Questions
Please refer to the weakness section!

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Denoising Entropy (DE)—a principled metric for quantifying uncertainty in Masked Diffusion Models (MDMs). The authors define State Entropy (hDE) and Path Entropy (HDE) to measure local and cumulative uncertainty along decoding trajectories and propose two corresponding algorithms: Entropy-based Best-of-N (E-BON) and Entropy-guided Sequential Monte Carlo (E-SMC). These methods leverage DE to guide decoding towards more stable, low-uncertainty generative paths. Theoretical analysis establishes hDE as an upper bound on the joint entropy of masked tokens and as a proxy for instantaneous training loss. Experiments show consistent gains in text, reasoning, planning, and code benchmarks.

### Strengths
1. **Elegant and Principled Framework.** The formulation of *path uncertainty* and *denoising entropy* is mathematically clean and conceptually satisfying. It connects diffusion decoding with information-theoretic measures in a natural way.
2. **Strong Theoretical Justification.** The proofs that hDE bounds joint entropy and approximates per-token loss are technically sound, providing a rare formal underpinning for an uncertainty metric in MDMs.
3. **Simple but Effective Algorithms.** E-BON and E-SMC are straightforward extensions of existing sampling schemes but grounded in a clear optimization objective. Their connection to sequential Monte Carlo provides interpretability and generality.
4. **Clear Empirical Improvements.** Across multiple reasoning and code benchmarks (GSM8K, GPQA, Sudoku, Countdown, HumanEval), the proposed entropy-guided search yields consistent performance gains over strong baselines such as PC-Sampler and P² sampling.
5. **Broader Implications.** The notion of *path-level uncertainty* could serve as a foundation for calibrated or self-evaluating diffusion decoders, potentially extending to planning, editing, or reinforcement-guided diffusion.

### Weaknesses
**Empirical Scope and Novelty of Gains.** While the method consistently improves over baselines, the absolute improvements (typically 1–2% accuracy gains on large reasoning models) may be modest given the added complexity.

**Limited Computational Analysis.** The results primarily focus on accuracy or perplexity; runtime or budget trade-offs for E-SMC versus simpler strategies are not quantified, though SMC is known to be resource-intensive.

### Questions
- Does evaluating “path uncertainty” (their Denoising Entropy) require running full decoding simulations, which could be expensive?

- P2 is only shown in Figure 4 but is missed in Figure 2. Can the authors add P2 to Table 2 benchmark?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Masked Diffusion Models (MDMs) offer multiple sampling paths during inference, unlike Autoregressive Models. Prior work has shown that choosing the correct sampling path can significantly affect performance, but most existing methods rely on locally greedy strategies. This paper proposes optimizing the decoding path using denoising entropy, defined as the average of the entropies of all masked tokens along the path. To achieve this, the method maintains K particles (candidate paths) during generation and steers them toward low-entropy paths using Sequential Monte Carlo and then ultimately selects the best candidate path according to the denoising entropy. The paper provides a theoretical explanation of the approach and presents extensive experiments on language modeling and several reasoning benchmarks.

### Strengths
- The paper studies an important problem — optimizing the decoding path in Masked Diffusion Models — and introduces a relatively simple method that scales inference-time computation to achieve notable performance gains.

- The paper also provides extensive experiments evaluating the proposed method across a variety of tasks. It is also nice that the paper attempts to offer a theoretical justification for the proposed approach (though, as noted below, there are some issues with it).

### Weaknesses
- Propositions 1 and 2 rely on several assumptions that require further justification. In particular, both propositions appear to assume $p_{\theta}(x_0^{\ell} | z_t) = q(x_0^{\ell} | z_t)$. This is a strong assumption in the context of optimizing decoding paths for MDMs. If the learned posterior equals the true data posterior, then all decoding paths would yield the same distribution (see page 7 in [1] for an explanation). This makes the assumption a bit unrealistic, and the practical implications of the results remain unclear.

- The paper’s writing could be improved in several places. For instance, the proof of Proposition 1 (lines 825–830) seems to rely on the assumption that the learned posterior equals the data posterior, but this assumption is not explicitly stated in Proposition 1. Additionally, the oracle state entropy is defined in the appendix but is used in Proposition 1. There are also minor typos (e.g., “L” on line 803).

- As pointed out in [2], MDMs can exploit the generative perplexity metric by repeating high-frequency words. Therefore, the entropy metric from [2] should also be reported in Table 1 to ensure that the proposed method is not exploiting this issue.

- For decoding strategies such as confidence, margin, and EB-sampler, it is standard practice to use a smaller block size (e.g., 64) and apply these strategies within each block — as done in the LLaDA paper. However, this paper uses the full block size, which likely causes a significant performance drop for the baselines.

- Most of the performance gains in the reasoning and planning experiments come from the Countdown task, while other tasks show only marginal improvements.

[1] Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

[2] Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling

### Questions
See weaknesses section.

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
This paper studies the problem of decoding from masked diffusion models (MDMs). They make the observation that the quality of the final output is highly sensitive to the order with which the tokens are decoded in an MDM. The authors attribute the variability in output quality to the cumulative predictive uncertainty along the different generative paths. In an attempt to quantify such uncertainty, the authors introduce *denoising entropy*. The authors then develop two decoding algorithms designed to optimize the decoding path based on *denoising entropy*, a post-hoc filtering approach as well as a real-time guidance strategy. They find that the proposed entropy-guided decoding significantly improve generation quality.

### Strengths
- The paper is fairly well-written with an abundance of figure to aid with the communication of ideas

- The authors show that their approaches E-BoN and E-SMC manage to increase the average accuracy when paired with any suite of decoding approaches.

- The authors provide a theoretical justification for their newly proposed decoding approach

- The experiments cover a range of tasks and models

### Weaknesses
- The authors use GPT2 to evaluate the perplexity of generations. I expected to see the perplexity reported using a much more capable LM given the current landscape.

- As the authors might know, in LLMs, greedy decoding and beam search, which by definition attempts to approximate the lowest-uncertainty generation path, tends to perform quite poorly and are typically avoided as a decoding approach. I would've expected a discussion of how one is to consolidate these well established finding in the autoregressive language modeling community with the findings of the paper.

- Given the argument that authors are attempting to make, I would have expected an impractical beamsearch baseline that (potentially) always upperbounds their more tractable entropy criterion.

- Typos:
-> line 029 "Unlike ARMs rely"

### Questions
- In the introduction, "High cumulative uncertainty harms output consistency, while low uncertainty indicates reliable paths". Is this a heuristic that has been shown to work in practice and has been established apriori, a heuristic that the authors are putting forth, or something that can be ascertained?

- Could you please walk me through the definition of the *Oracle State Uncertainty* and why it is intractable to compute? I would have expected us to be more interested in deriving bounds on the entropy/uncertainty of a decoding path (I also find the bound perhaps too loose to be very useful, unless the authors manage to convince me otherwise). Furthermore, proposition 2 only makes the case that state entropy is a proxy for the MDM loss, but fails to make a strong case (to my understanding) for why it is a good criterion to optimize for while decoding. I also refer the authors to [1] which studies the delicate balance between accuracy and calibration in LLMs (which might potentially be applicable in MDMs)

-Have the authors considered running a form of beam search to further validate their claims that states with the lowest entropy have the highest quality? It could perhaps be a more expensive upper bound of their approach, and would make for a more convincing argument. 

References:

[1] Mark Braverman, Xi Chen, Sham Kakade, Karthik Narasimhan, Chiyuan Zhang, & Yuhuai Zhang. (2020). Calibration, Entropy Rates, and Memory in Language Models. In Proceedings of the 37th International Conference on Machine Learning (ICML 2020).

### Soundness
3

### Presentation
3

### Contribution
3
