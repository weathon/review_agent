# Multi-path reasoning on a budget: towards theoretically optimal hyperparameter-free adaptive self-consistency

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Self-consistency (SC) is one of the most popular test-time inference techniques for augmenting performance in chain-of-thought reasoning. It consists of generating multiple responses, or ``samples", from a large language model (LLM) and selecting the most frequent answer, which can be viewed as an application of majority vote and mode estimation. Despite its effectiveness, self-consistency is prohibitively expensive at scale when naively applied to datasets. By leveraging the mode estimation and voting theory, we design Blend-ASC, a novel variant of self-consistency that dynamically allocates samples, achieving state-of-the-art sample efficiency. We show that our approach uses 
$6.8\times$  fewer samples on average compared to adaptive and fixed-allocation self-consistency baselines, demonstrating the superiority of our approach in terms of efficiency. We note that Blend-ASC is not only lightweight but also hyperparameter-free, ensuring it can be easily applied to any self-consistency applications. Finally, we derive novel scaling laws, offering a way to predict sample efficiency for a given target error.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
The paper propose a theorical provement of efficient adaptive sc together with the experiments prove the theory.

## A bad news is I am not familar with theory provement, I will only focus on the experiments part. Please reduce the weight of my rating, since it is a theory-based research paper.

### Strengths
The paper give  a

### Weaknesses
Could you briefly clarify the current experimental setup? It seems to be quite confusing for the readers.It might also be helpful to: Compare how 'efficiency' is defined and Compare the final performance under the same sampling conditions (or 'given the same sampling budget').

Could you provide a simple summary in layman's terms? It should explain what signals are used to assist ASC？

Does this require additional time/source to caculate whether to stop?

### Questions
See weakness.

### Soundness
3

### Presentation
3

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
This paper presents a comprehensive theoretical and empirical study of Self-Consistency (SC), a popular test-time inference strategy for reasoning in large language models (LLMs). The authors reinterpret SC as a form of mode estimation and majority voting, deriving tight error bounds and scaling laws that explain its performance across datasets. Building on these insights, they propose Blend-ASC, a novel adaptive and hyperparameter-free variant that dynamically allocates inference samples to questions under a fixed compute budget. Blend-ASC combines the theoretical optimality of PPR-1v1 with the practical efficiency of Adaptive SC, achieving superior sample efficiency—up to 6.8× fewer samples than vanilla SC—across various LLMs (LLaMA-3.2, Qwen2.5) and benchmarks (GSM8K, MATH, MMLU, GPQA-Diamond). Theoretical analysis establishes power-law scaling for dataset-level performance and exponential convergence for aligned questions, both supported by extensive empirical validation.

### Strengths
1, Builds a principled bridge between SC and classical statistical learning theory.

2, Blend-ASC is parameter-free and easy to implement.

3, The derivation of the asymptotic bound is mathematically elegant.

4, Theoretical, synthetic, and real-data analyses align convincingly.

### Weaknesses
1, More related works should be discussed. e.g. https://aclanthology.org/2024.findings-emnlp.135.pdf, https://arxiv.org/abs/2401.02009, https://arxiv.org/abs/2308.00436. For example, at the same cost, does the proposed method perform better than mirror-consistency, self-contrast & self-check?

2,  The dataset-level theoretical analysis relies on idealized margin distributions, which may not perfectly capture real model behavior.

3,  Real-world runtime or energy cost comparisons would enhance practical relevance.

4, While Blend-ASC is efficient, the paper could better explain its qualitative decision process and failure modes under extreme conditions.

### Questions
1, What's the performance comparison between the consistency-based methods and other inference-time methods? e.g. multi-agent systems or other prompting methods like step-back https://arxiv.org/abs/2310.06117. Or let me ask in another way, why should we keep optimizing consistency-based methods, given all other prompting strategies?

2, The method is mainly a prompting engineering work. Can the llm be trained to be better at self-consistency? Which i mean is, the model can be trained to generate with a voting strategy.

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
This paper revisits SC and provides a clean, interpretable bound for SC as a mode estimation method (exponential decay at rate $ (\sqrt{p_1}-\sqrt{p_2})^2$, Theorem 1). In simpler terms, cases where the margin between the top two answers is less take more samples to converge. This “margin” lens also explains dataset‑level power‑law behavior. On the methods side, they take two reasonable ideas: ASC (good early) and PPR‑1v1 (asymptotically optimal late), and blend them into a new method (Blend-ASC) that spends samples where the top-2 are unstable. Empirically, they report ~6.8 times fewer samples than vanilla SC to match accuracy.

### Strengths
* Clear and usable bound: exponential decay with an interpretable margin that practitioners can understand; accurately characterizes behaviors across models and datasets (see Fig. 3). This is, in my view, the most impressive aspect of the paper.

* A blend of two sensible policies: They combine two methods with complementary strengths, ASC for speed early and PPR-1v1 for guarantees late, to achieve good practical results.

### Weaknesses
* The main theory and many plots emphasize aligned items (the majority answer is correct). The misaligned behavior is weaker, and MC is sometimes non-monotonic (Fig. 5). This is not a weakness per se, but it does make the results somewhat less compelling.

* Practical impact is unclear: 
  * How to set budgets, handle latency/cost trade‑offs, partial allocations across streaming datasets, and interaction with caching/batching is unclear. 
  * If a team can only perform “SC@N per query” (without cohorting), is there a lightweight variant of Blend-ASC that still achieves a decent fraction of the gains?

### Questions
* For real-world evaluation, can you report at least one end-to-end LLM experiment with wall-clock and monetary costs to demonstrate that Blend-ASC retains its advantage?






* Suggestion for (exciting) future work: extend the analysis to “thinking” models by replacing additional samples with additional thinking steps, and test whether longer thinking on harder questions outperforms extra votes under realistic cost/latency constraints.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies test-time scaling via self-consistency (SC) sampling for large language models. The authors formalize SC as a majority-vote estimation problem and introduce a margin quantity defined by the gap between the top-2 answer probabilities. Under an aligned assumption—where the correct answer corresponds to the mode—the paper establishes an exponential error decay rate governed by this margin. At the dataset level, the error rate can be expressed as a Laplace transform over the margin distribution, which induces power-law scaling. Building on these insights, the authors propose a compute-allocation strategy (Blend-ASC) that blends an asymptotically optimal but conservative pairwise comparison rule with a more aggressive adaptive allocation heuristic. Experiments based on resampling from limited initial generations suggest that Blend-ASC reduces the number of samples required to match vanilla SC performance on several benchmarks.

### Strengths
### Originality
The paper provides a clean and unified theoretical framing of SC through a classical margin-based lens. Relating dataset-level behavior to the margin distribution helps consolidate several empirical scaling observations. 

### Quality
The theoretical arguments are well-organized, and assumptions are clearly stated. The proposed algorithm is reasonable from a systems perspective and demonstrates consistent improvements under the authors’ experimental protocol. 

### Clarity
The paper is well written, with helpful intuition behind proofs and algorithmic components. The distinction between aligned and misaligned questions is explicitly surfaced, which avoids overstating the theoretical coverage.

### Weaknesses
**Practical utility of the theory**
The exponential decay guarantees require strong alignment assumptions. In settings where the model’s top mode is systematically wrong, margin-based confidence can be misleading, limiting the operational usefulness of Section 4.1.

**Relative novelty and applicability**
Compared to prior adaptive schemes such as ASC and ESC, the proposed approach does not appear to introduce a substantially new conceptual perspective on the core allocation problem; rather, it refines similar margin-based heuristics and inherits some of their limitations. 


**Batch inference assumption**
The algorithm fundamentally assumes a global batch of questions and a shared budget. Many realistic deployments are online or have per-query latency constraints. The paper does not discuss feasibility or degradation in such scenarios.

### Questions
**Misaligned regimes:**
   When the model’s top mode is incorrect, do margin-based confidence estimates systematically misallocate budget? Are there diagnostics or caps that prevent over-exploration of misleading modes?

**Online/latency settings:**
   How might Blend-ASC be adapted when queries arrive incrementally or must be answered individually under strict latency? Is there a small-batch approximation that preserves most gains?

### Soundness
3

### Presentation
3

### Contribution
2
