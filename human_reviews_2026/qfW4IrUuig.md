# LLM-based Automated Theorem Proving Hinges on Scalable Synthetic Data Generation

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Recent advancements in large language models (LLMs) have sparked considerable interest in automated theorem proving and a prominent line of research integrates stepwise LLM-based provers into tree search. In this paper, we introduce a novel proof-state exploration approach for training data synthesis, designed to produce diverse tactics across a wide range of intermediate proof states, thereby facilitating effective one-shot fine-tuning of LLM as the policy model. We also propose an adaptive beam size strategy, which effectively takes advantage of our data synthesis method and achieves a trade-off between exploration and exploitation during tree search. Evaluations on the MiniF2F and ProofNet benchmarks demonstrate that our method outperforms strong baselines under the stringent *Pass@1* metric, attaining an average pass rate of $60.74\\%$ on MiniF2F and $21.18\\%$ on ProofNet. These results underscore the impact of large-scale synthetic data in advancing automated theorem proving.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses automated theorem proving (ATP) in Lean 4 using large language models through a tree search approach. The authors propose two main contributions: (1) a proof state exploration method for synthetic data generation that enforces diversity by using constrained decoding over a curated set of 60 common tactics, and (2) an adaptive beam size strategy that starts with large beam sizes for exploration and gradually reduces them for exploitation. The method achieves 60.74% on MiniF2F and 21.18% on ProofNet under Pass@1, outperforming existing tree search baselines. The approach avoids expert iteration and relies on one-shot fine-tuning with approximately 20 million synthetic proof transitions.

### Strengths
- Well-mentioned Contributions
  - The proof-state exploration method with constrained decoding is clearly motivated and addresses a critical bottleneck in LLM-based ATP. Using constrained decoding to force diversity over 60 curated tactics. The constrained decoding mechanism for forcing tactic diversity is well-motivated and technically reasonable.
  - Adaptive beam size strategy, a simple linear decay function for beam size. The Equation 2 is simple and interpretable
  - DoBeVi tool: A useful engineering contribution but primarily a refactored version of LeanDojo with visualization.
- Clear identification of beam size importance: The analysis in Section 4.3 and Figure 5 clearly demonstrates how beam size affects tree structure and performance, which is an underappreciated factor in prior work
- Honest disclosure of limitations: Section 6 and Appendix A.6 transparently discuss failed attempts (MCTS, scoring functions, tactic/premise separation), which is valuable for the community
- Scalability: Generating 20M training samples and achieving one-shot fine-tuning without expert iteration is practically valuable

### Weaknesses
- The abstract lacks clarity: The phrase *"proof-state exploration approach for training data synthesis"* is vague; readers won't understand what this means without reading the full paper.
- The motivation flow feels disjointed — Section 1 jumps between concepts (tree search vs. whole-proof, data scarcity, EI inefficiency) without a clear narrative thread.
- Figure 1 is confusing: the example showing tactics like `rw`, `rcases`, `have` without explaining what these mean or why they matter is not helpful for a general ML audience (although some are mentioned later).
- Algorithm 1 has notation inconsistencies: it uses `s0, n` and then drops `n`, and mixes implementation details (`q.add`, `lean_prover`) with algorithmic logic.

- No comparison with whole-proof generation methods (e.g., *DeepSeek-Prover-V1.5*).
- No ablation studies isolating the contribution of  
  (a) data synthesis vs.  
  (b) adaptive beam size.
- No analysis of what types of problems benefit most from the approach.

- The scoring function is acknowledged to be flawed (Section 4.3, Appendix A.6: *"significant issues," "nearly indistinguishable"* from baseline).
- Multiple exploration strategies failed (*MCTS, length-normalized BFS, top-p*).
- The paper concludes that *"the scoring function used in current tree search methods still suffers from significant shortcomings"* but offers no fix or alternative.

- No ablation on the 60-tactic set: what happens with 30 or 100 tactics? How sensitive is performance?
- No ablation on pruning parameters: `α = 0.25`, `γ = 0.9`, `β` — are these optimal?
- No ablation separating data synthesis from adaptive beam size: which contributes more to the reported gains?

### Questions
See the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a proof-state exploration framework for automated theorem proving in Lean 4, aiming to improve LLM-based provers through self-supervised data generation.

Starting from existing formal theorems (mainly Mathlib and STP), the authors perform beam-based exploration of proof states guided by a policy model (Qwen2.5-Math-7B). They propose two key components: proof-state exploration with heuristic pruning and adaptive beam-size search.

The resulting synthetic dataset is used to fine-tune the base model, yielding a policy that achieves 60.74 % Pass@1 on MiniF2F and 21.18 % Pass@1 on ProofNet, surpassing several prior Lean-based provers.

### Strengths
1. The paper formulates a well-structured pipeline for data synthesis via proof-state exploration, bridging self-improvement and verifiable proof checking in Lean 4.
2. The reported 60.74 % Pass@1 on MiniF2F is competitive with or better than recent specialized theorem-proving LLMs, suggesting tangible benefits from the proposed training pipeline.
3. The proposed adaptive beam-size strategy which dynamically shrinking the beam with depth is a simple yet effective heuristic that mitigates local-trap issues in tree search and improves computational efficiency without sacrificing success rate.
4. The inclusion of a BLEU-based decontamination step against MiniF2F and ProofNet benchmarks is a necessary step that shows the authors’ awareness of data leakage concerns.

### Weaknesses
My main concern is that the paper does not report the raw (zero-shot) performance of Qwen2.5-Math-7B on MiniF2F or ProofNet. As a result, it is unclear how much improvement stems from the exploration-based fine-tuning versus the inherent strength of the base model. For fair comparison, the authors should either (a) report base model results, or (b) fine-tune the same base models used by prior baselines (e.g., BFSProver) under the same training pipeline.

### Questions
1. Could the authors report the zero-shot performance of the base Qwen2.5-Math-7B model on MiniF2F and ProofNet? This would clarify the absolute contribution of the exploration-based fine-tuning.
2. Did you evaluate how many generated samples were removed during BLEU-based decontamination?  
3. Is the adaptive beam-size heuristic compatible with parallel expansion or batched inference during search, or is it inherently sequential?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a scalable data synthesis framework for LLM-based automated theorem proving. The core idea, Proof State Exploration performs constrained decoding over a curated tactic set to systematically explore intermediate proof states, generating diverse (state, tactic, next state) triples for SFT without relying on inefficient expert iteration. In addition, an adaptive beam size strategy dynamically adjusts search breadth during proof generation, balancing exploration and exploitation.

### Strengths
1. The proposed proof state exploration pipeline provides a well-structured and scalable solution for synthetic data generation and the idea of decoupling tactic exploration from premise generation and forcing low-probability tactic sampling is well motivated
2. The experimental results are sound, with detailed comparisons, ablation on beam size, and transparent computational settings

### Weaknesses
While the method is conceptually strong, it remains largely heuristic.
1. The pruning and beam-decay hyperparameters are fixed ad hoc without sensitivity analysis or further ablations
2. The methodology section is long and very formalized, but lacks intuition for key ideas like why constrained decoding enhances tactic diversity or how pruning parameters (α, β, γ) are chosen and affect synthesis efficiency
3. The scoring function used in tree search still relies on local log-probabilities in Eq. 1, which the paper itself admits to be suboptimal but no quantitative evidence is given for failure case analysis
4. the exploration may remain bounded to “local neighborhoods” of seed datasets, raising concerns about out-of-distribution generalization, especially beyond ProofNet

### Questions
1. Could the adaptive beam control learned instead of manually scheduled, like via a small controller model?
2. I think ProofNet differs significantly from Lean distributions, how does the model handle domain shift, was any domain-specific adaptation adopted?

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
3

### Summary
The paper proposes a scalable synthetic–data generation pipeline for Lean 4 theorem proving by exploring intermediate proof states using constrained decoding, premise completion, and validation through Lean execution, producing roughly 20M high-quality proof transitions for supervised fine-tuning. Combined with an adaptive beam-decay strategy for inference, this approach yields substantial performance gains over strong baselines such as InternLM2.5-StepProver and BFS-Prover on MiniF2F and ProofNet, showing that large-scale, systematically generated synthetic data is key to improving LLM-based automated theorem proving.

### Strengths
- Well written and easy to follow; the illustrative figures and sampled examples are particularly helpful.
- Code is easy to access, and the experiments are reproducible within an academic compute budget.

### Weaknesses
- As a data synthesis strategy, I don’t think it has been critically evaluated: How well does the synthetic data perform under standard search techniques (e.g., greedy decoding)? How much does the exploration component actually contribute to the final proof success rate? An ablation study would help clarify this.
- It would be helpful to include an algorithmic comparison between the newly proposed tree-search method and existing approaches, rather than only reporting raw performance. It would also be valuable to see how the main baseline, BFS-Prover, performs with a slightly larger compute budget.

### Questions
- In Table 1, should I assume that your models are fine-tuned on the synthetic dataset described in Section 3.1, while the baselines InternLM2.5-StepProver and BFS-Prover are not?

### Soundness
2

### Presentation
3

### Contribution
2
