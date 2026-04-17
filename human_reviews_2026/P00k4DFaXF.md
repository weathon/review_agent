# Process-Verified Reinforcement Learning for Theorem Proving via Lean

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 8

## Abstract
While reinforcement learning from verifiable rewards (RLVR) typically has relied on a single binary verification signal, symbolic proof assistants in formal reasoning offer rich, fine-grained structured feedback. This gap between structured processes and unstructured rewards highlights the importance of feedback that is both dense and sound. 
In this work, we demonstrate that the Lean proof assistant itself can serve as a symbolic process oracle, supplying both outcome-level  and fine-grained tactic-level verified feedback during training. Proof attempts are parsed into tactic sequences, and Lean's elaboration marks both locally sound steps and the earliest failing step, yielding dense, verifier-grounded credit signals rooted in type theory. We incorporate these structured rewards into a GRPO-style reinforcement learning objective with first-error propagation and first-token credit methods that balances outcome- and process-level advantages.
Experiments with STP-Lean and DeepSeek-Prover-V1.5 show that tactic-level supervision outperforms outcome-only baselines in most settings, delivering improvements on benchmarks such as MiniF2F and ProofNet. Beyond empirical gains, our study highlights a broader perspective: symbolic proof assistants are not only verifiers at evaluation time, but can also act as process-level reward oracles during training. This opens a path toward reinforcement learning frameworks that combine the scalability of language models with the reliability of symbolic verification for formal reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper uses Lean as a verifier for process oracle rewards (which are tactic-level rewards in the case of theorem proving). It then incorporates it into a GRPO-style reinforcement learning objective with first-error propagation and first-token credit methods that balances outcome- and process-level advantages. The authors report that on MiniF2F and ProofNet, the proposed approach outperforms both outcome-only RL and vanilla baselines.

### Strengths
- The paper is well-written and easy to understand. This includes a dedicated formalization of the problem and clear mathematical representation.
- The approach is straightforward. Once the additional process rewards are defined, incorporating them with GRPO seems standard and I believe reproducible.

### Weaknesses
- My major concern is novelty. Leveraging feedback from ITPs has long been known to (and practiced by) the ML for theorem proving community. The paper doesn’t offer much other than demonstrating that combining it with GRPO did bring some benefits empirically. This is particularly concerning because the tactic-level rewards being considered here essentially only distinguish between “applicable” and “inapplicable” tactic applications. While useful, they are not novel signals.

### Questions
Have you thought about other more fine-grained tactic-level rewards?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes using the Lean proof assistant as a symbolic process oracle during reinforcement learning training for theorem proving. The key idea is to parse generated proofs into tactic sequences and leverage Lean's elaboration feedback to assign tactic-level rewards in addition to outcome-level rewards. The authors integrate these structured rewards into a GRPO-style objective with first-error propagation and first-token credit assignment. Experiments on STP-Lean and DeepSeek-Prover-V1.5 show modest improvements on MiniF2F and ProofNet benchmarks over outcome-only baselines.

### Strengths
1. Clear formulation and motivation: The paper provides a well-structured formalization of Lean as a process oracle, defining parsing function f, outcome function g, and tactic-level scoring function φ. The MDP formulation is clear and the integration with GRPO is technically sound.

2. Comprehensive ablation studies: The authors conduct thorough ablations on token-level credit assignment (Table 3), reward strategies (Table 4), timeout settings (Figure 3), and hyperparameter sensitivity (Table 5, Appendix C), demonstrating careful experimental work.

3. Stability analysis: The comparison with outcome-only GRPO shows that tactic-level supervision provides more stable training dynamics, with analysis of entropy and proof length providing useful insights.

### Weaknesses
1. **Fundamental semantic limitation without learned PRM comparison (Critical)**: The paper's most significant limitation is that Lean's tactic-level feedback only provides **syntactic/type-theoretic correctness**, not **mathematical progress or strategic value**. As the authors acknowledge in Appendix H (imo_2019_p1 example), tactics that elaborate successfully (e.g., introducing hypotheses h_1 through h_9) receive no penalty despite being mathematically unproductive. The method only penalizes the final failing tactic (omega), missing that the entire proof strategy was flawed from the beginning.

   More critically, the paper does not compare against learned process reward models (e.g., Lightman et al., Wang et al.) that could potentially capture mathematical progress beyond syntactic verification. While the authors claim no Lean tactic-level dataset exists, this actually highlights the core limitation—**symbolic verification captures local soundness but misses global strategic value that learned PRMs aim to model**. The absence of this comparison makes it impossible to assess whether the symbolic approach is fundamentally limited compared to learned alternatives, or whether combining both approaches could address the semantic gap.

2. **Marginal empirical improvements**: The performance gains are quite limited:
   - MiniF2F: +1.2-1.4% at pass@32, +2.5% at pass@64
   - ProofNet: +1.4% at pass@32, essentially no gain (-0.1%) at pass@64
   
   These improvements are within typical variance ranges and raise questions about whether the added complexity is worthwhile. The gains are particularly underwhelming considering that the method requires calling Lean's parser and elaborator during training.

### Questions
The main questions and concerns are detailed in the Weaknesses section above.

### Soundness
3

### Presentation
2

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
Authors demonstrate Lean proof assistant's fine grained feedback on proofs can be harnessed as tactic level process rewards that produce improved RL training results. They compare outcome only, process only, and their hybrid method and show improved training dynamics. They demonstrate solid results on miniF2F and ProofNet and contribute detailed ablation studies for several design decisions.

### Strengths
- thorough definitions and descriptions of their procedures
- clean idea that leverages the additional signal afforded by powerful proof assistants
- thorough ablation studies confirming the significance of various design decisions

### Weaknesses
- analysis limited to full proof generation-- search and iterative repair remain important techniques in automatic theorem proving
  - this point is slightly biased towards the human perspective, but theorem proving is often understood as a search problem

- nit: reward definition hard codes values given to "valid steps" before incorrect steps. This measure of validity is only in the syntax sense, right? The tactic is not necessarily contributing to a correct proof. The empirical analysis with d1/d2 shows that it should be treated distinctly from syntactically incorrect tactics which makes sense, but beyond this ordering, selecting a fixed value doesn't seem grounded.

### Questions
- has there been any opportunities to further scale training? the differences between different conditions seem fairly small on the outcome reward chart in figure 2, and RL is infamously unstable (even in the R1 paper poor trends (negative/plateau) are observable for hundreds of steps before another rise). I totally understand if it's a resource limitation, but curious to see if the patterns highlighted here hold at scale.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper studies reinforcement learning for formal mathematics in Lean. It proposes to view Lean compiler feedback not just as a binary final reward, but as more fine-grained process supervision: a large penalty is assigned to the tactic that produces the first error message. Continued RL experiments on DeepSeekProver-V1.5 and STP models suggests that this method performs better than standard outcome-only RL.

### Strengths
- The method is sensible and clearly described, the experimental setup is sound, the experimental results are good.
- The problem of credit assignment is highly relevant: both in code generation and (natural language + formal) math, the state of the art uses outcome-only reinforcement learning, which forgoes the finer, localized signal that compilation error messages or generative reward models provide.

### Weaknesses
- The presentation is overly convoluted (e.g. suggestion: remove TacSet etc, write: "viewing a proof $Y$ as a sequence of tactics $(T_1, ..., T_n)$ parsed from the AST") and lacks crucial details (which values for $d_1$ and $d_2$ are chose? which one is larger?).
- The method could be grounded more theoretically: first come up with a credit assignment scheme, then explain that any token-level advantage sequence can be integrated into GRPO. The optimal way of assigning credit is via $r_t + V(s_{t+1}) - V(s_t)$ where $V$ represents the expected value under the policy $\pi$. Rewards $r_t$ are still sparse at the end of the sequence. Under the modeling assumption "no recovery after the first error" that the authors rightly put forward in the context of Lean, $V(s_{t+1})$ is zero after the first error. $V(s_0)$ is estimated from the $n$ rollouts in GRPO. Inbetween, there is incomplete information to fill in for $V$ precisely, but the constant extrapolation suggested in the paper is likely suboptimal. Under a model of independent Bernoulli-distributed error variables at each tactic, the value $V$ should *increase* over the progression of the trajectory until reaching a value of 1 upon success or until the first error. I do not expect the resulting assignment scheme to produce fundamentally different behavior, but it would be a more theoretically sound derivation of a scheme similar to the one used in the paper. See: potential-based reward shaping.

### Questions
- values for $d_1$ and $d_2$?

### Soundness
3

### Presentation
3

### Contribution
3
