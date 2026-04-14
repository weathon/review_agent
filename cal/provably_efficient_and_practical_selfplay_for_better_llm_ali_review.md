=== CALIBRATION EXAMPLE 9 ===

# Final Consolidated Review
## Summary

This paper proposes Two-Agent Nash Policy Optimization (TANPO) and its single-agent approximation SADPO for self-play RLHF. The authors formulate RLHF as a two-player zero-sum game with KL-regularized objectives, derive practical DPO-style training objectives via the minimax theorem and DPO reparameterization, and prove sublinear regret under the low Two-player Generalized Eluder Coefficient (TGEC) condition. Empirically, both methods outperform Online DPO, Hybrid GSHF, and SELM on AlpacaEval 2.0, MT-Bench, and several academic benchmarks using Zephyr-7B-SFT as the base model.

---

## Strengths

- **Clean theory-to-algorithm derivation**: The algebraic path from the two-player optimistic framework (Section 3.2) through the minimax theorem (Assumption 4) and DPO reparameterization to the final practical objectives (Eqs. 14–15) is technically rigorous and elegant. This is more principled than the typical gap between theory and algorithm in self-play RLHF papers.

- **Explicit exploration mechanism with empirical support**: The min-player's exploration bonus (Eq. 13/15) addresses a concrete limitation of prior self-play methods. The paper also provides a non-trivial mechanistic insight in Section 4.2: max-player improvement comes *not* from its own explicit exploration objective (which is identical to DPO) but from being trained on data generated collaboratively with an exploratory min-player. Figure 1 directly validates this by showing increased log-prob diversity in TANPO training data vs. Online DPO, and Section 6.2 confirms that the max-player alone surpasses Online DPO solely due to richer training data.

- **Overfitting investigation across two epochs**: Figure 4 tracks both players across six iterations (two epochs on the same dataset), demonstrating monotone improvement—a meaningful empirical result that addresses a known failure mode in online RLHF where models overfit to the reward model after repeated passes over the same prompt distribution.

- **SADPO's practical elegance**: The single-agent approximation uses rejection sampling to simulate both player roles from a single policy, reducing the two-model training burden while retaining competitive performance. The selection criterion (highest vs. lowest length-normalized reference log-probability) has clear intuitive grounding in the two-agent framework.

---

## Weaknesses

### Fatal
None.

### Major

- **Min-player reported as the primary TANPO result without prominent justification**: Table 1 presents TANPO's AlpacaEval 2.0 result as 27.66% LC Win Rate (min-player), while the max-player (the agent that theoretically approximates Nash equilibrium) achieves approximately 25.05% LC Win Rate—only ~0.69% above Online DPO (24.36%). The min-player's exploration bonus explicitly encourages sampling from lower-probability regions of the action distribution, which may correlate with longer, more detailed responses that GPT-4-turbo rewards. Presenting the min-player as the headline result without upfront explanation and without reporting the max-player alongside it in Table 1 misleads readers about the true algorithmic improvement. The full results exist in Table 2 (appendix) but should be surfaced in the main text.

- **Critical self-play baselines (SPIN, SPPO) are absent**: The introduction explicitly positions this work against SPIN (Chen et al., 2024) and SPPO (Wu et al., 2024) as "self-play style RLHF" works. Yet neither appears in Table 1. The paper's central empirical claim is superiority over self-play RLHF methods; omitting the most directly comparable self-play baselines is a substantial gap that calls the empirical conclusion into question.

- **SADPO outperforms TANPO with no satisfying explanation**: SADPO (28.43% LC Win Rate) exceeds TANPO min-player (27.66%) despite being framed as a simplified heuristic approximation. The paper does not explain whether this is because (a) K=4 rejection sampling induces higher-quality diversity than the two-agent setup, (b) the changed exploration bonus in Eq. 16 is more effective, or (c) SADPO coincidentally outperforms in this single run. This inverts the expected relationship between the "full" method and its approximation, and demands a substantive explanation.

- **Structural discrepancy between TANPO and SADPO exploration bonuses is unacknowledged**: The min-player objective in TANPO (Eq. 15) uses E_{a~π^{t+1}(·|x)}[log μ(a|x)]—the cross-entropy between the *current max-player policy* and μ. SADPO's objective (Eq. 16) instead uses E_{a~π_ref(·|x)}[log π(a|x)]—cross-entropy between the *reference policy* and π. These are structurally different: the distributions over which the expectation is taken differ, and the roles of the two policies are swapped. The paper silently introduces this change without acknowledging or justifying it, which undermines the claim that SADPO is a principled approximation of TANPO.

### Minor

- **Theory-to-practice gap under Assumption 4**: Section 5 states that the regret guarantee applies to TANPO "provided the reward function class satisfies Assumption 4." Assumption 4 requires the value function to be concave-convex for the minimax interchange to hold—this is non-trivial for neural network policy classes and is relegated to Appendix C with no discussion of whether it is plausible in the experimental setting. The main text should explicitly acknowledge where the theoretical guarantees do and do not carry over.

- **Computational overhead unaddressed**: TANPO requires training two models simultaneously. The paper claims practical efficiency but provides no training time, GPU memory, or throughput comparison against single-agent baselines. SADPO requires K=4 inference passes per prompt. Practitioners cannot assess feasibility without this information.

- **No formal limitations section**: The paper omits discussion of key practical constraints: dependence on PairRM quality, applicability to larger models or different model families, and the theory-practice gap noted above.

- **Overfitting diagnostic conflates training and evaluation reward**: Figure 4 uses GPT-4-turbo (AlpacaEval) to assess overfitting while training uses PairRM. If GPT-4-turbo and PairRM share reward biases, the "no overfitting" conclusion may reflect reward-model alignment rather than true generalization. Evaluating on an independent reward model would strengthen this claim.

### Tiny

- The min-player update (Eq. 15) requires an expectation under the freshly updated max-player policy π^{t+1}, creating an intra-iteration sequential dependency. In practice this must be approximated by samples, but the approximation quality is not discussed.
- The TGEC condition (Assumption 2) is the linchpin of the theoretical guarantee but is given only a brief qualitative description in the main text. Even an informal discussion of what classes of problems exhibit low TGEC would aid readers.

---

## Nice-to-Haves

- **Ablate the min-player exploration bonus**: Train TANPO without the term E[log μ(a|x)] in Eq. 15 to disentangle whether the theoretical exploration mechanism or simply data diversity from two different models drives improvement.
- **Plot performance vs. number of online queries**: The paper claims "provably efficient" but only reports final metrics. Learning curves would provide empirical support for the sample efficiency argument.
- **Sweep K in SADPO**: K=4 is unjustified. A brief sensitivity analysis would clarify the trade-off between diversity and compute, and whether K=2 (closer to TANPO) closes the performance gap.
- **Verify the diversity–quality correlation**: Figure 1 shows log-prob diversity differences but does not demonstrate that higher-diversity pairs correspond to higher win rates. A scatter plot or correlation analysis would substantiate the diversity-drives-quality narrative.
- **Discuss TGEC plausibility for LLMs**: Even an informal argument for why LLM policy spaces might exhibit low TGEC (or where this is a theoretical idealization) would improve the paper's intellectual honesty.
- **Compute cost comparison table**: A brief table of GPU hours, peak memory, and generation cost relative to Online DPO for both TANPO and SADPO.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **No confidence intervals or error bars (Harsh Critic)**: Single-run evaluation is standard practice for LLM benchmarks at this scale. Demanding multiple-run statistics is not a community norm here and does not constitute a valid weakness.
- **Academic benchmark numbers seem implausibly large (Harsh Critic)**: The specific numbers (e.g., TANPO ~75% MMLU, ~80% GSM8k) are extracted from a radar chart image and are acknowledged by the critic as potentially unreliable artifacts. The actual values are in Table 3 (appendix). Criticizing results based on uncertain image extraction is not reliable; if the appendix numbers support a similar pattern, no criticism applies.
- **Length bias concern for AlpacaEval (Harsh Critic)**: The paper already uses both LC Win Rate and Win Rate specifically to control for length bias. This concern is adequately addressed.
- **Calling active exploration for the max-player an "overstatement" (Harsh Critic)**: The paper itself explicitly states in Section 4.2 that "the max-player's objective in TANPO remains identical to DPO objective" and motivates TANPO through data diversity, not direct exploration of the max-player. The paper is transparent about this mechanism; the criticism misreads the paper's own framing.

---

## Novel Insights

The reviews collectively surface an important conceptual distinction that the paper demonstrates but does not fully articulate: TANPO separates "exploration through adversarial data generation" (which benefits the max-player indirectly) from "exploration through objective modification" (which benefits the min-player directly). The max-player never has an explicit exploration term, yet it outperforms Online DPO solely because its training data is more diverse—generated by a competing exploratory agent. This cleanly demonstrates that for policy optimization in LLMs, the *distribution of training comparisons* may matter more than *the optimization objective itself*, and that diversity-inducing data collection mechanisms (even without reformulated losses) can be a powerful lever for alignment. The structural discrepancy between SADPO's and TANPO's exploration bonuses, combined with SADPO outperforming TANPO, further suggests the two objectives are not equivalent in practice and that the reference-policy-anchored cross-entropy in SADPO may be independently valuable—this warrants deeper analysis.

---

## Suggestions

1. **Promote max-player results to Table 1** alongside min-player results, with explicit labeling and a brief in-text explanation of why min-player performance is also reported. This prevents misinterpretation of TANPO's algorithmic gain over Online DPO.

2. **Add SPIN and SPPO to Table 1**, or explain in the main text why they are excluded (e.g., architectural incompatibility, unavailability of checkpoints). The self-play comparison is the paper's primary empirical positioning.

3. **Explain or reconcile the SADPO > TANPO result**: Run ablations that isolate (a) two-agent data diversity alone (using DPO objective on both players), (b) the min-player bonus alone without two-agent setup, and (c) the SADPO rejection sampling strategy without the modified loss, to identify which component drives SADPO's edge.

4. **Explicitly address the structural difference between Eq. 15 and Eq. 16** in the main text. Either justify it as a deliberate design choice (and provide intuition) or acknowledge it as an approximation error and discuss its consequences.

5. **Include a compute cost table** (GPU hours, memory footprint, per-iteration sampling cost) for TANPO, SADPO, and Online DPO to substantiate the "practical" claim.

6. **Add an independent reward model evaluation for the overfitting experiment**: Use a model distinct from PairRM (e.g., GPT-4 direct scoring) to validate that Figure 4's monotone improvement reflects true generalization rather than PairRM-specific reward hacking.

---

**Axis evaluations:**
- **Novelty**: Moderate-to-high. The combination of two-player Nash formulation with practical DPO-style derivation and an explicit exploration bonus is a meaningful contribution over prior work. The SADPO approximation is novel and practically useful.
- **Technical soundness**: Moderate. The theoretical framework is sound within its assumptions, but the theory-to-practice gap (Assumption 4 unverified), the unacknowledged discrepancy between Eqs. 15 and 16, and the unexplained SADPO > TANPO result introduce non-trivial concerns.
- **Empirical support**: Weak-to-moderate. Results are promising but undermined by absent self-play baselines, the choice of min-player as the primary result, and unexplained benchmark anomalies. The diversity analysis (Figure 1) is a genuine empirical contribution.
- **Significance**: Moderate. The framework addresses a real gap (theory-practice bridge, active exploration in self-play RLHF) and SADPO is practically deployable. Impact is limited by the scale of experiments (7B only) and unresolved questions above.
- **Clarity**: Good in the derivation sections; weaker in the experimental section where the choice of min-player as headline result and the silent change in exploration bonus design undercut transparency.

# Actual Human Scores
Individual reviewer scores: [5.0, 3.0, 6.0, 3.0]
Average score: 4.2
Binary outcome: Reject
