Now I have enough calibration. Let me write the final review.

## Summary

The paper proposes WeCAN, an end-to-end reinforcement learning framework for heterogeneous DAG scheduling with task-resource compatibility coefficients. Its key contributions are: (1) a weighted cross-attention (WeCA) layer that places compatibility coefficients K_acc as diagonal scaling outside the softmax, enabling variable-size heterogeneous environment embeddings; (2) a theoretical analysis showing list scheduling has an inherent optimality gap that is most pronounced for heavy tasks, along with a skip-action mechanism that provably closes this gap within single-pass inference; and (3) empirical validation on TPC-H and Computation Graphs benchmarks showing up to 18% improvement over heuristics and 7.7–9.5% over neural baselines at near-heuristic inference speed.

## Strengths

- **The WeCA mechanism is a principled and well-motivated design choice.** The outside-softmax placement of K_acc (Eq. 4) directly addresses a real limitation: when compatibility coefficients are placed inside the softmax (log form), normalization collapses the embedding distinction between tasks with identical attributes but different compatibility profiles (the v₁/v₂ example in Section 3.1). The ablation in Table 3 confirms this: WeCA achieves 14.0% vs. 10.5% for WeCA-inside on TPC-H-30. The architecture also naturally handles variable pool counts and task types, which is validated in Figure 2's generalization experiments.

- **The skip-action design provides both theoretical grounding and practical benefit.** Theorem 1 (parts ii–iii) formally establishes that without skip, the generation map fails to assign positive probability to optimal solutions for some problems, and that the skip mechanism restores this property. Figure 3 empirically validates that this gap manifests for heavy-task scenarios, where WeCAN with skip outperforms all baselines including the non-list-scheduling HEFT. The single-pass skip score formula u_skip = u_a(1−k/(2n))^{u_b} + u_c avoids multi-round computation while preventing excessive idling.

- **Strong empirical performance with dramatic efficiency gains.** WeCAN-Greedy achieves makespan 62,587 in 1.72s on TPC-H-100, compared to PPO-BiHyb's 67,695 in 179.19s (7.7% improvement, ~104× speedup) and One-Shot-S(256)'s 66,173 in 9.85s (5.4% improvement). On Computation Graphs, improvements over the best neural baseline reach 9.5% (Table 2). The greedy mode runs at heuristic-scale speeds (0.15–1.72s), making it practically deployable.

- **Comprehensive generalization experiments.** Figure 2 tests robustness to changes in pool number, pool type, task number, and task type from a single training configuration, showing that WeCAN's architectural design enables zero-shot adaptation to unseen environment configurations.

## Weaknesses

### Fatal
None.

### Major

- **Unclear adaptation of neural baselines for the heterogeneous setting.** The paper explicitly notes (Section 2.1) that One-Shot "does not consider compatibility coefficients or pool allocation." Yet One-Shot and PPO-BiHyb serve as the primary neural baselines in Tables 1–2, and the paper describes applying "three pool-selection rules" only for the four heuristic baselines, remaining silent on how the neural baselines handle pool assignment and compatibility. Without this information, it is difficult to assess whether the 7.7–9.5% improvements over the best neural baseline reflect genuine architectural advances or simply a stronger adaptation to the heterogeneous setting. This concern matters because the improvements over heuristic baselines (which are fully adapted) are even larger, suggesting baseline adaptation matters significantly for comparisons.

- **The skip score formula u_skip = u_a(1−k/(2n))^{u_b} + u_c lacks formal or empirical justification for this specific parametric family.** Theorem 1 part (iv) guarantees that *some* set of scores exists to greedily recover an optimal solution, but this is an existence result — it does not justify why this particular three-parameter polynomial-decay form is the right choice, nor does the paper provide an ablation against simpler alternatives (e.g., a single learned scalar, or a linear decay). The claim that this form "clusters most poor solutions in the high-u_a, high-u_c region" (Section 4.2) is asserted without formal analysis or empirical validation of the loss landscape. Since the skip action is one of the two core technical contributions, this gap matters.

### Minor

- **The non-autoregressive vs. autoregressive comparison is deferred to Appendix B.** The decoder uses a non-autoregressive design where action probabilities depend only on s₁ (Section 3.2), discarding conditioning on previously selected actions. Since the paper claims improvements over an autoregressive baseline (PPO-BiHyb), having this comparison in the main text would strengthen the architectural argument. However, this is a presentation issue rather than a fundamental concern since the empirical results already include the comparison.

- **The heavy-task experiment (Figure 3) only compares WeCAN variants against heuristics, not against neural baselines.** Showing that One-Shot and PPO-BiHyb also struggle (or don't) with heavy tasks would more directly validate the skip mechanism's unique contribution.

### Trivial
None.

## Nice-to-Haves

- A worked numerical example (4–5 task DAG) illustrating how list scheduling fails and skip succeeds would make the theoretical contribution more accessible.
- Confidence intervals or statistical significance tests for comparisons against deterministic baselines would strengthen the empirical claims.
- Analysis of WeCA's computational overhead scaling with the number of pools/task types (n_c) would help assess scalability.

## Removed Points

- **"Theorems 1 and 2 proofs are inaccessible in Appendix A"** — The appendix is part of the submission. The parser strips appendices, but they exist in the original. The paper states "we provide the details of the proof in Appendix A," which is standard practice. Removed: this is a parser artifact, not an author error.

- **"F(t,v) notation inconsistency"** — This is a minor notation issue that appears to be a formatting artifact. The meaning (tasks running on pool c at time t) is clear from context. Removed as a formatting nitpick.

- **"The abstract overclaims novelty for observing list scheduling suboptimality"** — The paper's specific contribution is the analysis of S_list's non-surjectivity and the reduced-space mapping T∘S framework, not merely observing that list scheduling is suboptimal. The wording in the abstract ("revealing their inability to guarantee optimal solutions") is the paper's formal result (Theorem 1, parts ii–iii), not a generic observation. Removed as a mischaracterization.

- **"The outside-softmax argument is a design heuristic rather than principled"** — The paper provides both a concrete counterexample (v₁/v₂ tasks) and empirical validation via ablation (Table 3: 14.0% vs 10.5%). This is a well-justified design choice, not merely heuristic. Removed as understating the evidence provided.

- **"Request for statistical significance / confidence intervals"** — Standard practice in this community for RL-based scheduling methods is to report mean and std over seeds, which the paper does. Demanding significance tests against deterministic baselines is beyond community norms. Moved to Nice-to-Have.

- **"Request for computational overhead analysis of WeCA"** — This is a reasonable suggestion for future work but not a weakness that affects the current claims, since inference times are already reported. Moved to Nice-to-Have.

- **"Claim that WeCA's generalization is a strength"** — This is genuinely supported by Figure 2 experiments and is kept.

## Novel Insights

The paper identifies a clean theoretical characterization of when list scheduling provably fails: T∘S_list is neither identity nor surjective, meaning the generation map shrinks its image and can exclude optimal solutions. The skip action transforms the reduced space to make the composition surjective, which is a non-obvious insight. The outside-softmax placement of K_acc is also more than a design choice — it preserves the norm/magnitude of compatibility information, which inside-softmax normalization would destroy. However, the specific skip score formula remains an engineering choice without clear theoretical necessity beyond the existence guarantee of Theorem 1(iv).

## Suggestions

- Clarify in the experimental section (or an appendix) exactly how One-Shot and PPO-BiHyb handle compatibility coefficients and pool assignment in the heterogeneous experiments, so that the 7–10% improvement over neural baselines can be properly interpreted.
- Ablate the skip score formula against simpler alternatives (e.g., constant u_skip, linear decay) to demonstrate that the polynomial-decay form is beneficial rather than merely sufficient.

## Evaluation

**Originality**: High. The WeCA layer and skip-action formalization for single-pass heterogeneous scheduling are novel contributions not seen in prior work.

**Importance**: High. Heterogeneous DAG scheduling with compatibility coefficients is a practically important problem; the near-heuristic-speed results are compelling for time-sensitive applications.

**Claims support**: Moderate. The WeCA claims are well-supported by ablation and generalization experiments. The skip-action theoretical claims are partly verifiable (Theorem statements are clear) but the specific formula lacks justification. The neural baseline comparison would be stronger with adaptation details.

**Experiments**: Good. Strong results across two datasets, ablation study, and generalization experiments. The main gap is the lack of clarity around baseline adaptation.

**Clarity**: Adequate. The paper covers significant ground (architecture, theory, experiments) and is generally readable, though Sections 3–4 could benefit from a concrete example to ground the abstract formalism.

## Score and Decision

**Calibration comparison:**

- High anchors: jsWCmrsHHs (avg 7.5, DRL for JSSP with novel GNN mechanism, strong empirical results) — this paper has comparable novelty and empirical quality but weaker theoretical grounding and the baseline adaptation concern.
- Medium anchors: z2z9suDRjw (avg 6.25, GOAL generalist CO agent with mixed-attention) — this paper has more focused and deeper contributions than GOAL.
- Low anchors: riQmzq5FaQ (avg 3.75, RL with elastic time steps, weak theory/experiments); 8WtBrv2k2b (avg 5.0, quantum scheduling RL with overclaimed 3× improvement but weak theoretical justification).

This paper is substantially stronger than the low-scoring papers — it has genuine technical novelty (WeCA + skip), provable theoretical results (Theorems 1–2), and strong empirical results. It is somewhat below the high-scoring papers in terms of completeness (baseline adaptation concern, skip formula justification), but the core contributions are solid. The score should be in the 6–7 range, leaning toward 6.5 given the two major weaknesses but strong overall contribution.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>