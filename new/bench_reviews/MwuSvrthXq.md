Now I have all the information I need. Let me synthesize the final review.

## Summary

The paper proposes WeCAN, an end-to-end reinforcement learning framework for heterogeneous DAG scheduling that uses weighted cross-attention (WeCA) layers to incorporate task-resource compatibility coefficients adaptively while maintaining flexibility across varying environment sizes. It also introduces a skip action mechanism for single-pass inference that theoretically closes the optimality gap inherent in list-scheduling-based generation maps, supported by formal analysis of solution spaces and generation maps. Empirical results on TPC-H and Computation Graph benchmarks demonstrate makespan improvements of 7.7–9.5% over the best neural baselines and 13.4–18.1% over heuristics, with near-heuristic inference speed.

## Strengths

- **The WeCA layer is a clean, well-motivated architectural contribution.** Placing compatibility coefficients as diagonal scaling outside the softmax (Eq. in Section 3.1) naturally masks incompatible assignments and preserves per-task compatibility profiles. The ablation in Table 3 convincingly demonstrates its importance: WeCA+LDDGNN achieves 14.0% improvement over Tetris on TPC-H-30, while removing WeCA (WeCA-final-only) collapses to 0.5%, and the "inside" variant drops to 10.5%. The motivating example of two identical-attribute tasks with different compatibility profiles (Section 3.1) is clear and intuitive.

- **Single-pass inference with heuristic-level running times is a practically important result.** WeCAN-Greedy achieves 0.15s on TPC-H-30 (Table 1), comparable to heuristic runtimes, while substantially outperforming them on makespan. This is significantly faster than PPO-BiHyb (20.48s) and competitive with One-Shot (2.26s).

- **Generalization experiments (Figure 2) are a strong validation.** Testing a single trained model under varying pool counts (2–5), pool feature shifts, task counts (100–400), and task types demonstrates the architecture's adaptability—directly validating WeCA's claimed advantage over fixed-size embeddings.

- **Theoretical analysis of the optimality gap is sound.** Theorem 1 establishes that skip is necessary and sufficient for positive probability on optimal solutions under the proposed generation map, and Theorem 2 provides the surjection criterion (Assumption 1). The insight that T∘S_list is not surjective explains a real limitation of list scheduling.

- **Consistent improvements across multiple graph types and datasets.** Gains hold on ER graphs, layer graphs, stochastic block models, and real-world TPC-H instances (Tables 1 and 2), suggesting the method generalizes beyond specific graph structures.

## Weaknesses

### Fatal
None.

### Major

- **Skip action lacks ablation on standard benchmarks.** The skip action is listed as a co-equal contribution (#3) and is the primary subject of Section 4's theoretical analysis. Yet its empirical evaluation is confined to Figure 3, which tests it only on modified TPC-H datasets with 1% heavy-task replacement. The main results in Tables 1 and 2 include skip by default with no "WeCAN without skip" row, making it impossible to assess whether skip contributes to the headline numbers (7.7% over neural baselines, 18.1% over heuristics) or only helps on the artificially constructed heavy-task cases. The paper states "Appendix C further validates the effectiveness of this design," but the main text fails to establish skip's contribution on the primary evaluation benchmarks. This matters because if skip provides negligible benefit on standard problems, contribution #3 is overstated and the theoretical analysis, while correct, has limited practical relevance for the paper's main claims.

- **Unclear adaptation of neural baselines for heterogeneous settings.** The paper acknowledges (Section 2.1) that One-Shot's architecture "does not consider compatibility coefficients or pool allocation," and PPO-BiHyb was designed for bi-level optimization on homogeneous graphs. Yet the paper does not specify how these methods were adapted for the heterogeneous evaluation setting. Were they retrained on the same heterogeneous data? Were any architectural modifications made to enable pool selection? If One-Shot was simply run with its original architecture that cannot natively handle pool assignment, the comparison significantly favors WeCAN by structural design rather than learned policy quality. The 7.7% improvement claim over One-Shot requires clarification of what One-Shot was actually given and how it was modified.

### Minor

- **Disconnect between theoretical skip guarantees and the practical skip formula.** Theorem 1(iv) establishes existence of scores enabling greedy optimality, but the actual formula u_skip = u_a(1 − k/2n)^u_b + u_c is a hand-crafted heuristic with no provable connection to the theorem. The claim that "our design clusters most poor solutions in the high-u_a, high-u_c region" (Section 4.2) is asserted without empirical or analytical evidence of the score landscape or training dynamics. The theoretical and practical contributions are logically separate; the paper could be clearer about this distinction.

- **Non-autoregressive decoder choice is insufficiently justified in the main text.** The paper employs a non-autoregressive decoder where "the action probability p_θ(π_t|s_t, π_{<t}) depends only on the initial state s_1" (Section 3.2), which is a significant architectural restriction for a scheduling problem where decisions are highly interdependent. The comparison with an autoregressive decoder is deferred to Appendix B. The cost of this design choice should be quantified in the main text, as it directly affects the quality of generated schedules.

### Trivial
None.

## Nice-to-Haves

- Reporting the frequency of skip actions during inference on standard benchmarks would clarify whether skip is actively used outside heavy-task scenarios.
- A sensitivity analysis of the skip formula's parametrization (u_a, u_b, u_c) would strengthen confidence in the design.
- Qualitative scheduling visualizations (e.g., Gantt charts comparing with/without skip on heavy-task instances) would make the optimality gap concrete.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh Critic's claim that the One-Shot comparison is "structurally invalid"**: Overstated. One-Shot was likely retrained on the same data (it produces reasonable results better than heuristics), so the comparison shows that WeCAN's architecture is better suited for heterogeneous scheduling—which IS the contribution. The real issue is lack of clarity about how One-Shot was adapted, not that the comparison is invalid.
- **Cherry-picking in abstract**: The "up to 18.1%" framing is standard practice in ML papers. The paper also reports the lower bound (9.1% on ER graphs). Trivial observation.
- **Reproducibility concerns about random dataset modifications**: Nitpick about reproducibility. The paper likely uses fixed seeds, and dataset generation randomness is standard practice.
- **Continuous K_acc values vs binary compatibility**: The paper uses K_acc values continuously in the attention mechanism (as diagonal weighting) and only uses the binary mask K_acc > 0 for feasibility. This is actually a feature, not a weakness.
- **Overclaimed "novel criterion" framing**: The harsh critic says the optimality gap of list scheduling is well-known. While true, the specific formalization via surjectivity of T∘S and the connection to skip action in the single-pass setting is a genuine contribution, even if the general idea is known.
- **Strength Finder's "principled skip-score formula" as a presentation strength**: This conflicts with the verified weakness that the formula lacks provable connection to theory. Moved to removed.

## Novel Insights

The paper reveals an interesting structural asymmetry in neural scheduling baselines: methods designed for homogeneous settings (One-Shot, PPO-BiHyb) are commonly compared in heterogeneous settings without adaptation, which creates an implicit architectural advantage for heterogeneous-specific designs like WeCAN. This pattern is likely widespread in the neural scheduling literature and merits broader discussion about fair evaluation protocols when extending methods beyond their original problem scope.

## Suggestions

- Add a "WeCAN without skip" row to Tables 1 and 2 to isolate the skip action's contribution on standard benchmarks—this single addition would significantly strengthen the empirical case for the skip contribution.
- Clarify in Section 5.1 exactly how One-Shot and PPO-BiHyb were adapted for the heterogeneous setting (e.g., "One-Shot was retrained with pool selection added to its action space; compatibility coefficients were provided as node features").
- Move at least a summary of the autoregressive vs. non-autoregressive decoder comparison from Appendix B to the main text, including the makespan gap.

## Score and Decision

**Calibration anchors:**
- High: Tke3BVwUz6 (MARL with DAG structure, avg 7.0) — stronger theory-practice integration; uJCGMBO6Qx (heterogeneity in cooperative MARL, avg 7.0) — more rigorous theoretical framework; h06l9w1clt (LPD, avg 7.0) — cleaner contribution with stronger empirical validation.
- Medium: UbWy2QVmke (GAA-PtrNet for DAG scheduling, avg 4.5) — this paper is clearly stronger, with better architecture, theory, and experiments; IdsRzAAkB2 (offline RL for JSP, avg 5.5) — similar missing-ablation issues but this paper has a cleaner main contribution; RAISE (7gA6ZYii0e, avg 4.0) — this paper is significantly stronger.
- Low: CVZFzsg1PJ (RL for MILP, avg 2.0) — fundamentally flawed methodology; bisWxwcK8D (RL for dynamic VRP, avg 2.5) — weak baselines and overclaimed results. This paper has no such fundamental issues.

This paper sits above the medium-band rejected papers (GAA-PtrNet at 4.5, RAISE at 4.0) due to its cleaner WeCA contribution, stronger theory, and more comprehensive experiments. It sits below the high-band accepted papers (7.0) due to the skip action's thin empirical validation on standard benchmarks and the baseline adaptation clarity gap. The WeCA layer alone would be a solid contribution; the skip action contribution is promising but under-validated for the headline claims. I place this at 6.0—above borderline but not comfortably so, primarily because the skip contribution (one of four listed) lacks the empirical grounding needed to support its co-equal status.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>