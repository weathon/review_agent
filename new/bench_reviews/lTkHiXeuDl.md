## Summary
This paper proposes HMoRA, a parameter-efficient fine-tuning method combining hierarchical mixture of LoRA experts with layer-dependent routing (token-level in shallow layers, task-level in deeper layers) and a novel Constrained GJS auxiliary loss for routing certainty and balance. The method claims to outperform full fine-tuning on 7 NLP benchmarks while training only 3.9% of parameters.

## Strengths
- **Novel Constrained GJS auxiliary loss formulation**: The CGJS divergence (Equations 11-12) is a well-motivated mathematical contribution that explicitly balances routing certainty (low individual entropy) against load balancing (high mean entropy). Table 1 demonstrates concrete improvements: adding L_aux increases average accuracy for both Soft (62.83→63.65) and Top-k (62.87→63.72) routing, with Figure 3 visually confirming the loss maintains balance while achieving lower entropy.
- **Layer-dependent hierarchical routing mechanism**: The mixing coefficient α^(l) (Equation 8) shifts routing granularity across layers, addressing a genuine limitation in prior uniform multi-granular routing approaches. Table 2 shows HMoRA w/o LW (64.16 avg) outperforms static routing baselines like MoLoRA (63.02) and HydraLoRA (62.70).
- **Unsupervised task differentiation capability**: Section 4.3 and Figure 4 demonstrate the task router differentiates 73.68% of unseen MMLU sub-tasks without explicit labels (vs. 0% without the auxiliary loss), supporting the claim about generalization to unseen tasks.

## Weaknesses

### Fatal
None

### Major
- **Full Fine-Tuning baseline constrained by step limit**: Section 4 states all methods including Full FT are limited to 10,000 training steps with early stopping. Full FT typically requires significantly more steps to converge than PEFT methods, especially on massive multi-task datasets like Flan v2. This experimental design demonstrates HMoRA converges *faster* than Full FT, not necessarily that it achieves a higher performance *ceiling*. The headline claim "outperforms full fine-tuning" (Abstract, Section 1, Table 2) is therefore overstated—this shows parameter-efficient methods can match or exceed Full FT at equal step budgets, which is a different claim. This affects the core contribution validation.
- **Core hierarchical mechanism lacks main-text ablation**: The central novelty is the layer-dependent α^(l) scheduling (Equation 8), yet Table 2 only compares the full HMoRA package against other methods. The validation that increasing α^(l) with layer depth improves performance is relegated to Appendix E.5. Without a main-text ablation comparing hierarchical vs. uniform routing (e.g., α^(l) = constant), it is unclear whether gains come from the *hierarchical* design or simply from having both routers available. This makes the main results insufficient to isolate the primary architectural contribution.

### Minor
- **Statistical significance not reported for marginal gains**: Section 4 states experiments were "repeated 5 times," but Tables 1-3 report only means without standard deviations. Given gains over Full FT are ~1% and auxiliary loss contributions are often <0.5%, variance metrics would help determine if improvements are statistically significant or within random seed noise—a known concern in PEFT research.
- **Limited model scale validation**: Experiments use Qwen2 1.5B (with LLaMA 3.2 1B in Appendix E.7). MoE routing dynamics (expert collapse, routing entropy) often scale non-linearly, and 1.5B is small relative to "foundation or frontier models" (Primary Area). No 7B+ results limit confidence in generalizability to larger LLMs.

### Trivial
- **TaskEncoder architecture underspecified in main text**: Section 3.2 states TaskEncoder "can be a single or multi-layer Transformer encoder" without specifying depth used for main results. Critical component architecture should be in main text for reproducibility.

## Nice-to-Haves
- **Routing distribution visualization**: A heatmap showing actual gate values across layers for sample inputs would visually confirm the hierarchical claim (shallow layers attending to different experts than deep layers for the same token).
- **Compute-equivalent comparison**: Comparing HMoRA against Full FT matched on wall-clock time or total FLOPs (rather than step count) would provide a more practical efficiency perspective, since Full FT steps are more expensive.
- **Stricter zero-shot generalization evaluation**: Training on a subset of Flan tasks and testing on held-out task families (e.g., train on QA, test on Translation) would more rigorously validate the "unseen task" generalization claim.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Introduction critique about sentence-level routing**: The harsh critic claims HMoRA's task representation is "functionally equivalent to sentence-level routing" which the paper criticizes. However, the paper distinguishes HMoRA's approach by the auxiliary loss clustering effect (Section 3.3, Appendix D) and the hierarchical combination with token routing—this is addressed, not a weakness.
- **Task router contributes marginally claim**: Section 4.3 notes the harsh critic claims the task router ablation (Table 3) shows only ~1% drop. However, Table 3 shows HMoRA (64.16) vs. w/o L_aux for Task Router (63.18)—a 0.98% drop, which is meaningful given the marginal gain context. This criticism overstates the issue.
- **Any criticism about appendix existence**: The parser strips appendices from all papers; Appendix E.5-E.8 exist in the original submission. Criticizing their absence is invalid.
- **Reproducibility nitpicks about hyperparameters**: Specific hyperparameter values (ε=4, μ=-2, etc.) ARE provided in Section 4.2. Claims they are undisclosed are incorrect.

## Novel Insights
The Constrained GJS loss formulation represents a genuine conceptual advance over standard load balancing losses: by capping both the balance term (γ_b log e) and certainty term (γ_c log e), it preserves model flexibility while preventing pathological routing states. This differs from prior work (e.g., ERC loss in MpeyjgWbKt) which focuses on router-expert coupling rather than the certainty-balance trade-off directly. The hierarchical routing design, while not entirely novel (similar ideas appear in CwQzoZ1WxH), is distinguished by the smooth sigmoid-based α^(l) scheduling rather than discrete two-stage routing.

## Suggestions
1. **Reframe the Full FT comparison claim**: Either (a) add a converged Full FT baseline (more steps) to verify the performance ceiling claim, or (b) rephrase the contribution as "matches or exceeds Full FT at equal step budgets with 25x fewer trainable parameters"—the latter is still valuable and defensible.
2. **Move hierarchy ablation to main text**: Add one row to Table 2 for "HMoRA-Uniform" (constant α^(l)) to isolate the hierarchical contribution from simply having dual routers.
3. **Add standard deviation to result tables**: Given 5 runs were performed, reporting ±std would strengthen confidence in marginal gains.
4. **Specify TaskEncoder depth in Section 3.2**: One sentence stating the architecture used (e.g., "2-layer Transformer encoder") would resolve reproducibility concerns.

## Score and Decision

**Calibration anchors consulted:**
- **High-scoring (≥6)**: MoBE (6.67, /home/wg25r/review_agent/human_reviews_2026/8RV6H50OSf.md) — strong compression results with clear claims; MpeyjgWbKt (6.67) — auxiliary loss for MoE routing with extensive pre-training validation; 86P3sb1dpr (6.00) — LoFT claims narrowing gap with Full FT with optimizer state alignment theory.
- **Medium-scoring (5-6)**: 4ST2YyTjI7 (5.50) — LD-MoLE with learnable dynamic routing, similar small-model limitation; zBgjWTWgCh (5.50) — MoE PEFT claiming parity with Full FT, rejected for limited scope; FNuvMnGAm8 (5.00) — novel load balancing loss for MoE routing.
- **Low-scoring (≤4)**: OXmRvlihi3 (3.50) — unfair baseline comparison flagged as critical weakness; rxUXAYhl2T (4.00) — missing critical ablation (heterogeneous vs. homogeneous MoE); CwQzoZ1WxH (4.00) — hierarchical routing paper rejected for limited experimental validation and unsubstantiated claims.

**Positioning**: HMoRA has stronger empirical breadth (7 benchmarks) than 4ST2YyTjI7 and FNuvMnGAm8, with a more novel auxiliary loss formulation than LD-MoLE. However, the unfair Full FT baseline concern mirrors OXmRvlihi3's weakness (3.50), and the hierarchy ablation-in-appendix issue resembles rxUXAYhl2T's missing ablation problem (4.00). The paper is better than these low anchors due to comprehensive benchmarking and the CGJS loss contribution, but the baseline issue prevents reaching 6+ territory where claims are fully validated. Positioned between the medium anchors (5.0-5.5) and high anchors (6.0+), the paper's solid contributions are tempered by experimental design concerns that would need addressing for full claim validation.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>