# ICLR Benchmark Results

Date: 2026-04-12 23:06
Critic/Merger: claude:claude-sonnet-4-6 (OpenRouter)
Neutral: qwen/qwen3.5-plus-02-15, Related Work: qwen/qwen3.5-flash-02-23:online (OpenRouter)

## 1c73HCZpbo

- GT: Reject (avg 4.0)
- Predicted: Reject (3.6/10)
- Match: YES

### Final Review

## Summary

REVEAL-IT proposes a framework for explaining RL agent learning by visualizing policy weight updates as node-link graphs and training a GNN-based explainer to identify critical updates. The framework uses these explanations to dynamically optimize training task sequences (curriculum learning), aiming to improve both interpretability and learning efficiency. Experiments are conducted on ALFWorld (embodied tasks) and OpenAI Gym environments.

## Strengths

- **Novel integration of interpretability and curriculum optimization:** The paper bridges explainable AI and curriculum RL by using policy structure insights to actively modify training sequences (Algorithm 1, lines 4–8). This is a creative approach that treats interpretability as actionable rather than purely post-hoc.

- **Visual insight into policy dynamics:** Figure 2 provides intuitive visualization of which policy weights change during training for each sub-task. The identification of shared components across tasks (orange squares) offers genuine insight into skill transfer—e.g., "open microwave 1" and "take apple 1 from microwave 1" show overlapping policy regions requiring spatial understanding of the microwave.

- **Algorithm-agnostic design:** Table 2 demonstrates compatibility with PPO, A2C, and PG algorithms across six environments, suggesting the framework's modular applicability to different online RL methods.

- **Substantial performance improvements on ALFWorld:** Table 1 shows REVEAL-IT achieving 0.80 average success rate versus InstructBLIP (0.22) and PPO without curriculum (0.04). While baseline fairness has caveats (see weaknesses), the improvement over the PPO baseline directly demonstrates curriculum optimization benefits.

## Weaknesses

- **Missing curriculum learning baselines:** The paper compares REVEAL-IT against VLM agents and a plain PPO baseline, but does not compare against standard curriculum RL methods (e.g., PLR, self-paced learning, or even a random-ordering curriculum). Table 3 compares GNN explainer variants but does not isolate whether performance gains come from the specific GNN-based optimization versus any structured curriculum. This makes it difficult to assess the incremental contribution of the proposed mechanism.

- **Underspecified algorithm details:** Algorithm 1, line 7 states "Sample training task sequence Seq_t in terms of {P(task_n, π_t)}" without specifying the selection criterion (greedy? proportional? sampled from a distribution?). Similarly, Section 4.2 states that "activated nodes in the policy will be tagged and utilized as the ground truth" without defining "activated" (threshold? non-zero ReLU output?). These gaps affect reproducibility.

- **Mixed OpenAI Gym results without statistical testing:** In Table 2, REVEAL-IT improves performance in 11 of 18 algorithm-environment pairs but regresses in 7 cases (e.g., Hopper with PPO, Reacher with A2C, InvertedPendulum with PG). The parenthetical values indicate REVEAL-IT uses 10–20% fewer environment steps, so performance per-step may improve even when final performance regresses—but no learning curves or multiple-seed results are provided. Standard deviations or confidence intervals are absent, which is concerning given RL's inherent variance.

- **Interpretability claims not validated:** The paper frames itself as an interpretability framework, yet evaluates only downstream task performance. No standard interpretability metrics (faithfulness, stability, sparsity) are reported, and no human evaluation tests whether users can actually use the visualizations to understand agent behavior or debug policies.

## Nice-to-Haves

- **Scalability discussion for larger architectures:** The experiments use a 4-layer × 64-node MLP. Discussion of how the node-link graph construction would scale to CNNs, Transformers, or larger networks would strengthen claims about applicability to "complex environments."

- **Computational overhead analysis:** Training a GNN predictor and explainer alongside the RL agent adds cost. Reporting time/memory overhead would help practitioners assess practical feasibility.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Claim that comparison in Table 1 is completely unfair:** The reviewer criticized comparing REVEAL-IT (a trained RL agent with curriculum) against VLM baselines. However, the paper does include a directly comparable PPO baseline (0.04 success rate), and the VLM baselines are state-of-the-art on ALFWorld, providing context. The comparison has limitations but is not wholly invalid.

- **Claim that Table 2 uses "fewer steps" as a flaw:** The parenthetical values show REVEAL-IT achieves comparable or better results with fewer environment steps, which is evidence of sample efficiency—a stated goal. This is a feature, not a bug, though equal-step comparisons would strengthen the final-performance claim.

- **Demand for user study on interpretability:** User studies are valuable but not standard for algorithmic interpretability papers at ICLR. Standard metrics (faithfulness, stability) would be more appropriate nice-to-haves.

- **Claim that "POMDP framing is scientifically empty":** The mention is brief and not central to the method. While unnecessary, it does not harm the paper.

- **Criticism that the abstract "conflates explanation and optimization":** The paper explicitly positions interpretability as a means to enable curriculum optimization—this is a coherent framing, not conflation.

- **Scalability concern about "8 nodes" being too coarse:** The paper states "we opt to depict the 8 interconnected nodes with the most significant weight adjustment" as a visualization choice for human interpretability. This is a presentation decision, not a fundamental limitation.

## Novel Insights

The visualization approach reveals that different sub-tasks activate overlapping vs. distinct policy regions, providing empirical grounding for curriculum design intuition: tasks requiring shared skills (e.g., "find object" and "pick object") should precede tasks requiring novel skills. This structural perspective on transfer—visualized directly in policy weight space—offers a novel lens on why certain curriculum orderings work better than others. The finding that the GNN predictor shifts task distribution toward "look," "pick," and "find" early in training (Figure 3) matches theoretical expectations about skill prerequisites, suggesting the explainer captures meaningful structure in the learning process.

## Suggestions

- Add a baseline comparison against a simple random-ordering curriculum to isolate the contribution of the GNN-based optimization logic. Report success rates for: (a) random task order, (b) fixed human-designed order, (c) REVEAL-IT's predicted order.

- Define the task selection criterion precisely in Algorithm 1 and specify what constitutes an "activated" node for GNN training ground truth.

- Report learning curves and results over at least 3–5 random seeds with standard deviations to address RL variance concerns.

---

## 33P4evE2ej

- GT: Reject (avg 4.8)
- Predicted: Reject (4.3/10)
- Match: YES

### Final Review

## Summary
DynaMer proposes a Gated Mixture-of-Experts adapter architecture that combines frozen Vision Transformer backbones from general-domain (DINO v2) and medical-domain (cell image pretrained) sources for medical image task adaptation. The method introduces token-level expert routing with a gating mechanism for stability and a layer-wise skipping router for inference efficiency. Experiments on the Med-VTAB benchmark spanning 23 medical imaging datasets demonstrate consistent improvements over prior adapter methods.

## Strengths
- **Comprehensive empirical coverage across medical modalities**: The paper evaluates on color images (9 datasets), X-rays (7 datasets), and OCT/CT/MRI (7 datasets), plus patient OOD tests and general-domain transfer (Tables 1–10). This breadth exceeds typical single-modality medical imaging papers and provides evidence of generalization.

- **Efficiency mechanism with demonstrated practical value**: The layer-wise skipping router reduces inference time from 0.165s to 0.086s at 50% tokens while maintaining accuracy (Table 7). This directly addresses clinical deployment constraints where latency matters.

- **Gating mechanism provides stability benefits**: Table 4 shows consistent improvements when both General and Medical gates are enabled (70.82 vs 70.38 without gates), validating the design motivation that randomly initialized adapters cause training instability.

- **Ablation studies validate architectural choices**: Tables 5 and 6 systematically vary gating dimensions (768→384→192→1) and gating layers (12→6→3→1), showing performance scales with gating capacity and depth.

## Weaknesses
- **Computational efficiency claims are misleading about total system cost**: The paper emphasizes parameter efficiency (1.21X in Tables 1–3, Figure 1) but does not account for the fundamental fact that running two frozen ViT-B backbones approximately doubles inference FLOPs regardless of adapter size. Figure 1 plots "Tunable Params (%)" against performance, creating an apples-to-oranges comparison since baseline methods use one backbone while DynaMer uses two. The paper should report total FLOPs or wall-clock memory to enable fair efficiency comparisons. This matters because the claimed "efficiency" may not materialize in actual deployment.

- **Table 7 shows 50% token skipping outperforms 100% tokens without explanation**: Across all nine color-image datasets, passing only 50% of tokens through the MoE adapter yields higher accuracy than 100% (e.g., HyperKvasir: 70.82→70.85, Kvasir Polyp: 83.92→83.96). This counterintuitive result suggests the main experimental configurations in Tables 1–3 may be suboptimal, yet the paper uses 100% tokens for all primary comparisons. No explanation or analysis is provided for why fewer tokens help.

- **Performance margins over the closest baseline (GMoE-Adapter) are consistently sub-1% without statistical significance testing**: Across Tables 1–3 and 8–9, improvements are typically 0.2–0.5 percentage points (e.g., Table 1: 70.75→70.82, Table 3: 67.76→68.23). No error bars, confidence intervals, or significance tests are provided despite 23 datasets. Given that GMoE-Adapter (Mo et al., 2024a) is itself from the same research group and shares core architectural ideas, these margins should be statistically validated.

- **Medical backbone domain mismatch is unaddressed**: The medical ViT is pretrained on 1.6 million cell images (Nguyen et al., 2023) but DynaMer is evaluated on brain MRI, chest X-ray, shoulder X-ray, and other non-cell modalities. The paper provides no analysis of whether this domain mismatch limits performance or whether a radiology-pretrained backbone would improve results. This is critical for assessing the method's actual value proposition.

- **Table 4 ablation structure is uninterpretable**: Rows 2–4 all show (✓, ✓) for General Gate and Medical Gate but have different parameter counts (1.20X, 1.21X, 1.21X) and different performance numbers. No additional column identifies what is being varied (number of experts? bottleneck dimension?). This makes the ablation impossible to interpret.

- **Table 9 uses undefined method name "GL-MoF Adapter"**: The proposed method is called "DynaMer Adapter" throughout the paper, but Table 9 lists "GL-MoF Adapter (ours)." This appears to be either an editing error or an undefined variant, creating confusion about which method was actually evaluated for OOD results.

- **Token pairing assumption lacks justification**: The architecture assumes the i-th token from the general ViT spatially corresponds to the i-th token from the medical ViT for expert computation. After transformer layers apply global self-attention, token representations mix globally—this spatial correspondence may degrade. No analysis or visualization verifies whether this pairing remains meaningful through the network depth.

## Nice-to-Haves
- **Analysis of gating weight distributions**: Visualizing how often each expert is selected across layers and modalities would reveal whether the medical backbone contributes substantively or if the general expert dominates.

- **Compute-matched single-backbone baseline**: A single-backbone model scaled to match DynaMer's total FLOPs would isolate whether gains come from domain knowledge fusion versus increased capacity.

- **Alternative medical backbone ablation**: Testing a radiology-pretrained medical ViT (if available) on X-ray/CT datasets would validate whether domain-aligned experts improve performance.

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **Title imprecision about "token merging"**: The phrase "dynamically merge tokens" is reasonable shorthand for the MoE adapter's token-level combination operation. This is a minor stylistic preference, not a substantive issue.

- **"Four four folds" typo and hyperbolic language ("ingeniously prioritizes")**: These are minor writing issues that do not affect technical correctness. ICLR reviewers should focus on substance.

- **Missing related work on model soups/task arithmetic**: While relevant to combining models, the paper's core contribution is the adapter architecture, not model merging in general. The existing related work covers the relevant MoE adapter literature.

- **Demand for theoretical proofs for this empirical contribution**: The paper makes empirical claims supported by experiments. While theoretical analysis could strengthen it, demanding formal proofs is excessive for an empirical systems contribution.

- **Complaints about concurrent work comparison**: The paper adequately compares against the most relevant prior work (GMoE-Adapter). Additional comparison with concurrent works is not required for ICLR standards.

## Novel Insights
Beyond the paper's contributions, a genuinely novel observation emerges: the counterintuitive result that 50% token skipping improves accuracy suggests the skipping router may function as implicit regularization rather than purely an efficiency mechanism. If tokens processed by the adapter are those the router deems "most informative," the remaining tokens may introduce noise that dilutes the signal. This interpretation, if correct, would reframe the skipping router as a feature selection module with secondary efficiency benefits. A follow-up analysis visualizing which tokens are skipped (e.g., background regions vs. clinically relevant areas) could validate this hypothesis and strengthen the paper's contribution.

## Suggestions
1. **Report total inference FLOPs and GPU memory**: Calculate and report the full system cost including both backbones, enabling fair comparison with single-backbone methods. A simple acknowledgment that dual backbones double base FLOPs while the adapter is efficient would address the transparency concern.

2. **Re-run main experiments with 50% token skipping**: If 50% tokens outperforms 100%, the primary comparisons should use the superior configuration, or explicitly explain why 100% was chosen despite inferior performance.

3. **Add statistical significance testing**: Run multiple seeds (at least 3) for key datasets and report mean ± std with p-values against GMoE-Adapter. The consistency of improvements across 23 datasets is encouraging but should be quantified.

4. **Fix Table 4 labels**: Clarify what varies across rows (e.g., add a column for "Number of Experts" or "Bottleneck Dimension") and correct Table 9's method name from "GL-MoF Adapter" to "DynaMer Adapter."

---

## F9JZiGradI

- GT: Reject (avg 5.2)
- Predicted: N/A (None/10)
- Match: N/A

### Final Review

ERROR: Connection error.

---

## EUAxxrxOM8

- GT: Reject (avg 5.0)
- Predicted: N/A (None/10)
- Match: N/A

### Final Review

ERROR: Connection error.

---

## 3lXZjsir0e

- GT: Reject (avg 5.6)
- Predicted: N/A (None/10)
- Match: N/A

### Final Review

ERROR: Connection error.

---

