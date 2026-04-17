Now I have a good sense of the calibration landscape. Let me synthesize the final review.

Calibration summary:
- Papers with similar LoRA-MoE integration but weak novelty/evaluation (MoRE, MoLE, PERFT) scored 3-5 range
- Papers with similar topic but better results/insights (Pushing MoE to the Limit, Self-MoE, C-Poly) scored around 6
- Papers with strong theory but marginal empirical gains (Aux-Loss-Free) scored 3-5
- The "Pushing MoE" paper (scores 6,5,8,8, accept poster) had similar structure but more controlled experiments
- Papers that are incremental combinations (A+B type) with marginal gains tend to score in the 4-5 range

For LoRA-Mixer specifically:
- The contribution is a specific architectural choice (projection-level MoE for LoRA) + an entropy-regularized routing loss
- The novelty is somewhat incremental - routing LoRA experts is well-explored, and negative entropy regularization is standard
- The experimental scope is broad but ablations are missing for the core architectural claim
- Some gains over baselines are modest, and RSL underperforms at moderate data sizes
- The theory claims (information bottleneck, strong convexity, generalization bounds) are under-justified in the main text
- Missing key ablations: attention vs FFN placement, equal-parameter LoRA comparison, w/o RSL in main tables

This seems like a 4.5-5.0 range paper. The practical contributions are real but the novelty is moderate and the claims overreach what's supported.

## Summary

LoRA-Mixer proposes routing task-specific LoRA experts into the projection matrices of attention modules (rather than FFN layers), combined with a Routing Specialization Loss (RSL) that adds negative-entropy regularization to the standard MoE auxiliary loss to balance load balance and input-aware specialization. The framework supports both joint training and plug-and-play composition of frozen LoRAs. Across 15 benchmarks and 3 base models (including an SSM), LoRA-Mixer with RSL outperforms existing LoRA-MoE baselines while using 48% fewer trainable parameters.

## Strengths

- **Well-motivated architectural design targeting projection layers.** The argument that placing LoRA-MoE at attention/SSM projection layers enables direct influence on core representation learning (rather than FFN or parallel branches) is principled and the compatibility with both Transformers and SSMs (validated on Falcon-Mamba-7B) is a practical advantage (Figure 1, Section 1, Tables 1–2).

- **Broad empirical coverage.** Evaluation on 15 benchmarks across 5 domains with 3 base models (including a non-Transformer SSM architecture) and comparison with multiple LoRA-MoE baselines (LoRAHub, MoLE, MixLoRA, LoRA-LEGO, PHATGOOSE) provides substantial coverage. The improvements on GSM8K, CoLA, and ARC-C are notable (+3.79%, +2.90%, +3.95% vs. LoRAHub/MoLE baselines).

- **RSL loss provides practical data efficiency.** Table 9 demonstrates that RSL-trained routing achieves comparable or better performance with ~50% of the training data compared to auxiliary loss routing, and Figure 4 shows qualitatively that RSL routing produces more input-aware expert specialization vs. the near-uniform distributions produced by standard auxiliary loss.

- **Plug-and-play LoRA reuse demonstrated.** Section 4.3 shows that Internet-sourced LoRAs from LoRAHub can be composed with only 2K additional routing data, achieving gains over single LoRA fine-tuning on 4/5 GLUE tasks with Flan-T5 (Table 3).

## Weaknesses

### Major:

- **RSL is a standard technique dressed in non-standard framing, and key theoretical claims are under-justified.** The Routing Specialization Loss is simply standard auxiliary loss plus a negative-entropy regularizer: $\mathcal{L}_{\mathrm{RSL}} = \alpha \sum_i \bar{p}_i \bar{f}_i - \lambda \mathbb{E}_x[\mathcal{H}(p(\mathbf{x}))]$. Entropy regularization for routing is well-established in the RL and MoE literature, but the paper frames it as a novel "information bottleneck" perspective without sufficiently distinguishing it from prior art. The claimed "strong convexity" and "generalization bounds" (Appendix A.1/A.2) are mentioned but not substantiated in the main text—the sketch in Section 3.3 (Eq. 5-10) provides only the trivial gradient of entropy and a Lagrange multiplier derivation, without actually proving convergence or generalization properties. The claim that auxiliary loss "leads to over-averaging" is deferred to a vague Appendix A.17 reference without any intuition or formal argument in the main body. While the practical effect of RSL is validated in Table 8 and Figure 4, the theoretical framing significantly overclaims.

- **Missing critical ablations to isolate the contribution of projection-layer placement.** The paper's core architectural contribution is placing MoE-LoRA at projection layers rather than FFN layers, yet there is **no ablation** comparing LoRA-Mixer applied to attention projections vs. FFN layers under the same parameter budget. Without this, one cannot attribute gains to the specific placement. Similarly, Table 2 (the main comparison) does not include LoRA-Mixer without RSL, making it impossible to disentangle whether gains come from RSL, the architectural placement, or simply from having more total parameters (6 LoRA experts vs. 1). There is also no comparison against a single higher-rank LoRA with the same total parameter count, which is a natural baseline—if the total rank from E experts of rank r equals a single LoRA of rank E×r, the gains might simply come from parameter scaling rather than routing.

- **Modest improvements over strong baselines and statistical weakness.** On LLaMA3-8B, many improvements over single-task LoRA are marginal: SST-2: +0.11, GSM8K: +0.39, HumanEval: +1.71 (Table 2). On Mistral-7B, LoRA-Mixer actually *underperforms* single-task LoRA on GSM8K (46.48 vs. 46.67). The paper claims "all experiments are run three times and the average reported" but reports no standard deviations, confidence intervals, or significance tests. For improvements in the 0.1–1.5 point range, variance information is essential. The RSL also underperforms auxiliary loss at 4K and 6K data sizes (Table 9: –0.37 and –0.04), which is acknowledged but relegated to the appendix.

### Minor:

- **Abstract claims are misleading about baselines.** The "+3.79%, +2.90%, and +3.95% on GSM8K, CoLA, and ARC-C" in the abstract are measured against weaker baselines (LoRAHub/MoLE), not against the strongest baseline (single-task LoRA, where gains are +0.39, +0.72, +1.09 on LLaMA3-8B). This inflated framing could mislead readers.

- **F_route is under-specified.** The routing function $\mathcal{F}_{\mathrm{route}}(\{\alpha_e(\mathbf{x}) \cdot \Delta W^{(e)} \mathbf{x}\})$ is described as "representing the routing function output by the fusion expert" (Section 3.2) but never formally defined. Whether it is a simple weighted sum, a per-head routing, or includes residual connections matters for both capacity and computational cost.

- **Cross-model transfer results are mixed.** Table 5 transfers Mistral-7B parameters to LLaMA3-8B, but ARC-E actually drops (0.97× relative performance at 0-shot). The claim of "extremely robust and transferable" routing is overstated given these mixed results. No transfer experiment to SSMs (Falcon-Mamba) is provided despite claimed compatibility.

### Trivial:

- **Table formatting issues.** Tables 2, 3, and 6 suffer from alignment/parsing issues that make some entries hard to interpret. This appears to be a PDF extraction artifact rather than an author error.

## Nice-to-Haves

- **Ablation on projection vs. FFN placement.** The single most important missing experiment—directly comparing LoRA-Mixer applied to attention projections vs. FFN layers would validate the core architectural claim.

- **Equal-parameter baseline comparison.** A single LoRA with total rank matching the sum of expert ranks (e.g., rank E×r) would clarify whether gains come from MoE routing or simply from having more parameters.

- **Token-level routing analysis.** The paper claims "fine-grained token-level specialization" but only shows per-task expert load distributions. Showing within-sequence routing heatmaps would directly validate this claim.

- **λ sensitivity analysis.** The novel component of RSL is the entropy coefficient λ, but no sensitivity analysis is provided in the main text (only α is explored in the appendix).

- **Inference cost analysis.** LoRA experts cannot be merged back into the base model during inference, increasing memory and compute. A latency/memory comparison with standard LoRA would help practitioners.

## Removed Points

- **Criticism about Falcon-Mamba baseline adaptation.** The paper explicitly notes that MixLoRA is excluded from Falcon-Mamba "due to its Transformer-specific design" (Table 2 caption). The claim that other baselines might not be properly adapted for SSMs is speculation about implementation details not in the paper.

- **Criticism about undisclosed hyperparameters being a reproducibility concern.** This is a nitpick about reproducibility that is standard for this type of systems paper. The paper provides LoRA ranks and key training details; detailed hyperparameter sensitivity belongs in the appendix (which A.8 addresses).

- **Demand for comparison against baselines that also target attention projections.** The paper's point is that existing methods target FFN; requesting a baseline that targets attention projections would require implementing a variant of existing methods not proposed by anyone, which is scope creep.

- **Concern about missing related works.** Cannot verify whether cited or uncited works actually exist, so removed per hard rules.

- **Formatting/style nitpicks.** Removed per hard rules.

## Novel Insights

The observation that standard auxiliary loss in MoE routing drives routing distributions toward uniform behavior (low input-aware variance) is important in practice, even if the theoretical framing here is overstated. Figure 4 provides direct visual evidence that RSL produces meaningfully different expert load distributions across tasks compared to the near-uniform distributions of auxiliary loss. This practical benefit—achieving both load balance and input-aware specialization with less training data—is the most empirically grounded contribution, even though the information-bottleneck narrative is under-justified.

## Suggestions

1. Add a direct ablation comparing LoRA-Mixer at projection layers vs. FFN layers under identical parameter budgets and LoRA experts. This is the single most important experiment for validating the architectural claim.

2. Include "LoRA-Mixer without RSL" (i.e., standard auxiliary loss) in the main comparison tables to isolate RSL's contribution.

3. Report standard deviations across 3 runs for all main results, and reframe abstract claims relative to the strongest baseline (single-task LoRA) rather than weaker baselines.

4. Discuss the RSL underperformance at 4K/6K data in the main text rather than deferring to the appendix, and provide λ sensitivity analysis.

5. Clarify the F_route formulation explicitly in the method section.

## Score and Decision

**Calibration:**
- MoRE (LoRA-MoE for multi-task, weak novelty, marginal gains): scores 3,3,5,5, rejected
- MoLE (gated LoRA combination, marginal improvements, unclear novelty): scores 3,5,6,6, accepted poster
- Pushing MoE to the Limit (MoE + PEFT combination, good experiments, accepted): scores 6,5,8,8, accepted poster
- PERFT (PEFT framework for MoE, A+B type, limited novelty): scores 8,3,5, rejected
- C-Poly (parameter-efficient multi-task LoRA combination): scores 6,6,6,6, accepted poster
- Aux-Loss-Free Load Balancing (MoE routing loss, overclaimed theory): scores 3,5,5,3, rejected
- Self-MoE (self-specialized LoRA experts compositional model): scores 6,6,6,6, accepted poster

LoRA-Mixer shares similarities with the mid-range LoRA-MoE papers (MoLE, Pushing MoE, C-Poly) in being a practically useful combination of known ideas with reasonable empirical results. However, it has more issues than papers like Self-MoE or Pushing MoE: the theoretical claims overreach (similar to Aux-Loss-Free, which was rejected), key ablations are missing, and gains over the strongest baseline are sometimes marginal. RSL is essentially entropy-regularized auxiliary loss—a known technique—in a new packaging. The paper does offer genuine practical value (broad evaluation, SSM compatibility, plug-and-play LoRA reuse), but the core novelty is moderate and the empirical validation has significant gaps.

I place this paper below C-Poly and Self-MoE (6s) due to the missing ablations, overclaimed theory, and marginal gains, but above MoRE and Aux-Loss-Free (3-5 range) due to the broader experimental scope and genuine practical utility.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>