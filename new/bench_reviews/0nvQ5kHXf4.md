Now I have thoroughly verified all claims. Let me compose the final consolidated review.

## Summary

WASI (Weight-Activation Subspace Iteration) proposes a method for efficient on-device fine-tuning of transformer models by jointly compressing both weight matrices and activation maps into stable low-rank subspaces. Instead of performing expensive SVD at every training iteration, WASI leverages the hypothesized stability of the essential subspace to apply subspace iteration, and combines this with activation compression via Tucker decomposition (extending prior ASI work). The method is evaluated on ViT, SwinT, and TinyLlama across multiple datasets, with Raspberry Pi 5 hardware validation showing ~1.4× speedup.

## Strengths

- **Joint weight-activation low-rank training with inference savings.** Unlike LoRA-style adapters that are merged back at inference, WASI trains and infers directly in the factored low-rank representation (Eq. 8–11), simultaneously reducing both training memory and inference cost. This is a meaningful architectural distinction that prior work (ASI alone, SVD-LLM) does not provide, as acknowledged in Sec. 2 where SVD-LLM's inference matching is noted.

- **Real hardware validation on Raspberry Pi 5.** Fig. 8 reports measured per-iteration training and inference latency on a Cortex-A76 CPU, showing ~1.4× speedup at ε=0.9. This goes beyond FLOP-count simulation and is particularly relevant for the on-device learning setting the paper targets (Sec. 4.4).

- **Consistent accuracy-efficiency trade-offs across architectures and datasets.** On SwinT with ε=0.9, WASI matches vanilla accuracy across five datasets while cutting memory by up to 62× and FLOPs by 1.5×, even surpassing vanilla on CUB (Fig. 6). The ε-controlled explained variance threshold provides a principled, single-parameter knob for trading accuracy vs. efficiency (Eq. 5–7, Figs. 5–6).

- **Dynamic programming rank selection improves over ASI.** The DP strategy for rank determination reduces search cost from exponential to linear over ASI's brute-force approach (Sec. 3.3, Appendix A.2), a concrete algorithmic improvement.

- **WSI empirically matches full-SVD accuracy.** Fig. 3b shows WSI achieves the same accuracy as repeated full SVD with 1.36× fewer FLOPs, and outperforms SVD by ~35% at matched FLOP budgets. This provides empirical evidence that subspace iteration does not degrade convergence.

## Weaknesses

### Fatal
None.

### Major

- **Misleading presentation of TinyLlama results.** The paper reports "activation and weight memory drop by up to 953.86× and 30.12×" for TinyLlama (Sec. 4.3), but these numbers reflect per-layer metrics for only the 5 fine-tuned layers, not whole-model savings. While the paper does disclose this ("For comparison, we log the resource consumption only at the layers that are fine-tuned"), the 953.86× figure is stated prominently without reiterating the per-layer caveat, and the extreme value arises from using ε=0.1 (retaining only 10% explained variance) — a dramatically more aggressive setting than the ε ∈ {0.4,...,0.9} used for ViT/SwinT, with no justification for this discrepancy. The ε=0.1 setting appears cherry-picked to generate headline-grabbing numbers, and whole-model memory savings would be dramatically lower. This undermines confidence in the paper's efficiency claims and their generalizability.

- **Missing LoRA baseline for training memory comparison.** Although the paper correctly argues LoRA is a different category (adapters merged back at inference, Sec. 2), the on-device training setting makes training memory a primary concern. LoRA's training memory profile (frozen weights + low-rank adapters) should be compared against WASI's joint weight-activation compression. The paper compares against SVD-LLM (which uses LoRA adapters) and shows WASI achieves up to 100× higher memory efficiency, but a direct LoRA comparison would clarify the practical advantage. Additionally, variants like LoRA-FA (cited in the paper as Zhang et al., 2023) keep low-rank structure and reduce training memory — their absence from experiments leaves the claimed advantage over "state-of-the-art methods" partially unsubstantiated.

### Minor

- **Rank stability validated, but subspace stability assumed.** The paper's Algorithm 1 depends on subspaces being stable across iterations (not just their dimensions), but Sec. 4.2 only validates that layer *ranks* remain stable (Fig. 3a). Rank stability is weaker than subspace stability: singular vectors can rotate while spectrum shape stays fixed. However, the WSI vs. full-SVD accuracy comparison (Fig. 3b) provides end-to-end empirical evidence that the method works correctly, partially mitigating this gap. The concern is that this validation is limited to one model (ViT), one dataset (Pets), and one ε (0.8) — insufficient for a method whose generality is claimed.

- **ε=0.1 for TinyLlama is unjustified and inconsistent with other experiments.** The paper uses ε ∈ {0.4,...,0.9} for all ViT/SwinT experiments but switches to ε=0.1 for TinyLlama without explanation. No ablation or sensitivity analysis is provided for this choice, making it impossible to assess whether the extreme compression ratios are representative or cherry-picked.

- **Main experiments focus on MLP blocks only; attention layers deferred to appendix.** Attention layers often dominate memory in transformers. The paper states that "focusing on linear layers within multi-perceptron blocks for fair comparison with previous methods" (Sec. 4.1), but this means the core experimental evaluation does not demonstrate WASI's effectiveness on the components that matter most for memory.

- **Per-iteration rather than end-to-end wall-clock speedup on Raspberry Pi.** The on-device latency results (Fig. 8) report time per training iteration, not end-to-end fine-tuning time including data loading, preprocessing, and full convergence. The 1.4× per-iteration speedup may not translate directly to wall-clock savings for a complete fine-tuning run.

### Trivial
None.

## Nice-to-Haves

- Report whole-model memory savings for TinyLlama (not just the 5 fine-tuned layers) to give readers the complete picture.
- Include a direct LoRA/LoRA-FA comparison, at minimum for training memory, to position WASI's advantage clearly.
- Measure subspace stability (e.g., principal angles between successive singular vector matrices) to directly validate the assumption underlying Algorithm 1.
- Run TinyLlama with the same ε range as ViT/SwinT to enable fair cross-architecture comparison.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"PyTorch 1.13.1 is outdated"**: Software version nitpick; irrelevant to the scientific contribution. Per rules, removed.
- **"Undefined notation in Algorithm 1 (R_i^{-1,(t-1)})"**: This appears to be a parser artifact from PDF extraction, not a genuine undefined term in the original paper. Per rules, formatting/parser artifacts are removed.
- **"Gram-Schmidt approximation error never analyzed or bounded"**: Demanding theoretical convergence analysis for an empirical systems paper is outside field norms for this venue. Nice-to-have, not a weakness.
- **"f_LR gradient projection deferred to appendix"**: Missing appendix content is a parser artifact — per rules, removed.
- **"Theoretical speedup ratios are upper bounds, not realistic predictions"**: The paper explicitly states this is a simplification for analysis (Sec. 3.4: "For simplicity, we assume that the same optimal rank is applied to both A_i and W_i"). Standard practice.
- **"WSI vs SVD comparison is unfair because no one would do full SVD every iteration"**: The comparison serves to validate that subspace iteration approximates full SVD well — a meaningful scientific question, even if no practitioner would use full SVD every iteration. The critic's suggested comparison (single initial SVD + fixed subspace) addresses a different question.
- **"First method claim overclaims novelty"**: The paper's claim is scoped to "efficient model-activation-decomposition-aware training" and correctly notes ESPACE requires downstream data (infeasible for on-device). The distinction is meaningful within the stated scope.
- **"Absolute accuracy for TinyLlama is opaque"**: Figure-dependent criticism that cannot be verified from parsed text; may be a reader error due to figure quality rather than a paper deficiency.
- **"Demand for convergence analysis of joint WSI+ASI training dynamics"**: Theoretical demand outside norms for an empirical systems paper at this venue.
- **Strength claim "subspace stability assumption is empirically validated" (from Strength Finder)**: Partially conflicts with the verified weakness that only rank stability, not subspace stability, is validated. The WSI vs. SVD comparison (Fig. 3b) provides indirect empirical support but does not directly measure subspace stability. Downgraded this strength.

## Novel Insights

The gap between rank stability and subspace stability is a genuinely important observation that prior work on subspace iteration for neural network training has largely overlooked. While the paper shows WSI matches full-SVD accuracy (which implicitly validates subspace sufficiency), directly measuring principal angles between successive subspaces would provide much stronger evidence and could reveal failure modes of subspace iteration that aren't captured by rank monitoring alone. This is a potentially fruitful direction for future work on low-rank training methods.

## Suggestions

- Re-report TinyLlama results with the full ε range used for ViT/SwinT, and include whole-model memory savings (even if estimated), so readers can compare across architectures fairly.
- Add a LoRA training-memory comparison even if inference memory is not LoRA's strength — the on-device training story is incomplete without it.
- Expand the subspace stability validation beyond a single model/dataset/ε setting, ideally measuring principal angles directly.

## Score and Decision

**Calibration anchors:**

- **High (>7)**: gdZ6J5hZzF (7.33, low-rank logit structure in LLMs, novel framework + strong theory+empirics); dTWfCLSoyl (7.33, In-Place TTT, novel idea with extensive evaluation). WASI lacks the novelty and depth of validation these papers have.
- **Medium (4–6)**: QD4DL0OUmZ (4.0, LoRAct, very similar topic — activation compression for fine-tuning — rejected for limited novelty and missing baselines); f3KD7jfSWY (4.5, CERSA, SVD-based subspace adaptation, rejected for untested key assumption, missing compute characterization); HfUyKUxAq3 (4.5, LANCE, activation compression for on-device learning, rejected for questionable metrics). WASI is somewhat stronger than these due to joint weight-activation compression and real hardware validation, but shares similar issues (missing baselines, incomplete assumption validation).
- **Low (<3)**: qnvvoECibL (2.5, overclaimed sparse training with missing SOTA baselines); 4YBRDJ5TN3 (1.5, transformer compression methods that fail to maintain performance). WASI is clearly above these — it has real, working results.

WASI has genuine contributions (joint weight-activation compression, real hardware speedup, principled ε-control) but is held back by the misleading TinyLlama numbers, missing LoRA comparison, and limited validation of the core stability assumption. It sits above the rejected CERSA/LANCE/LoRAct papers (which scored 4.0–4.5) due to broader evaluation and hardware results, but well below the accepted papers at 7+. The paper is in the borderline range.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>