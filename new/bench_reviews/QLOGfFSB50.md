Now I have a thorough understanding of the paper and the calibration anchors. Let me synthesize the final review.

Key observations after reading the paper:

1. **Parameter comparison fairness**: The harsh critic raises a valid concern about comparing parameter counts between DSC-based and standard-conv architectures. The paper does acknowledge it uses a DSC-based architecture (Section 3.4, the experimental setup mentions it), and references Appendix D.2 for "competing approaches with SPARC-like backbone." However, this critical comparison is in the appendix, not in the main paper, and this is a legitimate concern. Still, the paper's core contribution is NOT just "we use fewer parameters" — it's a design that uses DSCs as part of the method. The DSC architecture enables the working/semantic memory decomposition. The methodological contribution is the decomposition + EMA consolidation + weight re-normalization. The parameter efficiency is a real outcome of this design, though the comparison table does mix architectural and methodological contributions.

2. **Linear inference cost for Class-IL**: This is confirmed in Section 3.4: "each image is independently processed through all sub-networks." This is a real cost. However, the paper claims "1 F, 1 B" for SPARC in Table 1, which is the *training* cost per task. The inference cost scales linearly with the number of tasks. This IS mentioned in the paper (Section 3.4), but the paper does NOT discuss this as a limitation, nor report inference times. However, all parameter isolation methods (PNNs, etc.) have similar multi-forward-pass inference requirements. This is a minor-to-moderate concern: it is a known property of parameter isolation methods for Class-IL, and the paper does state it clearly.

3. **Missing ablations for weight re-normalization**: This is confirmed. Table 5 ablates the semantic memory, but there is no ablation for weight re-normalization (Eq. 5). The paper claims this as one of three contributions. This is a real gap.

4. **Semantic memory as EMA**: The paper explicitly describes the EMA mechanism in Eq. 4. The framing as "semantic memory" inspired by CLS theory is somewhat overclaiming, as the actual mechanism is just EMA averaging — no active consolidation or replay. However, this is more of a framing/naming concern than a methodological flaw. The paper is clear about what the mechanism does.

5. **"Matches rehearsal-based methods"**: On Seq-CIFAR10, SPARC gets 61.22% vs. DER++ at 64.88% and OCDNet at 73.38%. The abstract says "matches rehearsal-based methods on various CL benchmarks." This is a fair criticism — "matches" is a stretch for the simpler benchmarks. However, on Seq-CIFAR100 and Seq-TinyImageNet, SPARC genuinely outperforms most methods.

6. **DSC backbone baselines**: The paper references in Appendix D.2 that it compares with "competing approaches with SPARC-like backbone." Since the parser strips appendices, we don't have access to this. The paper does acknowledge the different backbone, but the key comparison is relegated to the appendix. This is a valid concern but not fatal.

Now let me compare with calibration anchors:
- SD-LoRA (7.5, oral): Novel LoRA-based rehearsal-free CL with theoretical grounding and strong empirical results. Much stronger theoretical contribution. SPARC is simpler but less theoretically grounded.
- DIRAD/PREVAL (2.33, reject): Very weak paper with poor baselines, MNIST-only, no ablations. SPARC is clearly much stronger.
- Dual-Learner DLCPA (4.0, reject/withdrawn): Simple EMA-based approach, unclear contributions, missing baselines. SPARC has stronger empirical results and clearer methodological contribution.
- YFdopzmpdr (5.2, reject): Architecture-based CL, overclaimed efficiency, unfair comparisons. Somewhat similar to SPARC concerns about architecture confounding.

SPARC falls in between: it has real empirical contributions, a reasonable method, but with some overclaimed aspects and an architecture confound. It's better than the 2-3 range papers, weaker than the 7+ papers. The main weaknesses (architecture confound in parameter comparison, missing ablation for weight re-normalization, linear inference cost) are real but not fatal.

I'd place this paper around 5-6.</think>Now I have enough information to write the final review. Let me verify one more thing about the inference cost claim in the paper.

The paper clearly states in Section 3.4: "For inference in the Class-IL setting, each image is independently processed through all sub-networks." This confirms the linear inference cost. And Table 1's "1 F, 1 B" refers to training cost per task, not inference cost for Class-IL.

Let me now compose the final consolidated review.

## Summary

SPARC proposes a rehearsal-free, parameter-efficient continual learning method that uses depthwise separable convolution (DSC) based ResNet-18 as a backbone, decomposing it into task-specific "working memories" (depthwise filters) and task-agnostic "semantic memory" (shared pointwise filters updated via EMA). It also introduces weight re-normalization to mitigate task-recency bias. SPARC achieves competitive or superior Class-IL accuracy on Seq-CIFAR100 and Seq-TinyImageNet while using far fewer parameters than surrogate-based methods like CLS-ER and OCDNet.

## Strengths

- **Strong Class-IL performance on harder benchmarks**: On Seq-TinyImageNet, SPARC achieves 32.29% Class-IL (Table 1), outperforming CLS-ER (23.47%) and OCDNet (17.60%) by large margins, and on Seq-CIFAR100 achieves 49.03% vs. CLS-ER's 43.80%. These are real improvements on the more challenging, more realistic benchmarks where buffer-based methods struggle with low buffer-to-class ratios.

- **Genuine parameter efficiency**: Even accounting for the DSC architecture, SPARC's design yields compact sub-networks per task. Table 4 shows SPARC uses only 3.62M parameters for 20 tasks vs. 2645.05M for PNNs and 33.69M for CLS-ER, demonstrating meaningful efficiency gains over parameter isolation baselines that share the same fundamental approach of growing architectures.

- **Clean architectural mapping to CLS theory**: The decomposition of DSCs into task-specific depthwise filters ("working memories") and shared pointwise filters ("semantic memory") provides an elegant, implementable mapping to Complementary Learning Systems theory (Sections 3.1–3.2, Figure 1). Table 5 validates that semantic memory consolidation nearly matches fully separate filters (49.13% vs. 51.57%) while saving 59% of parameters.

- **Honest limitations section**: Section 5 explicitly acknowledges task boundary requirements, static resource allocation, linear growth, and CNN-only design — which is commendable.

## Weaknesses

### Fatal
None.

### Major

- **Parameter efficiency claims conflate architectural and methodological contributions**: The headline claim of "using only 6% of parameters" (abstract, Section 1) compares SPARC's DSC-based backbone against baselines using standard-convolution ResNet-18. DSCs are inherently ~8–9× more parameter-efficient than standard convolutions by design — this is an architectural property, not a contribution of the CL method. Table 1's parameter comparison (1.04M vs. 11.23M–33.69M) mixes these factors. The paper does reference SPARC-backbone baselines in Appendix D.2, but this critical apples-to-apples comparison is relegated to supplementary material rather than presented prominently. Without it, readers cannot assess how much of SPARC's efficiency comes from the DSC switch versus the CL innovations (working memory, semantic memory, re-normalization). **Why this matters**: The parameter efficiency claim is the paper's central selling point, and it is partially attributable to a well-known architectural trick rather than the proposed method.

- **Missing ablation for weight re-normalization**: The paper claims three contributions (Section 1): (1) SPARC architecture, (2) task-agnostic semantic memory, and (3) weight re-normalization. Table 5 ablates semantic memory, but there is no ablation isolating the effect of weight re-normalization (Eq. 5). Since this is explicitly listed as one of the paper's three contributions and is claimed to address "task-specific biases" — a key challenge for Class-IL — the absence of this ablation leaves one of three claimed contributions unsupported. **Why this matters**: Without this ablation, it is unclear whether re-normalization provides meaningful gains or whether SPARC's Class-IL performance derives solely from parameter isolation + DSC architecture.

### Minor

- **Linear inference cost for Class-IL is acknowledged but not analyzed**: Section 3.4 states that for Class-IL, "each image is independently processed through all sub-networks." For Seq-CIFAR100 with 20 tasks, this means 20 forward passes per image at inference. Table 1 reports "1 F, 1 B" per task (training cost), but the linear inference scaling is not quantified or discussed as a limitation. The paper's repeated use of "scalable" (abstract, Sections 3, 5, 6) could mislead readers who assume scalability refers to inference, not just parameter count. **Why this matters**: For memory-constrained deployment scenarios (the paper's stated target), compute overhead at inference matters as much as parameter count, but this trade-off is left implicit.

- **"Matches rehearsal-based methods" is an overclaim for simpler benchmarks**: On Seq-CIFAR10, SPARC achieves 61.22% vs. DER++ at 64.88% and OCDNet at 73.38% — gaps of 3–12 percentage points. The abstract's claim of "match[ing] rehearsal-based methods on various CL benchmarks" is a stretch for this benchmark, though it holds on Seq-CIFAR100 and Seq-TinyImageNet.

- **Semantic memory as EMA is a simple heuristic framed as cognitive consolidation**: Equation 4 shows "semantic memory" is just an exponential moving average of task-specific pointwise filters — the shared parameters are never jointly optimized on data from Tasks 2+. This is a passive averaging mechanism, not the active consolidation through replay described in CLS theory. The cognitive framing overstates the mechanism, though the paper is transparent about what it does.

### Trivial
None.

## Nice-to-Haves

- An ablation showing SPARC with and without weight re-normalization on at least two benchmarks, which would substantiate the third claimed contribution.
- Reporting inference time per image for Class-IL across task counts, making the k-forward-pass cost explicit.
- Comparison against rehearsal-based baselines using the same DSC backbone (the Appendix D.2 results should be in the main paper).
- Analysis of what the shared pointwise filters encode (e.g., probing or visualization), which would clarify whether EMA averaging produces genuinely "semantic" features.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"DSCs claim implicit regularization without empirical/theoretical support"**: The paper states DSCs "introduce implicit regularization, which helps prevent overfitting" (Section 3.1). While this is a strong claim without direct evidence in the paper, it is a commonly cited property of DSCs in the literature (Chollet 2017; Howard et al. 2018 are cited). This is a framing concern, not a methodological flaw, and removing it from major weaknesses because it does not undermine the core contribution.

- **"Why κ=5? Why IQR and not standard deviation?"**: The harsh critic questions the lack of theoretical justification for these design choices. These are reasonable questions but are hyperparameter/design choices, not methodological errors. The paper sets κ=5 empirically and the method works — demanding theoretical justification for every hyperparameter is an unreasonable standard for an empirical paper.

- **"Rehearsal methods disadvantaged with only 200 buffer on TinyImageNet"**: The paper acknowledges this (Section 4) and provides buffer size 500 results (Table 10) and discusses the buffer-to-class ratio. Not a hidden weakness.

- **"Figure 2 only shows Task-IL, not Class-IL"**: Task-IL is the natural evaluation mode for parameter isolation methods. Showing both would be informative, but Task-IL results are not inappropriate.

- **"No comparison against trivial no-shared-parameter DSC baseline"**: Table 5's "Separate point-wise & depth-wise filters" row essentially serves this purpose — it shows the fully separate architecture still outperforms the shared variants.

- **"Missing dynamic filter allocation"**: This is listed as future work in Section 5. Criticizing absence of future work is scope creep.

- **"Inference time comparison missing"**: Promoted from trivial assessment to minor in the main review. Not a fatal flaw but worth mentioning.

## Novel Insights

The decomposition of depthwise separable convolutions into task-specific (depthwise) and task-agnostic (pointwise) components is a genuinely clever architectural insight that maps naturally onto the CLS framework. However, the most interesting empirical finding — that simple EMA averaging of pointwise filters across tasks can nearly match fully separate filters (49.13% vs. 51.57% in Table 5) — suggests that the transfer benefit from shared representations in CL is substantial even without active consolidation. This raises an underexplored question: how much of continual learning's difficulty is architectural (capacity allocation, interference) versus algorithmic (consolidation, replay), and SPARC's results hint that architectural choices may matter more than commonly assumed.

## Suggestions

- Move the DSC-backbone baseline comparison from Appendix D.2 to the main paper or at minimum, add a summary row to Table 1. This directly addresses the most critical fairness concern.
- Add a simple ablation: SPARC with and without Eq. 5 weight re-normalization. Even one benchmark would substantiate the third contribution.
- Add a brief discussion of inference compute scaling (k forward passes for k tasks in Class-IL) alongside the parameter efficiency narrative, either in Section 3.4, Section 5, or both.

## Score and Decision

**Calibration anchors used:**
- High-scoring: SD-LoRA (7.5, oral) — novel LoRA-based CL with theoretical grounding and strong empirical results across multiple benchmarks. SPARC is less theoretically grounded and has weaker novelty.
- Medium-scoring: Dual-Arch (5.2, reject) — similar architectural angle for CL, overclaimed efficiency. SPARC has stronger empirical results but similar overclaiming of efficiency.
- Medium-scoring: OVOR (6.0, poster) — rehearsal-free CL method with reasonable empirical results but with identified comparison gaps.
- Low-scoring: DIRAD/PREVAL (2.33, reject) — weak baselines, MNIST-only, no ablations. SPARC is clearly much stronger.

SPARC has real empirical contributions, a clean architectural design, and strong results on hard benchmarks. Its main weaknesses — conflating architectural and methodological efficiency, missing weight re-normalization ablation, and under-discussed inference cost — are meaningful but not fatal. It is stronger than Dual-Arch (5.2) thanks to better empirical results and a more complete method, but weaker than OVOR (6.0) and significantly weaker than SD-LoRA (7.5) due to the overclaimed efficiency and missing ablation. The paper sits in the borderline-moderate range.

MY FINAL SCORE: 5.5
MY FINAL DECISION: <orange>Reject</orange>