## Summary

This paper identifies that Vision-Language Models (VLMs) in few-shot settings disproportionately rely on spurious attributes within attribute pools, harming out-of-distribution generalization. The authors propose two modules: Spurious Attribute Probing (SAP) to automatically identify and filter spurious attributes using MLLMs and Concept Bottleneck Models, and Spurious Attribute Shielding (SAS) to mitigate spurious feature learning via pseudo-category construction. Evaluation across 11 datasets and three generalization tasks shows consistent improvements.

## Strengths

- **Strong empirical validation of the core hypothesis**: Table 1 demonstrates that manually removing <7% of spurious attributes from existing pools significantly improves OOD accuracy (e.g., CPL "New" increases from 65.30% to 67.66%) without compromising in-distribution performance. This provides concrete evidence distinguishing this work from prior attribute-based methods that assume all generated attributes are beneficial.

- **Targeted debiasing evidence**: Table 2 shows SAS improves accuracy on a "Counter" test group (images filtered to remove spurious attributes) more substantially than on the standard test set (e.g., CPL Counter gains 4.47% vs. 1.07% on Test). This differential gain provides specific evidence that the method reduces reliance on spurious features rather than offering generic performance improvement.

- **Comprehensive evaluation scope**: The paper evaluates on 11 datasets across three generalization tasks (base-to-new, cross-dataset transfer, domain generalization) with consistent improvements across multiple PEFT baselines (CoCoOp, PromptSRC, CPL, ArGue, etc.), providing broad empirical support.

- **Adaptive thresholding improvement**: Table 4 demonstrates the proposed adaptive threshold strategy (γ_c) outperforms fixed thresholds on Harmonic Mean (80.38 vs. 80.03 for best fixed), indicating a technical improvement over static heuristics for attribute selection.

## Weaknesses

### Fatal

None

### Major

- **SAS confounds debiasing with generative data augmentation**: The SAS module constructs pseudo-categories using synthetic images from Stable Diffusion, effectively introducing generative data augmentation into training. Baselines (CoCoOp, PromptSRC, CPL, etc.) do not use external generative models for additional training samples. While Table 4 shows performance drops when the spurious attribute threshold γ is too high or too low, this does not fully disentangle the augmentation effect from the debiasing effect. Without a control baseline where SAS generates pseudo-categories for *random* attributes (not identified as spurious), the claim that gains stem specifically from "mitigating spurious attributes" rather than regularization from additional synthetic data remains unsupported. This undermines the "state-of-the-art" comparison against standard PEFT methods.

- **SAP identification accuracy unvalidated**: SAP identifies spurious attributes by thresholding CBM weights trained on 16-shot support sets, but the paper does not report Precision/Recall of SAP's identification against the manual annotations used for Table 1. Without this validation, there is no evidence that SAP is actually finding the "Black Sheep" it claims to target. The gap between manual removal performance (Table 1) and SAP-assisted performance is not explicitly analyzed, leaving uncertainty about whether SAP's automatic identification matches the hypothesis-validation manual process.

### Minor

- **Theoretical gap between semantic "non-core" and statistical spuriousness**: SAP assumes attributes deemed semantically "non-core" by an MLLM (e.g., 'road' for 'car') are the spurious correlations harming generalization. However, semantic centrality does not guarantee statistical invariance—a semantically peripheral attribute (e.g., 'background texture') might be more stable across domains than a core part (e.g., 'wheel' design which changes by model). The paper does not analyze cases where MLLM-deemed "non-core" attributes were actually robust across domains, leaving the theoretical justification for SAP's filtering mechanism incomplete.

- **CBM weight stability in few-shot regimes unanalyzed**: Training a linear probe to determine attribute importance on only 16 shots per class is prone to high variance. The paper does not report variance of CBM weights (𝒲) across different few-shot seeds, nor does it validate SAP's identification stability. If CBM weights are noisy, SAP may filter core attributes or retain spurious ones based on sampling noise rather than true correlation.

### Trivial

- **Computational cost discussion incomplete**: Section 4.3 and Table 5 address training time but do not quantify API/carbon costs of running GPT-4V and Stable Diffusion for every few-shot task. For practitioners considering adoption, a cost-benefit analysis comparing compute overhead to accuracy gains would be useful.

## Nice-to-Haves

- Compare SAS against standard few-shot augmentation techniques (e.g., MixUp, CutMix) to isolate the value of the "spurious shielding" mechanism beyond generic augmentation benefits.

- Include failure case analysis showing examples where SAP incorrectly filtered a core attribute or SAS failed to mitigate a spurious correlation, which would build trust in the method's limitations.

- Visualize generated pseudo-category images from Stable Diffusion to assess quality and confirm the model is learning semantic attributes rather than SD artifacts.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Reviewer claim about "unfair comparison" due to hyperparameters**: The paper explicitly states in Section 4 (Implementation Details): "we strictly adhere to the settings of existing works, including optimizers, batch size, learning rate, and other strategies." This is not an unfair comparison—the authors intentionally use baseline hyperparameters to demonstrate plug-and-play compatibility.

- **Reviewer claim about Figure 5 saliency maps not proving spurious feature suppression**: While qualitative, Figure 5 does provide supporting evidence when combined with Table 2's Counter group results. This is appropriate supplementary evidence, not a standalone proof.

- **Reviewer claim about "first approach" overlooking MLLM supervision**: The paper's claim in Section 1 states "without explicit human supervision"—MLLM supervision is indeed weak supervision, not explicit human labeling. The claim is accurate as written.

- **Strength Finder claim about "efficiency optimization"**: Table 5's selective trick reduces time by ~22% but this is a minor engineering contribution, not a core strength. Moved to Nice-to-Have for cost discussion.

- **Harsh Critic claim about "missing appendix proofs"**: Per hard rules, weaknesses about missing appendix content must be removed—the parser strips those sections from all papers.

## Novel Insights

The paper's core insight—that a small subset (<7%) of spurious attributes within otherwise beneficial attribute pools disproportionately harms OOD generalization—is genuinely novel and well-supported by Table 1's manual ablation. The "Black Sheep" framing effectively captures this phenomenon. However, the automatic identification (SAP) and mitigation (SAS) mechanisms introduce confounds that prevent fully validating whether the proposed *methods* (as opposed to the underlying insight) are the source of improvement. The Counter group analysis in Table 2 is a clever evaluation design that provides stronger evidence for targeted debiasing than standard accuracy metrics alone.

## Suggestions

1. **Add a random-attribute control baseline**: Generate pseudo-categories for randomly selected attributes (not identified as spurious) and compare performance. If random attributes yield similar gains, the benefit is from augmentation; if spurious-specific attributes outperform, the debiasing claim is supported.

2. **Report SAP identification metrics**: Against the manual annotations from Table 1, compute Precision/Recall/F1 for SAP's automatic spurious attribute identification to validate the probing mechanism.

3. **Include CBM weight variance analysis**: Run SAP across multiple few-shot seeds and report standard deviation of attribute weights to demonstrate identification stability.

4. **Analyze semantic vs. statistical robustness cases**: Identify and discuss examples where MLLM-deemed "non-core" attributes were actually stable across domains, acknowledging the limitation and potential risks of filtering valid contextual cues.

## Score and Decision

**Calibration anchors retrieved:**

| Paper | Avg Score | Decision | Comparison |
|-------|-----------|----------|------------|
| mruL1LDjzV | 6.67 | Accept | Hierarchical DRO for spurious correlations—stronger theoretical grounding, similar empirical scope |
| KOO1cDm2bt | 6.50 | Accept | Task vector merging for accuracy-robustness—clearer causal attribution, comprehensive evaluation |
| ZGJJF1e2u0 | 6.00 | Accept | Manifold-preserving fine-tuning on 11 datasets—similar evaluation scale, stronger theoretical justification |
| td682AAuPr | 6.00 | Accept | Preference optimization for spurious associations—addresses similar problem with cleaner methodology |
| phRRjC0Da6 | 6.00 | Reject | Bayesian prompt distributions—strong results but missing key baselines (similar to this paper's gaps) |
| UZBQ7iZzYz | 5.20 | Accept | CBM for few-shot text—similar concerns about LLM-generated concept reliability in few-shot |
| SGSF9t9Vq2 | 5.00 | Accept | Generation-based debiasing—addresses augmentation confound more explicitly |
| F1pWCHoSSA | 4.00 | Reject | Spurious cue debiasing—unclear causal attribution, limited evaluation (2-3 benchmarks vs. 11) |

**Positioning:** This paper has stronger empirical evidence than F1pWCHoSSA (4.00, rejected for unclear causal attribution with only 2-3 benchmarks), but shares similar methodological gaps regarding augmentation confounds and identification validation. Compared to accepted papers at 6.0+ (ZGJJF1e2u0, KOO1cDm2bt), this paper has comparable evaluation scale but weaker theoretical grounding and missing control baselines. The paper sits between the 5.0-6.0 range: the core hypothesis is well-supported (Table 1, Table 2 Counter analysis), but the proposed methods have addressable limitations that prevent strong acceptance. Positioning at **5.5** reflects borderline accept/poster quality—genuine contribution with clear areas for improvement.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>