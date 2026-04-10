=== CALIBRATION EXAMPLE 12 ===

# Final Consolidated Review
## Summary
This paper introduces µLOs: learned optimizers (LOs) meta-trained under Maximal Update Parametrization (µP). The authors derive the µP scaling rules for two state-of-the-art LO architectures (VeLO and small_fc_lopt) and propose a simple, compute-efficient multi-width meta-training recipe. Empirically, µLOs demonstrate substantially improved meta-generalization to wider networks compared to standard-parameterization LOs. Unexpectedly, they also show strong generalization to deeper networks and much longer training horizons, despite being meta-trained only on shallow MLPs for short horizons.

## Strengths
- **Novel theoretical application of µP to learned optimizers.** The paper provides a principled derivation (Propositions 4.1 & 4.2) of µP for two complex LO architectures, formally bridging hyperparameter transfer theory with meta-learned optimization. This is a specific, non-trivial extension beyond prior work which focused on hand-designed optimizers.
- **Systematic and extensive empirical validation.** The evaluation suite is comprehensive, spanning 35 tasks across image classification (MLPs, ViTs) and language modeling, with rigorous use of multiple seeds and error bars. The experiments clearly demonstrate that µLOs meta-trained on small MLPs can smoothly optimize much wider networks (up to 8192 width), whereas standard LOs diverge. The compute budget is FLOP-matched to baseline LOs, making the improvement clearly attributable to the method.
- **Unexpected and valuable empirical discoveries.** Beyond the theorized width generalization, the paper shows that µLOs generalize surprisingly well to networks 5× deeper and training horizons 25× longer than those seen during meta-training. These are practically significant findings that suggest broader stabilizing benefits of the µP framework.

## Weaknesses
### Major:
- **Claims of outperforming hand-designed optimizers require careful qualification.** The paper shows µLOs achieve better average rank than per-task-tuned AdamW and µAdam when the hand-designed optimizers are tuned *only on width=1024* and transferred zero-shot to larger widths. This setup is reasonable for studying zero-shot transfer, but the strong wording ("substantially outperform...hand-designed optimizers") could imply a broader superiority that is not established. A per-width tuned AdamW oracle baseline is absent due to computational constraints (acknowledged in Limitations). The claim should be tempered to reflect that µLOs outperform these baselines *under this specific zero-shot transfer protocol*.
- **Limited analysis of why µP improves depth and horizon generalization.** The improved generalization to deeper networks and longer unrolls is presented as a surprising empirical finding with only a brief hypothesis linking it to pre-activation stability. A deeper mechanistic analysis (e.g., tracking gradient statistics, update norms, or loss landscape properties across depth and time for µLO vs. SP LO) is missing. Without this, the findings remain observational rather than explained, limiting insight into the method's full capabilities.

### Minor:
- **Meta-training distribution is narrow.** The core µLOs are meta-trained exclusively on MLP image classification tasks. While evaluation on ViTs and language models shows positive transfer, the paper does not ablate whether including architectural diversity in meta-training would further close the performance gap on these far out-of-distribution tasks. This limits the strength of claims about universal meta-generalization.
- **Evaluation focuses on training loss.** All main results report training loss at fixed steps. While this is a standard metric for optimizer comparison, reporting final validation/test accuracy or convergence to a solution quality threshold would provide a more complete picture of practical utility, especially for the long-horizon experiments.

### Trivial:
- **Maximum width tested is bounded by computational resources.** The paper acknowledges it cannot test widths beyond 8192 for MLPs and 3072/12288 for transformers. This is a reasonable limitation for an academic study and does not invalidate the demonstrated trends.

## Nice-to-Haves
- An ablation study directly comparing the benefit of µP versus simply training a standard LO on multiple widths (i.e., an SP LO trained on widths 128,512,1024) would help isolate the contribution of the parameterization from the multi-width training recipe.
- A brief discussion comparing the potential of µP to other transferable parameterizations mentioned (e.g., CompleteP, SP with layer-wise LR) for meta-learning optimizers would provide useful forward-looking context.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Strength: "The paper is well-written"** - Removed as a generic strength.
- **Weakness: "The hand-designed baselines are unfairly compared because they are not tuned per width"** - Weakened and moved to Major Weaknesses. The paper's setup is an intentional, asymmetric zero-shot transfer comparison to prove a specific point about meta-generalization. The limitation is acknowledged, and demanding a full per-width tuning for all baselines is outside the paper's stated scope and computationally prohibitive.
- **Weakness: "Missing statistical significance tests"** - Removed. The paper reports averages over 5 seeds with standard error bars, which is standard practice in the field. Demanding formal significance tests is a methodological practice not universally required at this scale.
- **Weakness: "No wall-clock time comparisons"** - Removed. The paper states meta-training is FLOP-matched and reports GPU hours for µLO_M. Demanding detailed wall-clock breakdowns is a reproducibility nitpick not central to the core claims.
- **Weakness: "The theoretical assumption of alignment/LLN scaling is not verified"** - Removed. The paper provides empirical pre-activation stability plots (Fig. 2) as support, and the assumption is standard in the µP literature. Demanding further verification is scope creep.

## Suggestions
- Revise the abstract and results sections to more precisely frame the comparison with hand-designed optimizers (e.g., "outperform under a zero-shot transfer protocol" or "achieve better average rank than per-task-tuned baselines when those baselines are tuned on a smaller proxy task").
- In the discussion of depth and horizon generalization, expand the hypothesis section with a more detailed analysis plan or cite preliminary evidence (e.g., from appendix) to better ground the speculation.

**Overall Quality Assessment:** This is a **strong paper**. It makes a **novel** and **technically sound** contribution by successfully applying µP theory to learned optimizers. The **empirical support** is extensive, rigorous, and clearly demonstrates the core claim of improved width generalization. The **significance** is high, as robust meta-generalization is a critical bottleneck for practical learned optimizers. The **clarity** is good, with a well-structured narrative and thorough experimental presentation. The weaknesses identified are reasonable but do not undermine the paper's solid core contributions.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 2.0]
Average score: 5.0
Binary outcome: Accept
