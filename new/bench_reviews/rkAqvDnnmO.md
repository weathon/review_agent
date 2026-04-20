Let me construct the final consolidated review based on my thorough analysis of the paper.

## Summary

The paper proposes SimE, a class-incremental learning framework that inserts adapters into a frozen CLIP vision encoder, fine-tunes them only on the first task, and then freezes all weights for subsequent tasks while expanding a prototype-based classifier. The paper also introduces a "Multi-Adapter" variant with multiple adapter sub-modules per block and provides a systematic study of CLIP backbone scales and pre-training datasets for IL tasks.

The most significant evidence supporting the contribution is Table 1, where SimE achieves 91.66% Avg on CIFAR-100 (10 steps), outperforming ZSCL (85.94%) by ~5.7%, and 86.47% on TinyImageNet (5 steps), outperforming ZSCL (79.35%) by ~7.1%. These margins are consistent across step counts and datasets. The method requires significantly fewer trainable parameters (~1-3.5M in Table 2) and no memory bank.

## Strengths

- **Strong empirical accuracy across benchmarks:** Table 1 shows consistent 5-9% gains over contemporary CLIP-based IL methods (ZSCL, LwF-VKD, Continual-CLIP) on both CIFAR-100 and TinyImageNet across 10/20/50 step settings. The results are competitive and the margins are meaningful.

- **Systematic study of CLIP backbone choice and pre-training data:** Tables 3 and 4 provide practical guidance for IL practitioners by evaluating 5 pre-training datasets (WIT-400M through DataComp-1B, CommonPool-1B) and 4 backbone configurations (ViT-B/16 through ViT-L/14-336px). The consistent finding that LAION-2B pretraining and ViT-L/14 backbones outperform others is actionable and reproducible.

- **Parameter-efficient, memory-free design:** The framework requires no replay buffer and adapts only a small number of parameters on the base task (~1.19-3.57M per Table 2), addressing real deployment constraints around storage, privacy, and compute that are relevant to the CL community.

## Weaknesses

### Fatal

None identified. The core methodological design (frozen backbone after base-task adapter tuning + prototype classifier expansion) is a legitimate IL approach, as evidenced by related works in this space also using frozen-backbone paradigms. The paper is transparent about this design (Section 3.1: "we finetune the trainable parameters in SimE for task 1, while freezing all the parameters in SimE for the remaining tasks").

### Major

- **Inconsistent and confusing parameter reporting undermines efficiency claims.** The abstract states SimE uses "only thousands of parameters," Table 2 lists 0-3.57M trainable parameters for the Multi-Adapter variants, and the parsed Figure 4 data references ~10M and ~150M. While some of this discrepancy may be parser artifacts in Figure 4, the "thousands" vs "millions" inconsistency is present in the paper's own text. Furthermore, Table 2 contains apparent errors: three distinct adapter configurations (AdaptMLP alone, AdaptMLP+AdaptAtten, AdaptMLP+AdaptAll) are all reported at 1.19M parameters, which is implausible if additional adapter sub-modules are being added without changing bottleneck dimensions. This makes the Multi-Adapter ablation — one of the paper's stated contributions — difficult to interpret. This does not invalidate the accuracy results, but the efficiency claims cannot be independently verified.

- **Limited methodological novelty beyond combination of known techniques.** The SimE framework combines three well-established components: (1) adapter-based PEFT (AdaptFormer, 2022), (2) prototype-based classifier expansion (Snell et al., 2017 prototypical networks), and (3) frozen-backbone transfer for tasks 2+. None of these components are novel in isolation. The paper's primary claimed novelty is the Multi-Adapter ablation and the observation that more intra-block adapter connections don't always improve performance. However, as noted above, the parameter reporting issues in Table 2 weaken the evidence for this claim. The systematic CLIP study (Tables 3-4) is genuinely useful but is empirical rather than methodological.

### Minor

- **Missing comparisons with contemporary prompt-tuning CIL methods.** The paper compares against CLIP-based IL methods (ZSCL, LwF-VR, CoOP, Continual-CLIP) but omits other parameter-efficient CLIP IL approaches like DualPrompt, L2P, and RanPAC, which share the same frozen-backbone, parameter-efficient setting. Including these would better position SimE's efficiency-accuracy tradeoff, particularly for the claims about "thousands of parameters" — these prompt-based methods also achieve high efficiency.

- **No statistical reporting or robustness assessment.** Table 1 and all accuracy results are single values with no standard deviations, confidence intervals, or results averaged over multiple class-order permutations. Given that incremental learning performance depends heavily on class ordering, reporting mean accuracy ± std over multiple seeds/task orders is standard practice and would strengthen the reliability of the reported gains over baselines.

- **Equation 3 is notationally imprecise.** Equation 3 ($E(\mathbf{x}) = \sum_i^B (g_i(\phi_i, f_i(\theta_i, \mathbf{x}_i)) + d_i(\tilde{\eta}_i, \mathbf{x}_i))$) ambiguously presents the adapter output $d_i$ as simply added to the frozen block output. In practice, adapters use residual connections, and the summation notation suggests a global sum over blocks rather than sequential composition. Equation 7 compounds this with a double sum over blocks and adapter sub-modules without clarifying how sub-modules are composed.

## Trivial

- The paper's Figure 4 subplots, as parsed, show garbled data (e.g., "~150M" parameters for "Ours" in the steps comparison subplot). These appear to be parser extraction artifacts rather than paper errors.

## Nice-to-Haves

- Provide gradient or attention analysis to support the claim that early block adapters (positions "1-3") capture more "crucial primary features" (Figure 5). t-SNE/UMAP visualizations of extracted features across incremental tasks could show whether the frozen protocol maintains discriminative manifolds for new classes.
- Add a variant that incrementally adapts adapters on each subsequent task to empirically justify why freezing after Task 1 is preferred over true incremental adaptation.
- Compare SimE against traditional IL methods with the same frozen CLIP backbone (rather than methods trained from scratch) to isolate the effect of the adapter design from the backbone advantage.

## Removed Points

These points are flagged to be removed, treat them with caution:

1. **"The method freezes all weights after Task 1, fundamentally misaligning with CIL literature where continual adaptation is the core challenge."** — Removed. The frozen-backbone + prototype classifier expansion is a legitimate IL protocol. The paper is transparent about this design choice (Section 3.1). Related work using frozen features with non-parametric classifier expansion also exists in the CIL space. This is a design scope decision, not a fundamental flaw.

2. **"Missing related works: DualPrompt, L2P, RanPAC, S-Prompts, ADAM."** — Partially removed. While these are relevant CLIP-based IL methods, the paper cannot be expected to cover all parameter-efficient IL approaches. Including 2-3 of these is worth mentioning as missing baselines (in Minor), but treating their complete absence as a structural weakness is scope-adjacent.

3. **"Figure 4 garbled data showing ~150M parameters mixes units and prevents reproducible comparison."** — Removed entirely. The parsed Figure 4 data is a parser artifact (all methods show "~150M" in the steps comparison subplot). The original submission does not have these garbled numbers.

4. **"Fren-time, ARCLIP, and Bwarp-CL lack citations."** — Removed. These may be cited in the appendix or references section, which the parser strips ("Rest of paper (reference and Appendix) is removed"). Also falls under the "missing references" hard rule.

5. **"Equation 3 is mathematically incorrect/amiguous making exact reproduction impossible."** — Partially removed. The equations are notationally awkward but the intended architecture is clear from the text and figures. This is a presentation issue, not a methodological one.

6. **"The efficiency claims are contradicted by Table 2 showing ~1.19M-3.57M parameters rather than 'thousands'."** — Note: The "thousands" claim in the abstract is indeed imprecise (should be "~1 million" or similar), but the paper consistently reports parameter counts in tables. This is a minor wording issue, partially reflected in the Major weakness.

## Novel Insights

The paper's most novel observation is the non-monotonic relationship between intra-block adapter connections and IL performance (Table 2) — specifically, that adding AdaptAtten or AdaptAll alongside AdaptMLP can degrade or only marginally improve performance at smaller step counts, while showing benefits at larger steps (50 steps). This suggests that excessive parameter capacity within individual transformer blocks may interfere with the stable feature extraction needed for incremental tasks when the adaptation is limited to the base task. However, this finding is partially undermined by the parameter reporting inconsistencies in Table 2. The paper's broader empirical contribution — systematically demonstrating that LAION-2B pre-trained ViT-L/14 backbones consistently outperform other CLIP configurations for IL — is not novel per se but fills a gap in the CLIP-IL literature.

## Suggestions

1. **Clarify the parameter budget consistently.** Update the abstract to reflect actual parameter counts (~1M adapters) rather than "thousands," and ensure Table 2 correctly reports parameters for each Multi-Adapter configuration.
2. **Report results with statistical significance.** Include mean accuracy ± standard deviation over at least 3 random class-order permutations to establish the robustness of the reported margins.
3. **Add comparisons with prompt-based IL baselines.** Include DualPrompt, L2P, or RanPAC under identical backbone and split configurations to validate that SimE's efficiency gains hold against the full spectrum of parameter-efficient CLIP IL methods.
4. **Clarify the forward pass equations.** Replace the summation notation in Equation 3 with explicit sequential composition, and similarly revise Equation 7 to clearly specify how Multi-Adapter sub-modules are combined.

## Score and Decision

**Calibration anchors used:**
- **High-scoring anchor:** C-CLIP (`sb7qHFYwBc.md`, scores 6,6,8,6, decision Accept) — multimodal CL with CLIP adapters that prevent forgetting while preserving zero-shot capabilities. Scores higher than this paper due to stronger methodological novelty (multimodal CL framework) and no major reporting issues.
- **Mid-range anchor:** TAIL (`RRayv1ZPN3.md`, scores 8,6,5,6,6, decision Accept) — bottleneck adapter PEFT for continual task adaptation. Scores high due to clear methodological framing and no reporting inconsistencies. MetaAdapter (`88hh5GtLBJ.md`, scores 5,5,6,5,6, decision Reject) — frozen backbone IL with adapters, similar design paradigm but flagged for missing baselines.
- **Low-scoring anchor:** ProCEED (`gDZd8UGaxS.md`, scores 1,3,3,5, decision Withdrawn/Reject) — prototype-based IL with severe methodological concerns; YoooP (`H6pf70GZVU.md`, scores 6,3,5,6, decision Reject) — prototype IL with weak novelty.

This paper sits **above Low-scoring anchors** because it has real, strong empirical results on standard benchmarks and a genuinely useful systematic CLIP study. It sits **below High-scoring anchors** (C-CLIP, TAIL) because its methodological novelty is limited to a combination of known techniques, and the parameter reporting inconsistencies in Table 2 undermine the Multi-Adapter contribution. It is comparable to MetaAdapter (borderline, 5-6 range) but has slightly stronger empirical results and no misinterpretation of baselines. However, the novelty is more limited than what typically scores 6-7.

The paper delivers meaningful empirical results and practical guidance for CLIP-based IL, but the core methodological contribution is incremental, and there are notable reporting issues that weaken the evidence for the Multi-Adapter findings.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>