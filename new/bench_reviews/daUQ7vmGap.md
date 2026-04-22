Now I have enough information to write the final review. Let me consolidate my findings.

## Summary

The paper investigates whether Dynamic Sparse Training (DST) can outperform Dense Training in corruption robustness, proposing the "DSCR Hypothesis" that DST at low sparsity levels improves robustness against image corruption. Through extensive experiments across multiple DST algorithms (SET, RigL, MEST, GraNet), architectures (CNNs, DeiT, 3D ConvNets), datasets (CIFAR-C, ImageNet-C/3DCC, UCF101), and corruption types, the paper finds that DST models generally achieve higher robustness accuracy than their dense counterparts, particularly against noise-type corruptions. A spectral-domain analysis suggests that DST's implicit regularization reduces reliance on high-frequency information, explaining the advantage against high-frequency corruptions.

## Strengths

- **Novel empirical finding with broad experimental scope**: The paper systematically evaluates DST corruption robustness across 9 scenarios spanning images and videos, 5 architectures, 7 corruption benchmarks, 4 DST algorithms, and both regrow strategies (random vs. gradient-based). This breadth (Figure 2, Table 1, Figures 3-4) makes the directional finding hard to dismiss as an artifact of a single setup, and addresses a genuine gap—no prior work has systematically evaluated DST on corruption benchmarks.

- **Spectral-domain analysis provides a testable mechanistic hypothesis (Section 5.2, Figure 7)**: The frequency attenuation experiments show that DST models degrade similarly to dense models under low-frequency attenuation but degrade *less* under high-frequency attenuation. This is a concrete, falsifiable observation that explains why DST particularly helps with noise-type (high-frequency) corruptions, grounding the empirical finding in a spectral mechanism rather than merely post-hoc reasoning.

- **Large robustness gains at high severity for noise-type corruptions**: Figure 3 shows nearly 25% relative accuracy improvement at severity level 5 for impulse and Gaussian noise on ImageNet-C. The video results (Table 1, UCF101: 1.5–2.7 pp gains) are the most convincing in the paper, demonstrating practically meaningful margins.

- **Unified experimental framework controlling for regrow strategy**: The systematic evaluation of both random regrow (SET, MEST_r, GraNet_r) and gradient-based regrow (RigL, MEST_g, GraNet_g) across all methods disentangles the effect of memory budget policy from the regrow criterion.

## Weaknesses

### Fatal
None.

### Major

- **No variance or significance reporting despite marginal improvements on the most important (ImageNet-scale) experiments**: Table 2 shows differences of 0.26–1.06 pp on ImageNet-C (e.g., Dense 38.38% vs. RigL 38.70%, MEST_g 38.64%, GraNet_g 38.72%) and 0.25–0.84 pp on ImageNet-3DCC. These margins are within typical run-to-run variance for single-training deep network evaluations. The paper reports no standard deviations, confidence intervals, or multi-seed results anywhere. This directly undermines the core claim of "consistent outperformance"—without variance reporting, the sub-1pp ImageNet results cannot be distinguished from noise. For CIFAR-scale experiments the margins are larger (~4 pp on CIFAR100-C), which is more convincing, but the paper's framing places equal weight on all scenarios.

- **Clean test accuracy is never reported for any experiment**: All reported metrics are robustness accuracy (on corrupted data only). Without clean accuracy, it is impossible to determine whether DST's robustness gains reflect a genuine *robustness advantage* or simply a regularization effect that shifts the accuracy-robustness trade-off (i.e., potentially trading clean performance for corruption performance). This fundamental control is absent throughout Sections 4.1–4.3 and Table 2. If clean accuracy were also comparable or better, it would substantially strengthen the paper; if it is lower, the interpretation changes entirely.

### Minor

- **Rhetorical overclaiming relative to evidence**: The paper frames DST as the "unexpected winner" over dense training (title, abstract), but this is only demonstrated against *vanilla* dense training. The paper itself reviews AugMix, AugMax, PRIME, and other standard robustness techniques (Section 2.1) that practitioners routinely combine with dense training. The comparison is valid as a base-paradigm comparison, but the "unexpected winner" framing implies broader practical implications than the evidence supports. The authors partially acknowledge this in Section 6 ("vanilla DST outperforms vanilla Dense Training"), but the abstract and conclusion do not reflect this nuance.

- **Spectral analysis is correlational, not causal**: Section 5.2 establishes that DST models are less affected by high-frequency attenuation (Observation 2, Figure 7), and the paper infers that DST reduces reliance on high-frequency components, causing improved robustness. However, the direction of causality is not established—DST models could simply be *more robust overall* (including to the attenuation perturbation itself). A controlled experiment (e.g., training dense models with low-pass filtered inputs and testing on corruptions) would strengthen the causal chain, but is absent. This limits the mechanistic conclusion to a well-grounded hypothesis rather than a demonstrated mechanism.

- **Table 2 cherry-picks one sparsity level per scenario**: Each row selects a single sparsity ratio, presenting the best-case view for DST. Figure 2 shows more nuanced results where some DST methods at certain sparsity levels (e.g., sparsity 0.7 on CIFAR10-C) dip below the dense baseline. Presenting results at only one sparsity level in the summary table overstates the consistency of DST's advantage.

- **"Without adding resource cost" claim in the abstract is imprecise**: RigL requires full-gradient computation for its regrow step (acknowledged in Figure 1 caption), MEST uses a soft memory bound with extra parameters during training, and GraNet starts from a denser network. While the footnote on page 9 addresses the binary masking cost, it does not address RigL's full-gradient overhead or GraNet's initial over-parameterization. The claim is directionally reasonable (DST saves memory and compute overall) but too absolute for methods that add training-phase overhead.

## Nice-to-Haves

- Report clean accuracy alongside robustness accuracy in all tables and figures to reveal the full accuracy-robustness trade-off.
- Run multiple seeds (≥3) and report mean ± std for ImageNet-scale experiments to establish statistical significance of sub-1pp margins.
- Compare DST against dense training + standard augmentation (e.g., AugMix) as an additional baseline to assess practical implications.
- Include a controlled experiment training dense models with frequency-aware regularization to test the causal mechanism proposed in Section 5.2.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Straw-man baseline" as a fatal/structural flaw**: The harsh critic argued the dense baselines are "straw men" because they use no robustness techniques. However, the paper is explicitly comparing *base training paradigms* (sparse vs. dense), and the authors note in Section 6 that they analyze DST + data augmentation complementarity in the appendix. The comparison is valid for its stated purpose—showing DST provides implicit regularization benefits. The issue is framing, not methodology—moved to Minor (overclaiming).

- **Disentangling sparsity from dynamic topology (static sparse control)**: The harsh critic wanted a comparison against static sparse training. This would strengthen the paper but is outside its stated scope of comparing DST vs. dense training. Moved to Nice-to-Have.

- **Demand for dense + low-pass filter augmentation experiment**: While this would establish causality for the spectral mechanism, demanding controlled causal experiments for what is explicitly presented as observational analysis is a high bar. Moved to Nice-to-Have.

- **25% relative gain needing absolute numbers**: The 25% relative gain at severity 5 is presented alongside the absolute robustness accuracy in the figure titles (e.g., 38.64% for MEST_g vs. 38.38% dense baseline on ImageNet-C), so the absolute context is available.

- **Formatting/presentation nitpicks from the harsh critic**: Various presentation issues (e.g., Figure descriptions, notation) are removed as trivial formatting artifacts.

## Novel Insights

The spectral analysis in Section 5.2 reveals an asymmetric frequency sensitivity in DST models—they match dense training in low-frequency reliance but are less sensitive to high-frequency attenuation. This is a genuinely novel diagnostic observation that connects DST's implicit regularization to the well-established literature on frequency aliasing in downsampling operations (Li et al., 2021; Grabinski et al., 2022), providing a concrete mechanistic hypothesis for DST's noise-corruption advantage. The key insight is that the *type* of regularization matters: DST's hard sparsity selectively suppresses high-frequency channels, which is precisely where noise-type corruptions manifest, making this a more targeted regularizer than generic L2 decay.

## Suggestions

- Re-title and re-frame the paper to accurately reflect the contribution: "DST provides implicit regularization benefits for corruption robustness comparable to standard techniques" rather than "DST vs. Dense Training: The Unexpected Winner."
- Report clean accuracy for every model configuration—this is a critical missing control that affects interpretation of all results.
- Run at least 3 seeds for ImageNet-scale experiments and report variance; if the sub-1pp margins are not significant, tone down the corresponding claims accordingly.
- Show results across all tested sparsity levels in Table 2 (not just the best one) to provide an honest summary.

## Score and Decision

**Calibration anchors compared:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| G-Init for GNN oversmoothing | /7JigPd5Pm5.md | 2.5 | Much weaker: trivial theory + wrong experiments. Our paper is substantially stronger. |
| LLM domain knowledge | /9tMzqRaEL3.md | 4.5 | Similar: overclaimed empirical findings, missing variance/statistical rigor. Our paper has broader experimental scope but similar evidential issues. |
| Confidence-robustness correlation | /0IqriWHWYy.md | 4.25 | Similar: correlational finding overclaimed as causal, limited technical novelty. Our paper is somewhat stronger due to spectral analysis. |
| CSI for point cloud robustness | /S1qSHSFOew.md | 5.5 | Similar: overclaimed "significant outperformance" when baselines were stronger. Our paper avoids that particular pitfall. |
| Corruption robustness with learned Bregman | /7GCRhebJEr.md | 5.0 | Similar domain and scope. |
| Tuning Frequency Bias of SSMs | /wkHcXDv7cv.md | 7.5 | Much stronger: theoretical + empirical frequency analysis. Our paper's spectral analysis is correlational, lacks theory. |

The paper sits in the 4–5 range based on calibration. It makes a genuine and previously unexplored empirical observation with commendable experimental breadth and a promising spectral analysis. However, the key evidence on the most important benchmarks (ImageNet) has sub-1pp margins without variance reporting, clean accuracy is entirely absent, and the "consistent outperformance" claim is overstated. These are not fatal issues—the directional finding appears real (especially for noise-type corruptions at high severity and for video data)—but they significantly undermine the strength of the central claim. Similar overclaimed empirical papers without statistical rigor scored 4.25–5.0 in the calibration set.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>