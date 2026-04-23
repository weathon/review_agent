Now I have enough information to write the final review. Let me synthesize everything.

## Summary

This paper proposes the Dynamic Sparsity Corruption Robustness (DSCR) Hypothesis: that Dynamic Sparse Training (DST) at low sparsity levels (10–50%) can consistently outperform Dense Training in corruption robustness. The authors validate this claim across 9 scenarios spanning CNNs (VGG, ResNet, EfficientNet) and transformers (DeiT) on image (CIFAR, TinyImageNet, ImageNet) and video (UCF101) corruption benchmarks, using multiple DST algorithms (SET, RigL, MEST, GraNet). They further provide spatial and spectral analyses suggesting DST acts as an implicit regularizer that reduces attention to high-frequency information, explaining its particular advantage against noise-type corruptions.

## Strengths

- **Novel and well-motivated research question**: The paper identifies a genuinely under-explored phenomenon—DST's potential robustness benefits beyond efficiency—and formulates it as a clear hypothesis (DSCR). This opens a new perspective on sparse training research (Section 1, hypothesis stated on p. 4).

- **Broad empirical scope**: The evaluation spans 9 distinct scenarios including 4 DST algorithm families, 5 architectures (CNN and transformer), 2 data modalities (image and video), and multiple corruption benchmarks (C, C̄, 3DCC). This breadth makes the core finding difficult to dismiss as architecture- or domain-specific (Table 2, Figures 2–4, Table 1).

- **Per-corruption and per-severity granularity revealing a consistent pattern**: Figure 3 shows that DST achieves up to ~25% relative improvement specifically on high-severity, high-frequency noise corruptions (impulse noise, Gaussian noise, shot noise), while showing smaller gains on low-frequency corruptions. This fine-grained pattern is coherent with the proposed frequency-bias mechanism.

- **Spectral analysis provides a plausible mechanistic explanation**: The frequency attenuation experiments (Figure 7, Observations 1 & 2) demonstrate that DST models are consistently less sensitive to high-frequency attenuation than dense models, while equally reliant on low-frequency information. This creates a coherent chain from training method → frequency bias → corruption-type sensitivity → robustness gain, even if causality is not definitively established.

- **Results extend beyond CNNs and synthetic corruptions**: Table 1 and Figure 4 confirm that DST's robustness advantage holds for transformer architectures (DeiT) and realistic corruptions (ImageNet-3DCC), as well as video classification (UCF101), ruling out narrow explanations.

## Weaknesses

### Fatal
None.

### Major

- **No parameter-count-matched dense baseline — the central comparison is confounded by model capacity**: The paper compares a sparse version of a large architecture (e.g., 50% sparse ResNet34) against the dense version of that same architecture. It never tests whether a smaller dense model with comparable parameter count (e.g., dense ResNet18) also shows improved robustness. If reduced overfitting from having fewer parameters is the primary driver, the paper's claim that *dynamic sparsity* is the causal mechanism is unsubstantiated. The paper itself notes that "larger over-parameterized models do not necessarily gain more robustness" (Section 1) yet does not control for this obvious confound. This gap directly undermines the core DSCR claim. A parameter-matched dense baseline and/or a static sparse baseline would significantly strengthen the paper.

- **No static sparse training baseline — cannot disentangle sparsity from dynamic topology updates**: The paper cites Timpl et al. (2022) showing static sparse networks can improve robustness (Section 1), yet includes no static sparse baseline in its experiments. Without comparing DST to a fixed-mask sparse model at the same sparsity level, it is impossible to determine whether the robustness benefit comes from the *dynamic* remove-regrow process (the paper's claimed mechanism) or simply from *sparsity itself* (reduced capacity acting as regularizer). This is a critical omission for a paper whose central claim is about the benefits of dynamic sparsity specifically.

- **Marginal ImageNet improvements with no statistical significance tests**: On the most meaningful benchmark scale (ImageNet), the improvements are very small: ImageNet-C +0.32% (38.38→38.70), ImageNet-C̄ +1.06% (40.38→41.44), ImageNet-3DCC +0.46% (43.62→44.08). No error bars, standard deviations, multiple random seeds, or significance tests are reported anywhere in the paper. For a paper claiming DST "consistently outperforms" dense training, improvements of 0.3% on a single run could fall within run-to-run variance. The stronger CIFAR100-C results (51.6→55.6, +4.0%) drive much of the narrative, but these are at a much smaller scale. The DeiT results on ImageNet-C also show marginal improvements (~0.05–0.47% over a baseline that differs between Table 2 and Figure 4 caption, raising further concerns about precision).

### Minor

- **Overclaimed "consistently outperforms" and "9 wins, 0 losses" framing**: The "consistently outperforms" language in the abstract and throughout the paper overstates the evidence given the marginal ImageNet results. Table 2 cherry-picks one sparsity level per experiment (sparsity 0.4 for CIFAR where improvements are strongest; sparsity 0.1 for ImageNet), and on CIFAR10-C at sparsity 0.3, several DST methods appear to underperform the dense baseline (Figure 2a). The paper acknowledges this ("at certain sparsity ratios, such as 0.4") but the headline claim does not reflect this nuance.

- **Proposed causal mechanism (frequency bias → robustness) is correlative, not causative**: The spectral analysis (Section 5.2) shows that DST models are less affected by high-frequency attenuation, which is consistent with the frequency-bias story but does not establish causation. The paper does not show that (a) dense models forced to have similar frequency sensitivity gain comparable robustness, or (b) modifying DST's regrowth criterion to preserve high-frequency connections reduces robustness. While this is a high bar for an empirical paper, the paper's language ("dynamic sparsity process... acts as an implicit regularization mechanism, enabling the model to automatically focus on more important features") goes beyond what the evidence supports.

- **"MixNets" column in Table 2 is undefined**: The table includes a "MixNets" column with values that partially differ from the "Reg." column, but this term is never defined in the main text, leaving its purpose and meaning unclear.

### Trivial
None.

## Nice-to-Haves

- Include a parameter-count-matched dense baseline (e.g., compare 50% sparse ResNet34 to a dense ResNet18 with similar parameter count) and a static sparse baseline (fixed mask at the same sparsity) to isolate the contribution of dynamic sparsity from reduced capacity.
- Report multiple runs with standard deviations for ImageNet experiments, particularly given the small effect sizes.
- Conduct an intervention experiment in the spectral domain (e.g., modify DST's regrowth criterion to preserve high-frequency connections and check if robustness degrades) to move from correlation to causation.
- Test whether combining DST with established robustness methods (AugMix, adversarial training) yields compounding benefits, as the paper briefly discusses complementarity with Mixup in the appendix but not AugMix.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"Internally contradictory framing about efficiency"**: The harsh critic claims the paper contradicts itself by saying "efficiency is not considered as a main objective" while also claiming "at least 40% of computational and memory costs can be saved." These are not contradictory — the paper argues efficiency is a *benefit* of DST at these sparsity levels, but the *focus* of the paper is robustness, not efficiency. At sparsity 10–50%, efficiency gains exist but are not the primary claim.

- **"Unifying all DST methods under magnitude-based pruning departs from original algorithms"**: The critic argues that using a unified magnitude-based removal approach across different DST methods (rather than their original pruning criteria) raises questions. However, the paper explicitly explains this choice in Section 3.2 ("for simplicity in analysis, we use a unified magnitude-based removal approach") and tests both random and gradient-based regrowth strategies across methods. This is a reasonable experimental choice for a comparative study.

- **"CIFAR10-C at sparsity 0.3: some DST methods underperform dense baseline"**: While technically true, the paper explicitly acknowledges this in Section 4.1 ("For CIFAR10-C, at certain sparsity ratios, such as 0.4, the overall DST methods achieve decent robust performance"). The fact that DST doesn't win at every sparsity level on every benchmark is actually informative — it's not a weakness but a nuance the paper already addresses.

- **"Relative robustness gain metric can be misleading when baseline accuracy is low"**: This is a valid observation about relative metrics in general, but the paper also reports absolute robustness accuracy values in its tables, so readers can assess both metrics.

## Novel Insights

The paper's most interesting insight is the convergence of three independent signals: (1) DST's robustness advantage is strongest for high-frequency noise corruptions at high severity, (2) DST models are less affected by high-frequency attenuation in spectral analysis, and (3) DST's spatial weight patterns show concentrated pruning on specific channels. This three-way convergence (corruption-type specificity, spectral sensitivity, spatial pruning structure) creates a coherent mechanistic narrative even though causation is not formally established. However, the confounded experimental design (no parameter-matched or static sparse baselines) means this narrative could equally be explained by a simpler "reduced capacity reduces overfitting" story, which is a significant gap.

## Suggestions

- Add two critical baselines: (1) a smaller dense model with comparable parameter count to the sparse model (e.g., dense ResNet18 alongside 50% sparse ResNet34), and (2) a static sparse model (fixed random or Erdős–Rényi mask at the same sparsity). These would directly test whether the robustness benefit comes from dynamic sparsity, static sparsity, or reduced capacity.
- Report mean ± std across at least 3 random seeds for all ImageNet-scale experiments. Given improvements of +0.32% on ImageNet-C, establishing reproducibility is essential.
- Moderate the "consistently outperforms" claim to reflect the actual evidence: DST outperforms dense training at specific sparsity levels, with larger margins on smaller-scale benchmarks and marginal gains on ImageNet.

## Score and Decision

**Calibration anchors:**
- **High (>7):** DAREx (7.5, Accept Spotlight) — strong pruning paper with substantial improvements; this paper is weaker due to confounded comparisons and marginal ImageNet results. Implicit frequency bias papers (7.0–7.75) — deeper theoretical grounding.
- **Medium (4–6):** AdaSAP (6.0, Accept Poster) — similar topic (sparse+robust) with +6% ImageNet-C improvements and novel method; this paper has broader scope but weaker experimental design. GEARnn (5.8, Reject) — similar topic, limited novelty; this paper has a more novel question. S-TLLR (5.0, Reject) — confounded comparisons; this paper has similar methodological issues. Sparse Model Soups (5.75, Accept Poster) — clearer contribution.
- **Low (<3):** CAReNet (2.5, Reject) — overclaimed, no error bars, marginal improvements; this paper is clearly better with broader scope and more rigorous analysis. CPLLM (2.0) — marginal improvements over trivial baselines; this paper has more substance.

The paper's novel question and broad experimental scope place it above the low anchors. However, the confounded experimental design (no parameter-matched or static sparse baselines) is a significant methodological gap that directly undermines the paper's central claim that *dynamic sparsity* drives robustness gains. The marginal ImageNet improvements without statistical significance testing further weaken the evidence. The paper is roughly comparable to S-TLLR (5.0) in terms of methodological issues with confounded comparisons, but has a more novel question and broader scope.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>