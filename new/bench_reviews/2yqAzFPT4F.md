Now I have a thorough understanding of the paper and calibration anchors. Let me write the final review.

## Summary

The paper proposes Zero-Jack, a method for jailbreaking black-box multimodal large language models (MLLMs) by applying zeroth-order gradient estimation (two-point estimator) with a patch coordinate descent strategy to generate adversarial image perturbations. By dividing the image into patches (typically 32×32) and updating one patch at a time, Zero-Jack reduces the optimization dimension to ~2% of the full image, mitigating high-dimensional gradient estimation error while also eliminating the need for backpropagation, which yields substantial memory savings.

## Strengths

- **Strong empirical attack performance**: Tables 2 and 3 show Zero-Jack achieving 88–98.2% ASR across MiniGPT-4, LLaVA1.5, and INF-MLLM1 on both benchmarks, dramatically outperforming all transfer-based baselines (GCG 13%, AutoDAN 22% at best on Harmful Behaviors) and closely matching the white-box baseline (93% on MiniGPT-4).

- **Meaningful memory efficiency**: Table 1 provides concrete GPU memory comparisons—Zero-Jack enables attacking MiniGPT-4 70B on a single A100 (63G) where the white-box method runs OOM, and attacks 13B models on a single RTX 4090 (22G vs. 31G for white-box).

- **Patch coordinate descent is well-motivated and ablated**: The right subplot of Figure 4 validates the 32×32 patch choice by showing ASR degrades for both smaller patches (insufficient global information) and larger patches (increased gradient noise), demonstrating a genuine sweet spot.

- **GPT-4o direct attack as a real-world demonstration**: The attack on GPT-4o (Table 5, 69% ASR) demonstrates the practical relevance of the vulnerability, highlighting that even commercial APIs with partial logprob access are exploitable.

- **Honest limitations discussion**: Section 5 openly acknowledges that Zero-Jack requires logprob access, cannot attack web-only interfaces, and needs auxiliary prompts for GPT-4o efficiency.

## Weaknesses

### Fatal
None.

### Major

- **Misleading "black-box" framing obscures a substantial access requirement**: The paper's title ("black box MLLMs"), abstract, and introduction frame Zero-Jack as broadly applicable to "black-box" models, but the method requires access to output log probabilities (Equation 4, Section 3.3), making this a *score-based* black-box method rather than the standard (and more restrictive) *decision-based* black-box setting. While Section 5 acknowledges this limitation, the front-section framing does not, creating a misleading impression of the method's scope. Many commercial APIs (e.g., Anthropic) do not expose logprobs, and web interfaces certainly do not. The paper should consistently position itself as "score-based black-box" throughout, as the threat model difference is material to the claimed contributions—especially the claim to be "the first method that aims at jailbreaking black-box MLLMs directly."

- **Asymmetric comparison with transfer baselines conflates different threat models**: Zero-Jack has query-based access to the target model's logprobs during optimization (Eq. 4), while the text-based transfer baselines (GCG, AutoDAN) and the image-based A-Image baseline receive *zero* information from the target model. It is expected that score-based query access to the target substantially outperforms methods with no such access. The paper presents this as a competitive evaluation (Abstract: "surpasses previous transfer-based methods"), but it is fundamentally an apples-to-oranges comparison across different threat models. The more informative comparison is with the WB baseline (which Zero-Jack approximately matches), as both share the same level of information about the target. This does not invalidate the results, but the framing of "surpassing" transfer methods should be tempered by acknowledging the threat-model asymmetry. Additionally, the absence of any *score-based query* baseline (e.g., full-image zeroth-order optimization without patches) makes it impossible to assess whether the patch coordinate descent—the paper's core technical contribution—actually helps, since all other methods operate in a weaker threat model.

- **GPT-4o experiment conflates multiple factors**: Section 4.6 uses logit bias to force GPT-4o to generate target tokens during optimization and additionally uses an auxiliary text prompt from Andriushchenko et al. (2024) to "make the optimization easier." The paper acknowledges discarding the logit bias at evaluation time, but whether the optimized perturbations are effective *without* the logit bias during optimization is not independently verified. Meanwhile, the "Prompt + Original Image" baseline (18% ASR) is lower than "Text Prompt Only" (30%) in Table 5, suggesting that random images *hurt* attack success on GPT-4o, which lacks explanation. The 69% ASR cannot be cleanly attributed to Zero-Jack alone given the auxiliary prompt contribution.

### Minor

- **No query count reporting**: The paper claims "reasonable queries" (Section 3.3 contribution 2) and mentions ~$0.80 per GPT-4o sample, but never reports the total query count per successful attack. Each gradient estimate for one patch requires 2 forward passes (Eq. 6), and a 224×224 image with 32×32 patches has 49 patches per full update cycle. Without query count data, the claim of "reduced query complexity" is unsubstantiated, though the dollar cost gives some practical indication.

- **Confusing ablation labels in Figure 4 (left)**: The bar labeled "Zero-Jack" (~98% ASR) outperforms "Zero-Jack with Patch" (~45% ASR), which is counterintuitive since Zero-Jack *uses* patches by design. The paper's text says "patch updating can increase the performance," but the figure labels appear to show the opposite for the zeroth-order variants. This likely reflects a naming convention issue (perhaps "Zero-Jack" = full method with patch, and "Zero-Jack with Patch" = a variant?), but it needs clarification as the current presentation is potentially misleading.

- **Weak image-based baselines**: P-Image (unmodified images) and G-Image (Gaussian noise) are trivially non-adversarial inputs, and A-Image is the only adversarial image baseline (itself a transfer method). This limits what can be concluded about Zero-Jack's relative effectiveness among image-based attacks, though this is somewhat understandable given the paucity of existing direct image-based jailbreak methods for MLLMs.

### Trivial

- The novelty claim "the first method that aims at jailbreaking black-box MLLMs directly" (Section 1) is somewhat overstated—zeroth-order optimization for adversarial attacks is well-established in the vision ML literature, and the application to MLLMs, while new in application scope, is an anticipated extension.

## Nice-to-Haves

- A full-image zeroth-order optimization baseline (without patch coordinate descent) to directly validate the patch strategy's contribution.
- GPT-4o ablation without logit bias and without the auxiliary prompt, to isolate Zero-Jack's contribution.
- Extension to a decision-based (text-only output) black-box setting, which would substantially broaden practical impact.
- Adversarial perturbation visualizations to assess perceptibility and inform defense considerations.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Cannot be independently verified" / reproducibility concerns about cited models and tools**: Per rules, if the paper cites it, it exists. Removed.

- **Missing related works / appendix references**: The parser strips appendices; references cited in the paper are assumed to exist. Removed.

- **Formatting artifacts / typos / notation inconsistencies**: These are parser errors, not author errors. Removed.

- **White-box OOM is unsurprising without gradient checkpointing**: This is a nitpick—the paper appropriately demonstrates the memory advantage specific to not needing backpropagation, which is the point of the comparison regardless of whether other engineering techniques exist. Removed as minor nitpick.

- **Paper lacks convergence analysis or theoretical justification**: This is a standard empirical systems/safety paper; demanding theoretical proofs for an empirical contribution is scope creep. Moved to nice-to-have.

- **Harmful Behaviors dataset uses only 100 random subset out of 500**: The paper states this clearly; for a jailbreak evaluation paper, 100 samples with proper evaluation is standard and does not threaten core claims. Removed as generic weakness.

## Novel Insights

The paper reveals an interesting asymmetry in the MLLM safety landscape: the same logprob access that makes LLM APIs useful for downstream applications creates a direct attack surface for jailbreaking via the image modality, where continuous optimization circumvents the discrete token optimization challenge that limits text-based black-box attacks. The GPT-4o logit bias workaround is particularly noteworthy as a demonstration that even partial API access (top-20 logprobs + logit bias) is sufficient for an attacker to reconstruct the loss signal needed for optimization—an implication for API design that the safety community should consider.

## Suggestions

- Recast the abstract and introduction to consistently describe Zero-Jack as a "score-based black-box" method, and explicitly contrast with decision-based black-box settings. The current comparison with transfer methods should add a sentence acknowledging the threat-model asymmetry.
- Add a full-image zeroth-order optimization baseline (no patches) to Tables 2 and 3; this is the single most important missing comparison for validating the patch coordinate descent contribution.
- Report query counts (forward passes per successful attack) for at least one representative experiment, and for the GPT-4o experiment.
- Clarify the "Zero-Jack" vs. "Zero-Jack with Patch" labels in Figure 4 (left), as the current naming is contradictory to the method description.

## Evaluation Axes

**Originality**: Moderate. The core technique—zeroth-order optimization with patch coordinate descent—is a straightforward application of well-established tools (two-point estimator from Spall 1992; block coordinate descent) to the MLLM jailbreak setting. The novelty lies primarily in the application domain and the practical demonstration rather than in the technical components.

**Importance of research question**: High. Direct jailbreak attacks on MLLMs are an important and timely safety concern, and the finding that logprob API access enables effective adversarial image optimization has significant implications for API design and safety evaluation.

**Claims supported**: Partially. The central empirical claim (high ASR) is well-supported, but the framing of "surpassing transfer methods" confounds threat-model differences, and the GPT-4o result conflates multiple contributing factors. The "first method" claim is overreaching given the well-known zeroth-order optimization literature.

**Soundness of experiments**: The main results on open-source models are sound, but missing score-based baselines and the confounded GPT-4o setup are significant gaps. The ablation (Fig. 4 right) is well-designed and informative.

**Clarity**: Generally clear with well-structured method exposition, but the ablation figure labeling is confusing and the "black-box" terminology is inconsistently used.

**Value to community**: Moderate-to-high. The practical demonstration that logprob access enables effective direct jailbreaks on MLLMs—including commercial models—is valuable for the safety community, even considering the threat model limitations.

## Score and Decision

**Calibration anchors used:**

1. **High band (>7):** r42tSSCHPh — Catastrophic Jailbreak via Exploiting Generation (avg 7.0, Spotlight). Similar strong empirical results (>95% ASR), equally simple method, but that paper had a cleaner threat model and broader model coverage without confounded comparisons. Zero-Jack is weaker due to its threat-model framing issues and missing baselines.

2. **Medium band (4–6):** wNg0LibmQt — Gradient-based Jailbreak Images for Multimodal Fusion Models (avg 5.0, Reject). Very similar topical area (gradient-based image jailbreaks on MLLMs), but that paper had limited scope (only Chameleon) and questionable practical applicability. Zero-Jack is stronger: broader experiments, real commercial model attack, honest limitations. Also: Nazzz5GJ4g — Weak-to-Strong Jailbreaking (avg 5.4, Reject). Similarly strong ASR (>99%) but overclaimed black-box scope (required logprob access). Zero-Jack parallels this concern closely. uBnM3EFovQ — ReMiss (avg 5.75, Poster). Score-based jailbreak with overclaimed scope, but cleaner methodology.

3. **Low band (<3):** UWuTZYPSxJ — KDA (avg 2.5, Reject). Overclaimed properties, confounded evaluation. Zero-Jack is clearly stronger than this. 5kMwiMnUip — NEMESIS (avg 1.4, Reject). Minimal novelty, essentially rehashing known techniques.

Zero-Jack sits in the medium band: it has real empirical contributions and addresses an important problem, but the misleading black-box framing, asymmetric threat-model comparisons, and confounded GPT-4o experiment are substantive issues comparable to those in the 5.0–5.75 scored anchors. It is stronger than the Chameleon paper (5.0) due to broader evaluation and commercial model demonstration, but weaker than ReMiss (5.75) which had cleaner methodology despite similar overclaiming concerns. The ablation for patch size is a genuine contribution, but the missing score-based baseline is a notable gap.

**Score: 5.0**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>