Now I have a thorough understanding of the paper. Let me write the final consolidated review.

## Summary

The paper proposes Zer0-Jack, a method for jailbreaking black-box multimodal large language models (MLLMs) by applying zeroth-order gradient estimation to generate adversarial image perturbations. To address the high-dimensional estimation error inherent in zeroth-order optimization, the method introduces patch coordinate descent—updating one 32×32 image patch at a time instead of the full 224×224 image. Zer0-Jack achieves 90–98% attack success rates across MiniGPT-4, LLaVA1.5, and INF-MLLM1, rivaling white-box baselines and vastly outperforming transfer-based methods.

## Strengths

- **Strong empirical performance matching white-box methods**: Zer0-Jack achieves 90–98% ASR across four model configurations (Tables 2–3), consistently within 2–5% of the white-box baseline. This is a striking result showing that zeroth-order optimization can nearly match gradient-based attacks for MLLM jailbreak (Tables 2–3).

- **Patch coordinate descent effectively controls the noise-accuracy tradeoff**: The right subplot of Figure 4 shows ASR dropping sharply as patch size increases beyond 32 (from ~100% at size 32 to ~20–40% at size 256), providing clear empirical evidence that smaller patches reduce estimation error and improve ZO optimization quality (Section 4.4, Figure 4 right).

- **Concrete memory efficiency gains for large models**: Table 1 shows substantial memory savings for 13B+ models (31G→22G for MiniGPT-4 13B; 39G→25G for LLaVA1.5 13B), and critically enables attacking the 70B model (63G) where white-box attacks run OOM. This is a practical benefit directly enabled by avoiding backpropagation (Table 1).

- **Creative exploitation of logit bias for commercial MLLM attacks**: The method uses OpenAI's logit bias API feature to force target token generation and retrieve log probabilities, overcoming the top-20 logprob limitation—demonstrating the method works even under restrictive commercial API constraints (Section 4.6).

- **GPT-4 as judge evaluation**: Using GPT-4 to assess whether responses are genuinely harmful is more robust than string-matching approaches commonly used in the jailbreak literature (Section 4.1).

## Weaknesses

### Fatal
None.

### Major

- **Imprecise "black-box" framing makes the main comparison with transfer methods an apples-to-oranges evaluation.** Zer0-Jack requires access to output logits or token probabilities from the target model (Section 3.3: "we only need to get the output logits or probability"), which is a strictly stronger access level than transfer-based methods (which require zero queries to the target model). In the adversarial ML literature, this access level is typically called "score-based" rather than "black-box." The paper's headline result—that Zer0-Jack outperforms transfer methods—is trivially expected given the additional information Zer0-Jack has about the target model. The paper does acknowledge this limitation in Section 5, but the abstract, introduction, and Figure 1 all frame Zer0-Jack as a "direct black-box jailbreak" that "surpasses transfer-based methods," which overstates the significance of that comparison. A more meaningful comparison would be against other methods with comparable logit/score access (e.g., square attack, Sign-OPT, or even vanilla zeroth-order optimization on the full image), which are absent from the evaluation. That said, the comparison with white-box methods (where Zer0-Jack matches performance) is a meaningful and fair result.

- **Missing score-based adversarial optimization baselines.** No other score-based/zeroth-order optimization method is included as a baseline, despite this being a well-studied area. The paper compares against text-based and transfer-based image attacks, but not against any method that also queries the target model with logit/score access. Without this, it is impossible to determine whether Zer0-Jack's strong performance stems from the specific method (patch coordinate descent + ZO optimization) or simply from having logit access. A vanilla full-image zeroth-order optimization baseline under the same query budget would be the minimum needed to validate the patch coordinate descent contribution.

### Minor

- **The ablation study in Figure 4 (left) has confusing and potentially contradictory labels.** The left subplot shows "Zero-Jack" at ~98% ASR and "Zero-Jack with Patch" at ~45%, yet Zer0-Jack is defined in Section 3.3 and Algorithm 1 as already using patch coordinate descent. The meaning of "Zero-Jack with Patch" is unclear: if it represents the proposed method with patches, then it contradicts the main results (Tables 2–3) and the paper's claim that "Patch updating can increase the performance" (Section 4.4). If it represents a different variant, the labels are too ambiguous to interpret. Fortunately, the right subplot (patch size analysis) provides clear supporting evidence for the patch design, mitigating some of this concern—but the left subplot remains uninterpretable as presented.

- **The GPT-4o attack (Table 5) does not isolate the contribution of the image perturbation from the auxiliary text prompt.** The paper uses "a text prompt from (Andriushchenko et al., 2024) to make the optimization easier" (Section 4.6). Table 5 shows "Text Prompt Only" at 30% and "Prompt + Zer0-Jack" at 69%, but it is unclear whether "Text Prompt Only" uses the original harmful instruction or the auxiliary prompt. An ablation testing the auxiliary prompt alone (without image perturbation) vs. auxiliary prompt + perturbed image would clarify whether the image perturbation contributes meaningfully beyond the text prompt assistance.

- **Query efficiency is not reported in main experiments, limiting practical feasibility assessment.** The paper claims "reasonable queries" (Section 1) and reports ~$0.80/sample cost for GPT-4o (Section 4.6), but provides no query counts for any experiment. Each gradient estimate for a single patch requires 2 forward passes (Eq. 6), and with ~49 patches per image plus multiple iterations, this could involve hundreds or thousands of queries per sample. Without query counts, the trade-off between Zer0-Jack's higher ASR and its query cost relative to transfer methods cannot be assessed.

### Trivial
None.

## Nice-to-Haves

- Report query counts for all experiments to enable practical cost-benefit analysis relative to transfer methods.
- Show perturbed images and measure perturbation magnitude (L∞ or L2 norm) to assess the attack's detectability.
- Reframe the contribution as a "score-based" or "logit-access" attack, which would position the comparison fairly and avoid the misleading framing.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Memory savings are not a contribution because they come from not doing backpropagation"**: This mischaracterizes the paper's claim. The memory savings ARE a direct consequence of using zeroth-order optimization—which IS the paper's method. The 7B model savings (11G→10G) are indeed small, but for 13B and 70B models the savings are substantial and practically meaningful (31G→22G; OOM→63G). The critic's objection conflates "not novel in principle" with "not a contribution"—the concrete numbers and practical implications are useful regardless.

- **"GCG is a misleading baseline because it optimizes discrete tokens while Zer0-Jack optimizes continuous pixels"**: GCG is a well-known jailbreak method used as a standard baseline in the jailbreak literature. The paper includes it as a text-based baseline, not as a direct image-space competitor. This is a reasonable comparison to show how existing jailbreak methods perform on MLLMs, even if the search spaces differ.

- **"Transfer attack result (51.8% on GPT-4o) undermines the need for direct attacks"**: The paper addresses this directly in Section 4.3: "they still suffer from performance degradation, indicating the importance of attacking black-box models directly." The transferability results and direct attack results are complementary, not contradictory.

- **"The auxiliary text prompt used in GPT-4o attack may make the attack trivially easy"**: The paper acknowledges this in Section 5: "Zer0-Jack needs assistance from custom prompts, otherwise, Zer0-Jack requires far more iterations to attack GPT-4o." This is a limitation, not a fatal flaw.

- **"The two-point gradient estimator (Eq. 4) is standard"**: While the estimator itself is standard (Spall, 1992), the contribution is in the application of this estimator to MLLM jailbreak with patch coordinate descent. Standard components combined in a novel way can still constitute a contribution.

- **"Whether the logit bias changes the reported log probabilities is a critical ambiguity"**: This is speculative. The paper describes a clear procedure: use logit bias to force target token generation, retrieve the log probability, then discard the bias for final generation. The concern about "before vs. after bias application" log probabilities reflects a misunderstanding—the log probability retrieved during optimization is with bias applied, and the bias is only discarded for the final evaluation.

## Novel Insights

The paper reveals an important asymmetry in MLLM safety: adversarial perturbations in the image modality can be efficiently optimized via zeroth-order methods to near white-box levels of attack success (90–98%), while the same attack paradigm applied to text is much harder due to discrete optimization. This suggests that the image channel in MLLMs constitutes a particularly vulnerable attack surface for score-based optimization, a finding with implications for API design (e.g., whether to expose log probabilities).

## Suggestions

- Add at minimum a vanilla full-image zeroth-order optimization baseline (same ZO estimator, same query budget, no patch decomposition) to directly validate the benefit of patch coordinate descent. This is the single most important addition that would strengthen the paper.
- Clarify the labels in Figure 4 (left subplot) and explicitly define what "Zero-Jack" vs. "Zero-Jack with Patch" means in the ablation context.
- Report query counts (number of forward passes) per sample across all experiments.
- Reframe the paper as a "score-based" or "logit-access" attack to set fair expectations and enable proper comparison with methods at the same access level.

## Score and Decision

**Calibration anchors used:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| r42tSSCHPh.md (Catastrophic Jailbreak) | 7.0 | Zer0-Jack has comparable empirical strength but weaker framing clarity and less thorough evaluation |
| bhK7U37VW8.md (AutoDAN-Turbo) | 7.17 | Zer0-Jack is weaker: less novel strategy discovery, missing score-based baselines, imprecise framing |
| tIBAOcAvn4.md (Boosting Ray Search) | 7.5 | Zer0-Jack lacks theoretical analysis and has weaker baseline coverage |
| wNg0LibmQt.md (Gradient-based Jailbreak Images for Multimodal Fusion) | 5.0 | Zer0-Jack is comparable: both have limited baselines and incomplete evaluation, but Zer0-Jack has stronger empirical results |
| djcciHhCrt.md (Visual adversarial examples) | 4.25 | Zer0-Jack is stronger: more models, clearer method, higher ASR |
| efxXzrbgrX.md (Black-box TAA on SAM) | 3.75 | Zer0-Jack is clearly better: less overclaiming, more transparent about access requirements |
| UWuTZYPSxJ.md (KDA) | 2.5 | Zer0-Jack is clearly better: more honest about limitations, real empirical results |

Zer0-Jack falls between the medium-scoring papers (4.25–5.5 range) and the high-scoring papers (7+). Its core method works and achieves impressive ASR, but the imprecise "black-box" framing and missing score-based baselines prevent a clear assessment of whether the strong results are due to the method or simply the access level. Compared to wNg0LibmQt.md (5.0, similar topic area, similar evaluation issues), Zer0-Jack has stronger empirical results but a more significant framing concern. I place it at 5.0.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>