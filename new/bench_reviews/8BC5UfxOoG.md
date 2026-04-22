Now I have enough information to write the final review. Let me carefully evaluate each harsh critic point against the paper.

**Harsh Critic Point 1: Maximum bias vs. mean bias = variance, not amplification.**

This is a substantive concern. The paper explicitly states (Section 3.3, lines 110): "Although example selections reduce the mean bias value, the LLMs tested exhibit varying degrees of increase in the maximum gender or race bias value with random-based example selection for ICL." The claim that this constitutes "amplification" is indeed potentially misleading. The simultaneous decrease in mean bias and increase in maximum bias is exactly what you'd expect from increased variance. However, the paper does emphasize "maximum" specifically—the concern about worst-case fairness is legitimate. The framing as "amplification" without acknowledging the variance explanation is a real weakness. This is a **major** issue, not fatal—the finding is real (maximum bias increases) but the "amplification" interpretation is debatable.

**Harsh Critic Point 2: Spurious correlation attribution is not causally established.**

The paper's logic (Section 3.4, line 160): null prompts show similar fear-label tendencies for male/female → therefore "example selection contributes to spurious correlations." This is a false dichotomy: the paper considers only two factors (parameters and prompt), and eliminates parameters, so concludes it must be the prompt. But the prompt includes both the content/structure of the ICL format and the specific examples selected. The paper concludes "example selection contributes" but doesn't isolate example selection from other prompt effects. This is a **major** concern—the causal claim is overstated.

**Harsh Critic Point 3: ReBE worsens bias for Perplexity baseline.**

Table 3 confirms this: for Perplexity, GPT-J-6B AvgGF increases (+0.024), MaxTG increases (+0.060), MaxFG increases (+0.079); OPT-13B AvgGF increases (+0.009), MaxTG increases (+0.019), MaxFG increases (+0.096). The paper says "the average gender bias of most LLMs decreases" (Section 5.1)—this is technically true but the Perplexity case is a meaningful failure. This is a **minor-to-major** issue.

Let me assess the remaining harsh critic points:

- Finding ❶ being "trivially true": While intuitive, it is still worth empirically demonstrating. Not a major weakness.
- GPT-3.5-Turbo paraphrasing confound: Valid concern but speculative. **Minor**.
- Number of random seeds not stated: True—the paper doesn't state the number of seeds in the main text. This matters for interpreting maximum statistics. **Minor** but worth noting.
- Confusion matrices only for OPT-6.7B: Valid but the paper justifies this choice (it has the biggest fluctuation). **Minor**.
- A(i) definition excludes same-label/same-demographic: Let me check the original equation more carefully. Looking at Equation 4: 𝒫(i) = {j ∈ ℐ : y_j = y_i, s_j ≠ s_i} is the positives, and 𝒜(i) = {k ∈ ℐ : y_k ≠ y_i, s_k = s_i} is the contrastive set. So the denominator only includes different-label/same-demographic examples. This means same-label/same-demographic are excluded from both numerator and denominator. This is a design choice that seems intentional—aligning same-label/different-demographic as positives and separating different-label/same-demographic as negatives. It's not necessarily "degenerate." The harsh critic's complaint about "degenerate training dynamics" is overstated. **Minor** at best.
- L_bias alone drops accuracy to 0.26: True but this is expected—the ablation shows exactly what it should, that both losses are needed. Not a weakness of the paper.
- ReBE's MaxFG Max worse than L_acc alone (0.284 vs 0.250): The values in Table 4 show ReBE's MaxFG Max is 0.284 and L_acc alone is 0.250. This is a genuine concern—the bias-aware embedding increases certain bias measures. **Minor**.
- Debiasing baselines too thin: The paper justifies excluding FCG and compares with counterfactual and gender-balanced context. Only two baselines is indeed thin. **Minor-to-major**.
- Parameter analysis limited: True but it's an ablation/analysis section, not the core contribution. **Minor**.

Now let me consider the strength finder's claims:

1. "Novel finding that example selection amplifies maximum bias while reducing mean bias" - This is the core finding but the "amplification" framing is questionable. I'll keep this but moderate it.
2. "Causal attribution of spurious correlations" - The causal attribution is actually weak as discussed. I'll remove this as a strength.
3. "Effectiveness of ReBE" - Partially valid but has the Perplexity failure. I'll keep but moderate.
4. "EEC-paraphrase dataset" - Valid contribution. Keep.
5-8. Supporting strengths: Generally valid. Keep.
9. "ReBE compatible with existing methods" - Partially valid but Perplexity+ReBE worsens bias. Moderate.

Now let me write the final review.</think>Now I have reviewed the paper thoroughly and checked all claims against it. Let me compile the final review.

## Summary

This paper investigates whether example selection for in-context learning (ICL) amplifies social biases in LLMs. It constructs a paraphrased version of the Equity Evaluation Corpus (EEC-paraphrase) and evaluates eight LLMs across four example selection methods, finding that: (❶) high accuracy does not guarantee low bias, (❷) example selection increases maximum bias values compared to zero-shot, and (❸) example selection contributes to spurious correlations (e.g., male pronouns → sadness misclassified as fear). To mitigate this, the paper proposes ReBE, a prompt-tuning-based method that uses a bias-contrastive loss to learn debiased virtual token embeddings while preserving accuracy.

## Strengths

- **Important and underexplored research question**: The interaction between ICL example selection and social bias is a genuinely important concern that prior example selection work has largely ignored. Identifying that ICL prompts serve as a channel through which biases can manifest is a valuable observation (Section 1, Section 3).
- **Comprehensive empirical scope**: Evaluation across eight LLMs (LLaMA-2-7/13/70B, OPT-6.7/13/30B, GPT-J-6B, GPT-neo-2.7B) and four example selection methods (Random, Similarity, Perplexity, DPP) provides useful breadth for the bias analysis (Table 2, Figure 2).
- **Useful observation about maximum vs. mean bias divergence**: The fact that ICL reduces mean bias but increases maximum bias (Figure 2) raises a legitimate concern about worst-case fairness outcomes, which is practically important even if the underlying mechanism is debated.
- **Concrete spurious correlation identification**: The confusion matrix analysis (Figure 3) effectively reveals a specific, interpretable spurious correlation—male-pronoun sadness sentences misclassified as fear at 0.54 vs. 0.08 for female—and connects this to MaxTG/MaxFG metrics (Section 3.4).
- **Useful construction of EEC-paraphrase dataset**: Paraphrasing EEC template sentences via GPT-3.5-Turbo to create more natural sentence structures is a reasonable contribution that enables more realistic bias evaluation (Section 3.1).
- **Well-designed ablation study**: Table 4 cleanly demonstrates that L_bias drives bias reduction while L_acc preserves accuracy, confirming the complementary roles of the two loss components.

## Weaknesses

### Fatal
None.

### Major

- **The "amplification" framing is misleading—the observed increase in maximum bias is better explained by increased variance than by systematic amplification.** Figure 2 simultaneously shows that mean bias *decreases* under ICL while maximum bias *increases* across most models. This pattern is exactly what is expected when ICL introduces an additional source of variance (the randomly selected prompt), which naturally widens the distribution—raising maxima and lowering minima—without shifting the underlying distribution. The paper's central claim ❷ ("example selection for ICL amplifies the biases of LLMs") would require evidence that the *entire* bias distribution shifts upward (e.g., via quantile comparisons or distributional tests), not merely that its extreme values increase. The observation that maximum bias increases is worth reporting as a worst-case fairness concern, but the "amplification" terminology implies a systematic upward shift that the evidence does not support. This is especially important because the paper itself uses the word "amplifies" in its title and first finding, making this the paper's core claim.

- **The causal attribution of spurious correlations to "example selection" is not established.** Section 3.4 concludes that "example selection contributes to spurious correlations" by showing that null (content-free) prompts produce similar fear-label tendencies for male and female (Figure 4), thereby excluding LLM parameters as the sole cause. However, this is a false dichotomy: the difference between ICL and null-prompt behavior could be caused by any aspect of the ICL prompt—including the general format/structure of in-context examples, the emotional content of examples, or the interaction between test inputs and the prompt template—rather than specifically by the *selection* of examples. To establish that example *selection* (as opposed to ICL itself) causes spurious correlations, one would need to show that the spurious correlation varies systematically with the *choice* of examples (e.g., comparing demographic-balanced vs. imbalanced prompts). No such analysis is presented. Claim ❸ is therefore overstated.

### Minor

- **ReBE worsens bias for the Perplexity baseline in several metrics.** Table 3 shows that for Perplexity+ReBE, all average bias metrics increase for both GPT-J-6B (AvgGF: +0.024, MaxTG: +0.060, MaxFG: +0.079) and OPT-13B (AvgGF: +0.009, MaxTG: +0.019, MaxFG: +0.096). The paper states "the average gender bias of most LLMs decreases" (Section 5.1), which is technically accurate but the Perplexity failure is a meaningful limitation that deserves explicit discussion. For a method claiming compatibility with existing example selection methods, one failure out of four baselines is non-trivial.

- **The number of random seeds is not reported in the main text**, making it difficult to interpret the maximum-bias comparisons. If zero-shot and ICL conditions use different numbers of seeds, the maximum comparison is biased since more seeds yield higher maxima by chance. This information is critical for the paper's central claim but is deferred to the appendix (which is stripped from this version). Section 3.3 mentions "various random seeds" without specifying the count.

- **Debiasing baselines are thin**: Only counterfactual context and gender-balanced context augmentation are compared against ReBE (Table 5). While the paper justifies excluding FCG (requires explicit feature vectors), other debiasing approaches (e.g., self-debiasing, representation-based interventions, instruction-based debiasing) could potentially be adapted. Two baselines provide limited context for assessing ReBE's effectiveness.

- **The ablation reveals that ReBE's MaxFG Max (0.284) is worse than L_acc alone (0.250)** for GPT-J-6B (Table 4). This suggests the bias-aware embedding can actually increase certain bias measures compared to vanilla accuracy-driven training—a limitation that the paper does not discuss.

- **GPT-3.5-Turbo paraphrasing could introduce its own biases into the EEC-paraphrase dataset** (Section 3.1). The paraphrasing model may systematically produce more "fear"-associated language for sentences with certain demographic markers. While the paper mentions quality validation in Appendix A, it is important to verify that the paraphrasing preserves demographic neutrality in emotional content, not just fluency.

### Trivial
None.

## Nice-to-Haves

- **Distributional comparison of zero-shot vs. ICL bias**: Plot full distributions (overlaid histograms or CDFs) of each bias metric across random seeds, or conduct quantile-based comparisons (e.g., 90th percentile) and distributional tests (KS test). This would resolve the variance-vs-amplification question and strengthen or revise the central claim.
- **Vary prompt demographic composition to isolate example selection effects**: Compare prompts with balanced vs. imbalanced demographic compositions to test whether the choice of examples systematically affects spurious correlations, providing the missing causal evidence for claim ❸.
- **Evaluate on additional tasks beyond sentiment classification** (e.g., NLI, QA) where demographic attributes are less explicitly cued, to test generalizability of findings.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Finding ❶ is trivially true"**: While the observation that "high accuracy does not mean low bias" may seem intuitive, it is still useful to empirically demonstrate it in the ICL context, and prior work on example selection has implicitly assumed otherwise. Downgraded from a weakness to a minor observation.
- **"A(i) definition creates degenerate training dynamics"**: The exclusion of same-label/same-demographic samples from the denominator of the contrastive loss (Equation 4, Section 4.3) is a deliberate design choice that focuses the contrastive signal on cross-demographic alignment and intra-demographic separation. Claiming this creates "degenerate training dynamics" is speculative and not supported by the actual training results.
- **"L_bias alone drops accuracy to 0.26—shows the contrastive loss is too aggressive"**: This is exactly what the ablation is designed to show—that both loss components are necessary. It is not a weakness of the paper but a confirmation of the design rationale.
- **"Parameter analysis is preliminary"**: The robustness claims about example order (Table 4, Section 5.4) are based on limited evidence, but this is an analysis section, not the core contribution. Not a significant weakness.
- **"Semantic similarity evaluation procedure is underspecified"** (Section 3.2): The evaluation of LLM outputs via semantic similarity is a common approach in LLM evaluation. While a more detailed description would be helpful, this is a minor clarity concern.
- **Strength finder's claim about "causal attribution of spurious correlations"**: This was listed as a strength but conflicts with the verified major weakness that the causal attribution is not established. Removed from strengths.

## Novel Insights

The paper identifies a genuine tension in ICL fairness: example selection methods that improve average accuracy and even improve average fairness can simultaneously worsen worst-case fairness outcomes. This "variance-amplification" effect, where the distribution of bias across random prompt seeds widens rather than shifts, is a practically important observation—even if the paper misframes it as systematic "amplification." The insight that mean and maximum bias can diverge under ICL raises questions about which fairness guarantee matters for deployment: average-case fairness or worst-case fairness.

## Suggestions

- Reframe the central claim from "amplification" to "increased variance in bias" or "worse worst-case bias," and explicitly acknowledge that mean bias decreases. This would make the finding more accurate and arguably more actionable (worst-case fairness is a real concern regardless of whether it's a distributional shift or variance increase).
- Add an explicit analysis varying the demographic composition of in-context examples to test whether spurious correlations change systematically—this would provide the missing causal evidence for whether example *selection* vs. ICL *format* is responsible.
- Discuss the Perplexity+ReBE failure case transparently and investigate why ReBE is incompatible with this particular selection method.

## Calibration

**Anchors used:**

| Paper | Avg Score | Comparison |
|-------|-----------|-----------|
| `/home/wg25r/review_agent/human_reviews/oZtt0pRnOl.md` | 8.0 (Accept poster) | Privacy-preserving ICL with demonstrations. Stronger methodological contribution with clear experimental design. Our paper has weaker causal claims. |
| `/home/wg25r/review_agent/human_reviews/TlAdgeoDTo.md` | 7.25 (Accept spotlight) | First-person fairness in chatbots with large-scale evaluation and human validation. Stronger empirical backing and more careful claims. Our paper has a similar fairness topic but overclaims. |
| `/home/wg25r/review_agent/human_reviews/7GKbQ1WT1C.md` | 5.25 (Accept poster) | Causality-guided debiasing via prompting. Similar topic (LLM debiasing, prompting-based solution), similar level of empirical evidence. Our paper is comparable but has more significant framing issues with "amplification." |
| `/home/wg25r/review_agent/human_reviews/FEDnzAhIT4.md` | 5.75 (Reject) | Test-time fairness via out-of-context prompting. Similar fairness/debiasing in LLMs, also with modest baselines. Rejected despite reasonable contributions. |
| `/home/wg25r/review_agent/human_reviews/KX5hd1RhYP.md` | 4.67 (Reject) | Shows ACR metric is misleading (max-vs-mean statistics concern). Analogous to our paper's max-bias-vs-mean-bias issue. This paper was more rigorous in its critique but still rejected. |
| `/home/wg25r/review_agent/human_reviews/M7CblLwJB8.md` | 2.6 (Reject) | AutoCustomization: overclaimed "super-human" metric, weak evidence, poor scientific contribution. Our paper's overclaiming is less severe but follows a similar pattern. |

The paper is in a similar niche to the 5.25–5.75 debiasing papers but has a more significant central claim problem (the "amplification" framing) that undermines its main finding. It's not as weak as the 2.6 paper (it has real empirical content and a reasonable method), but its core claim is not well-supported. The paper sits below the borderline debiasing papers because of the framing issue and incomplete causal analysis, but above the clearly weak papers.

## Score and Decision

**Originality**: Moderate. The question about ICL example selection and bias is novel, but the methodology (contrastive learning for debiasing, prompt tuning) is adapted from existing work.

**Importance of research question**: High. Fairness in ICL is practically important and underexplored.

**Claims well-supported**: Weak. The central "amplification" claim is misleading, and the causal attribution to example selection is not established.

**Soundness of experiments**: Moderate. Eight models and four selection methods provide breadth, but distributional analyses are missing and the Perplexity failure case is not addressed.

**Clarity**: Moderate. Claims are stated with more certainty than the evidence supports.

**Value to community**: Moderate. The empirical observations about max/mean bias divergence and spurious correlations are useful, even if the interpretation needs revision.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>