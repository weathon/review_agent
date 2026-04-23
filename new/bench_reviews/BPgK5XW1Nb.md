Now I have all the information I need. Let me write the final review.

## Summary

The paper proposes SPA (Spread Preference Annotation), a framework for aligning LLMs with minimal human preference data by iteratively generating self-labeled preference data using the model's own implicit reward derived from the DPO formulation (Eq. 7), combined with a noise-aware learning algorithm using confidence-based label smoothing and de-coupled noise detection via logit extrapolation. SPA achieves strong results on AlpacaEval 2.0 and MT-Bench, outperforming standard DPO and iterative DPO with external reward models while using only 3.3% of the UltraFeedback ground-truth labels.

## Strengths

- **Principled derivation of preference labels from DPO implicit reward (Eq. 7):** Rather than relying on external reward models or large teacher LLMs, SPA extracts preference judgments from β log π_{i-1}/π_init, directly leveraging the reward implicit in the DPO formulation. This eliminates dependence on external models while being empirically more effective — SPA achieves 15.39% LC win rate vs. 11.87% for Iterative DPO with PairRM (Table 2).

- **Strong empirical results across multiple model families and settings:** Table 5 demonstrates consistent improvements across Phi-2-2.7B, LLaMA-3-8B, and Phi-3-14B. Table 3 shows robustness across seed data sizes (0.8%–10%). Table 1 shows SPA with 3.3% labels outperforms Zephyr-7b-β trained on 100% of UltraFeedback on both LC win rate (15.39% vs. 11.75%) and raw win rate (21.13% vs. 10.03%).

- **Compelling distribution-shift explanation for superiority over external reward models:** The paper provides a well-motivated argument (Section 5.2) that as iterative training proceeds, generated data distribution shifts away from the fixed external reward model's training distribution, while SPA's intrinsic reward model updates each iteration. This is clearly evidenced in Figure 3, where the gap between SPA and PairRM widens substantially from iteration 1 to 2.

- **De-coupled noise detection at zero additional computational cost:** The logit extrapolation (Eq. 12) requires no extra forward passes since h_θ and h_ref are already computed for the DPO objective. The ablation in Table 6 confirms DND provides a meaningful boost (14.70% → 15.39% LC win rate; 19.94% → 21.13% win rate).

- **Comprehensive ablation study and reference model analysis:** Table 6 cleanly isolates the contributions of data expansion, self-refinement, and DND. Table 7 provides useful empirical analysis showing that using the SFT model as reference for judgment outperforms using the previous-iteration model, which supports the design choice.

## Weaknesses

### Fatal
None.

### Major

- **No validation that self-annotated preference labels agree with human/gold labels on new prompts.** The paper's central claim is that SPA "spreads human preference" from seed data to new prompts, yet there is no experiment measuring agreement between the model's self-generated preference labels (Eq. 7) and actual human or gold-standard preferences. UltraFeedback provides ground-truth preference labels for the prompts in X_i; the paper could straightforwardly report label accuracy of Eq. 7 judgments against these labels. Without this, the core "spreading human preference" narrative is unsupported — the method may simply be reinforcing model likelihoods rather than propagating genuine human preferences. This is the single most important missing experiment: it would directly establish or falsify the core claim.

- **Confounded comparison in Table 2 makes it difficult to isolate the judgment method's contribution.** The baselines (Iterative DPO with PairRM or LLM-as-judge) do not include the self-refinement or de-coupled noise detection components that SPA uses, as the paper acknowledges (Section 5.1). The ablation in Table 6 partially addresses this: data expansion with direct preference judgment alone (no SR/DND) achieves 14.41% LC win rate, still well above PairRM's 11.87%, showing the judgment method contributes the majority of the improvement. However, the headline Table 2 comparison (15.39% vs. 11.87%) conflates the judgment method with the noise-aware training. The fair comparison for isolating the judgment method would be 14.41% vs. 11.87%, and the paper should present this clearly rather than only the confounded comparison.

### Minor

- **Headline results emphasize the length-biased non-LC win rate.** The abstract claims "16.4% increase in AlpacaEval 2.0 win rate" using the non-LC metric, which is known to strongly correlate with response length. The more reliable LC win rate improves from 7.58% to 15.39% — a 7.81 percentage point gain, which is meaningful but less than half the claimed 16.4% figure. The paper does report both metrics, but the headline framing is misleading. The paper also does not report average response lengths for all methods, which is essential context for interpreting non-LC numbers.

- **Reference model inconsistency between judgment and training is not discussed.** SPA uses π_init (the SFT model) as the reference for preference judgment (Eq. 7, confirmed in implementation details), but uses π_{t-1} as the reference for DPO training (Section 4.2, Algorithm 1). Table 7 empirically justifies using π_init for judgment, and using π_{t-1} for DPO training is standard iterative practice, but this design choice is never explicitly explained or justified in the text, leaving readers to discover it.

- **High variance of SPA across seed data samplings.** Table 4 shows SPA's LC win rate variance is 2.10 across seeds vs. DPO's 0.16 — a 13× difference. While the paper notes the lowest confidence interval (13.36%) still exceeds the strongest baseline (11.98%), this high variance raises practical reliability concerns. The paper could strengthen this by analyzing what causes the variance (e.g., are certain seed data compositions particularly effective or ineffective?).

- **Self-refinement alone contributes negligibly; DND is the main driver.** Table 6 shows SR without DND provides only 0.29 pp LC improvement (14.41% → 14.70%). Nearly all the self-refinement benefit comes from DND. This somewhat undermines the paper's framing of "self-refinement" as a co-equal technical contribution alongside the direct preference judgment method.

### Trivial
- None notable.

## Nice-to-Haves

- **Measure label accuracy of self-judgments against gold labels** — this would directly validate the "spreading human preference" claim and is the single most impactful addition the authors could make.
- **Report average response lengths** for all methods to contextualize non-LC win rate differences.
- **Qualitative examples** showing prompts where SPA improves over DPO and cases where self-judgment may introduce errors or amplify biases.
- **Analysis of failure modes** — characterize prompts where Eq. 7 judgments disagree with gold labels to reveal when self-judgment fails.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic's claim that the mechanism is "circular in a dangerous way" and fatally self-reinforcing.** While the self-referential nature is a valid concern, the DPO framework provides principled grounding: if π_{i-1} is the optimal policy under the RLHF objective with some reward r*, then the implicit reward r = β log π_{i-1}/π_init equals r* (up to a constant). The method is leveraging a learned reward model, not purely arbitrary self-preference. The noise-aware training and the empirical improvements across iterations (Figure 3) further mitigate this concern. However, the lack of gold-label validation (kept as Major weakness above) means the degree to which self-judgments reflect human preferences remains unknown.

- **Harsh critic's claim that "SPA works without initial human preference data" is overstated because Mistral-7b-instruct is already aligned.** The paper explicitly states it uses Mistral-7b-instruct and acknowledges that it leverages "intrinsic knowledge learned about humans, during the previous training, such as pre-training or supervised instruction tuning" (Section 5.3). The claim is that SPA can improve alignment even without seed preference data, not that it starts from scratch. The presentation could be clearer but the claim is not wrong.

- **Harsh critic's concern about logit extrapolation producing incoherent distributions.** While theoretically h_g = (1+λ)h_θ − λh_ref can produce arbitrary logits, the softmax normalization makes this a theoretical rather than practical concern. The empirical results (Table 6) validate the approach. However, the theoretical justification remains thin (kept as a Minor weakness).

- **Harsh critic's concern about the performance dip from iteration 2 to 3 in Figure 3 as "an early signal of bias amplification."** The dip (~16% → ~15% LC) is small and could be due to many factors (overfitting, larger dataset size in iteration 3, hyperparameter choices). Attributing it specifically to bias amplification is speculative without further analysis.

- **Strength finder's claim about "Minimal implementation overhead" as a core strength.** This is a practical benefit but too generic to be a meaningful strength; many methods could make similar claims.

- **Strength finder's claim about "Noise-aware label smoothing" as a core strength.** While the label smoothing is reasonable, its actual contribution is small without DND (Table 6), making this an overstatement as a standalone strength.

## Novel Insights

The paper makes an interesting empirical observation that the choice of reference model for preference judgment matters significantly (Table 7): using the fixed SFT model as reference outperforms using the previous-iteration model (15.08% vs. 13.73% LC), which in turn outperforms having no reference at all (12.83%). This suggests that a stable reference point is important for consistent preference judgment across iterations, and that the implicit reward computed against a fixed reference captures more transferable preference information than one computed against a shifting reference. This finding has implications beyond SPA for iterative preference learning methods more broadly.

## Suggestions

- Run the gold-label agreement experiment: for prompts in X_i that have UltraFeedback ground-truth labels, compute the accuracy of Eq. 7's preference judgment against these labels across iterations. Report this as a key validation of the "spreading preference" claim.
- In Table 2, add a row for "SPA (DE only)" from the ablation to provide a fair comparison of judgment methods without the confounding effect of self-refinement.
- Report average response lengths for all methods in Tables 1–2 to contextualize the non-LC win rate numbers.

## Score and Decision

**Calibration anchors:**

| Paper | Score | Comparison |
|-------|-------|------------|
| Self-Alignment with Instruction Backtranslation | 8.0 | More principled self-curation with quality validation; SPA lacks equivalent validation of self-judgment quality |
| Weak-to-Strong Preference Optimization | 7.25 | Clearer theoretical foundation; SPA has broader experiments but unvalidated core claim |
| Meta-Rewarding Language Models | 5.0 | Similar self-referential concerns and missing validation; SPA is stronger (more principled approach, multiple models, ablation, noise-awareness) |
| Self-Rationalization (Self-judge) | 5.0 | Similar iterative self-improvement for judgment; SPA has more models but similar novelty concerns |
| AutoCustomization (self-referential bias) | 2.6 | Fundamentally flawed self-referential loop; SPA is much stronger with principled grounding and real empirical results |
| Enhanced Label Propagation | 5.5 | Self-training with pseudo-labels acknowledging confirmation bias; SPA is comparable in quality but in a different domain |

SPA sits above Meta-Rewarding (5.0) because it has a more principled approach (DPO implicit reward vs. LLM-as-judge), tests across 4+ model families, includes a proper ablation study, and has noise-aware training. It sits below WSPO (7.25) and Self-Alignment (8.0) because those papers validate their core claims more rigorously and have cleaner methodology. The missing gold-label validation experiment is the key gap — without it, the "spreading human preference" claim remains unsupported by the most direct possible evidence.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>