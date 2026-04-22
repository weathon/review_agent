Now I have enough calibration. Let me write the final review.

## Summary

This paper investigates why protective perturbations succeed against personalized diffusion models (PDMs), proposing a "shortcut learning" explanation: adversarial perturbations create latent-space image-prompt mismatches that cause PDMs to associate noise patterns with unique identifiers rather than identity content. Based on this analysis, the authors propose a three-stage red-teaming framework: (1) image purification via CodeFormer+Super-Resolution (CodeSR), (2) Contrastive Decoupling Learning (CDL) with noise tokens during training, and (3) negative-prompt guided sampling during inference. Experiments across 7 protection methods show significant improvements over 8 purification baselines.

## Strengths

- **Strong empirical improvements across all protection methods:** Table 1 shows the proposed method achieves positive IMS and Q scores across all 7 perturbation types, whereas all 8 baselines consistently produce negative scores on most perturbations. This is a clear and substantial improvement in the purification task. (Section 5.2, Table 1)

- **CDL is the critical component with strong ablation evidence:** Table 4 cleanly shows that removing CDL drops average performance from 0.385 to -0.094, while CDL alone (without any purification) achieves 0.099 average — better than several full baselines. This demonstrates CDL provides meaningful benefit beyond image purification. (Section 5.4, Table 4)

- **Significant efficiency advantage:** Table 2 reports 51s/sample versus 675s for IMPRESS (>10× speedup) with better LPIPS (0.271 vs 0.451), making the method practically viable. (Section 5.2, Table 2)

- **Clear visual evidence of faithfulness:** Figure 4 shows other purification methods hallucinate new identities or introduce artifacts, while the proposed method preserves facial structure — a practically important advantage. (Section 5.2, Figure 4)

## Weaknesses

### Fatal

None.

### Major

- **Comparison in Table 1 is confounded by data enhancement:** The proposed pipeline (CodeFormer+SR+CDL) produces metrics that *exceed* the clean baseline (e.g., IMS=0.14 vs. -0.13 on clean data; IMS=0.23 vs. -0.13 on FSMG). The authors acknowledge this, explaining that "image-restoration-based approaches ... preserve the image structure well" and "CDL module contributes significantly to quality improvement" (Sec 5.2). However, this means the "Ours" pipeline enhances training data beyond its original quality, so the comparison with other baselines conflates purification effectiveness with data enhancement. Other baselines don't receive comparable image enhancement, making it unclear how much of the improvement comes from purification (the stated goal) versus from simply providing higher-quality training data. A fairer comparison would apply the same CodeSR pipeline to baselines or report the effect of CodeSR+CDL on already-clean data as a separate condition. This issue doesn't invalidate the method's practical utility but makes the headline quantitative comparison partially misleading.

- **CDL's mechanism is under-analyzed — training-time vs inference-time contributions not separated:** CDL involves both (a) training-time noise token augmentation ("with XX noisy pattern") and (b) inference-time negative guidance with w^{neg}=7.5 (Eq. 6). The ablation in Table 4 only compares full CDL on/off, not these two components independently. Since classifier-free guidance with a strong negative prompt (w=7.5) is known to be a powerful generation technique, it is plausible that most of CDL's effectiveness comes from the inference-time guidance rather than the training-time "shortcut learning mitigation." Without an ablation isolating these effects (e.g., train with CDL but sample without negative guidance, or train without CDL but sample with negative guidance), the claimed "causal intervention" mechanism remains unverified. This matters because it determines whether the shortcut learning framework actually drives the method or whether the results are primarily attributable to a well-known prompting technique.

- **Adaptive attack evaluation is incomplete:** Table 3's adaptive attacks only target the purification module (CodeSR), not the full pipeline including CDL. Given that CDL is the most impactful component (Table 4), an adaptive adversary aware of CDL could craft perturbations resistant to it. After even the partial adaptive attack (ε=16/255), the best configuration drops from Avg. 0.385 to Avg. 0.034 — nearly zero effectiveness. The E[Avg.] metric using P(AA)=50% is arbitrary. The claim of "stronger robustness against adaptive perturbation" (Abstract, Contributions) is thus not fully supported by the evidence.

### Minor

- **Causal analysis is conceptual rather than rigorous:** The causal graph in Figure 2 is asserted without formal derivation (the paper references App. C.1, which is not in the body). No do-calculus, formal interventions, or counterfactual reasoning are performed. The causal framework serves as narrative motivation but doesn't substantively guide the method's design beyond inspiring the CDL heuristic. This would be fine if framed as "inspired by" causal reasoning, but calling it "causal intervention" (Sec 4.2) overclaims the analytic depth.

- **Key empirical claim for shortcut learning hypothesis lacks in-body evidence:** The claim that "random perturbation with the same strength does not affect the learning performance" (Sec 4.1) is one of two empirical pillars supporting the shortcut learning thesis, but no experiment validates this in the paper body. This is an important supporting observation that should be shown, not just stated.

- **Generalization claim lacks quantitative support:** The paper claims the framework "can generalize to other domains beyond the facial domain" (Sec 6), and mentions WikiArt and CelebA in Sec 5.1, but only provides visual demonstrations — no quantitative results on non-face domains appear in the body.

### Trivial

None.

## Nice-to-Haves

- Ablation separating CDL's training-time effect from inference-time negative guidance — this would substantially clarify the mechanism and is the most impactful missing experiment.
- Applying the same CodeSR image enhancement to baselines, or at minimum analyzing the effect of CodeSR+CDL on clean data, to disentangle purification from enhancement.
- Full adaptive attack targeting the entire pipeline (including CDL), which would strengthen the robustness claims significantly.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Negative IMS values indicating metric opacity:** The harsh critic argued that negative IMS values (even -0.13 for the clean baseline) are suspicious for cosine similarity. However, the paper defines IMS as a weighted sum of two different face extractors' cosine similarities (λ=0.7 for IP, 0.3 for VGG), and the metric range is [-1, 1]. Negative values are possible if the two face extractors produce low or negative similarities, particularly for mismatched identity features. Since all methods use the same metric, relative comparisons remain valid. The metric's absolute interpretability is a minor concern, not a substantive flaw.

- **Notation inconsistency between δ^{(j)} and x^k in Equations 3-4:** This is a minor notational clarity issue — Eq. 3 uses δ^{(j)} for perturbation iterations while Eq. 4 uses x^k for the PGD update variable. Both are standard PGD formulations; the mapping between them is implied by the context. This is a minor presentation issue, not a methodological concern.

- **Demanding proof of what V_N^* learns (feature visualization):** While it would be nice to have direct evidence that the noise token absorbs noise patterns, the paper's concept extraction visualizations (Figure 2 right panel) and the ablation in Table 4 provide indirect evidence. Demanding specific feature visualizations of V_N^* is a reasonable but not essential addition.

- **ε=16/255 for adaptive attacks being too strong:** The harsh critic suggested the perturbation budget is larger than default without justification. However, using a stronger budget for adaptive attacks is standard practice in adversarial robustness evaluations — it tests worst-case scenarios. The choice is not a flaw.

- **Missing WikiArt/CelebA quantitative results:** The paper explicitly states "we also visually demonstrate the purification ability of our approach on samples from an artwork painting dataset, WikiArt" — it scopes the main quantitative evaluation to faces and only provides visual demonstrations for other domains. While quantitative extension would strengthen the paper, the face domain is the primary domain for PDM protection, so this is a minor gap.

- **Questioning the existence/reality of any cited models or benchmarks:** Per hard rules, all cited entities are assumed real.

## Novel Insights

The most interesting finding is one the paper itself under-emphasizes: Table 4 shows that CDL *alone* (without any purification) achieves IMS=0.160 on perturbed data, compared to -0.271 with nothing. This means a purely training+inference intervention nearly recovers performance on still-perturbed data — suggesting that protective perturbations' primary mechanism is indeed the creation of spurious prompt-token associations (consistent with the shortcut learning hypothesis), and that corrective prompting can largely bypass the need for pixel-level purification. This finding partially undermines the motivation for the CodeSR component, which takes up significant methodological space but contributes less than CDL.

## Suggestions

- **Add the CDL training-time vs inference-time ablation:** Train with CDL but sample with standard (not negative) guidance; train without CDL but sample with negative guidance. This is the single most impactful experiment that would clarify the method's mechanism.
- **Apply CodeSR enhancement to baseline methods** and re-evaluate as a fairer comparison condition, or add a "CodeSR-only on clean data" row to Table 1 to quantify the enhancement effect.
- **Scale the adaptive attack to target CDL** by including the noise token and negative prompt in the adversary's knowledge when crafting perturbations.
- **Tone down the "causal intervention" framing** to "inspired by causal analysis" unless formal causal reasoning (do-calculus, counterfactual evaluation) is performed.

## Score and Decision

**Calibration anchors:**

- `/home/wg25r/review_agent/human_reviews/agHddsQhsL.md` (avg 7.5, Accept Spotlight): Directly related paper on targeted attacks for diffusion protection. Stronger methodology and cleaner claims. This paper under review has comparable empirical strength but more evaluation concerns.

- `/home/wg25r/review_agent/human_reviews/5pKLogzjQP.md` (avg 5.25, Reject): VAE purification of availability poisons with strong empirical results but adaptive attack concerns. Very comparable — both have strong results but incomplete threat model evaluation.

- `/home/wg25r/review_agent/human_reviews/ZKnbIZefER.md` (avg 4.4, Reject): Availability attacks with shortcut learning + causal analysis framing. Similar superficial causal analysis issue, weaker results overall.

- `/home/wg25r/review_agent/human_reviews/6o9QUqUq9f.md` (avg 4.67, Reject): Causal analysis of LLM tokens with superficial methodology, weak theoretical justification, missing baselines. Worse paper than the one under review.

- `/home/wg25r/review_agent/human_reviews/KoQkr9eIUG.md` (avg 2.5, Reject): Monte Carlo downsampling defense with fundamental evaluation issues and limited practical value. Much weaker paper.

The paper under review has genuinely strong empirical results with a 10× speedup, comprehensive baselines (8 methods × 7 perturbation types), and an interesting core finding (CDL alone is powerful). However, the confounded comparison in Table 1, the unseparated CDL mechanism, and the incomplete adaptive attack evaluation are significant concerns. It sits above the VAE purification paper (5.25) because of more comprehensive evaluation and practical significance, but below the targeted attacks paper (7.5) because of fairness concerns. It is notably stronger than papers with superficial causal analysis (4.4, 4.67) because the empirical contribution is substantial even if the causal framing is overclaimed.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>