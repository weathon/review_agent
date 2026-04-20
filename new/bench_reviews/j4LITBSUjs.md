## Summary

This paper addresses hallucination in dense image captioning by two contributions: (1) **HalfScore**, a graph-based F1-style metric that extracts triplets via GPT-4o and matches them against ground-truth dense captions to measure both precision (hallucinations) and recall (completeness) at concept level; and (2) **PerturboLLaVA**, a training-time strategy that prefixes SFT data with GPT-4o-generated adversarial perturbation text designed to mislead the model's language priors, compelling it to rely more on the image. The method shows consistent improvements on hallucination benchmarks (CHAIR, HalfScore) while also improving general multimodal benchmarks—unlike several baselines that degrade general capability.

## Strengths

- **HalfScore correlates higher with human judgments than prior metrics.** Table 6 shows HalfScore precision correlates with human evaluation at 80.7% vs. MMHalBench's 71.7%, and recall at 78.1%. This provides empirical evidence that the graph-based, concept-level metric better reflects subjective caption quality than a single holistic LLM score.

- **PerturboLLaVA improves hallucination reduction while boosting general capability.** Table 3 shows PerturboLLaVA achieves +1.6 on MMBench, +0.3 on SEED, and +1.2 on CCBench over LLaVA1.5 baseline, whereas VCD and RLAIF-V both degrade on at least one general benchmark. This trade-off-free improvement is a practical advantage over decoding-based methods that trade general ability for hallucination reduction.

- **Method is complementary to decoding-based approaches.** Table 3 shows combining PerturboLLaVA with OPERA yields further gains (HalfScore 52.8 vs. 52.2 standalone; CHAIRs 33.1 vs. 36.1), demonstrating the training-time perturbation targets a different optimization direction than inference-time decoding corrections.

- **Well-designed ablation on perturbation strength.** Table 5 systematically varies perturbation intensity (Version1 → Version3) and reveals a clear precision-recall tradeoff: stronger perturbations lower hallucination scores but also shorten captions (100 → 78 tokens), providing useful practical guidance on operating points.

## Weaknesses

### Fatal
None.

### Major

- **Methodological novelty overlaps with adversarial instruction tuning.** The core idea—prepending adversarial/misleading text to training inputs and fine-tuning the model to ignore it—is conceptually very similar to Adversarial Instruction Tuning (AIT; sw6Wpx2LGr, scores 5,5,6,6, rejected), which prepended adversarial dialogues to fine-tune against dialogue hallucination. PerturboLLaVA applies this same pattern (adversarial prefix + SFT) to dense captioning, which narrows the scope of novelty. While HalfScore is an additional contribution, the methodological advance over AIT is primarily application-domain rather than conceptual.

- **GPT-4o circularity between data generation and evaluation.** The same GPT-4o model generates the perturbation training data (Sec 4.1), extract triplets, and perform graph matching for HalfScore evaluation (Sec 3). This creates a potential circular dependency: the perturbation text is designed by GPT-4o to be misleading, and the evaluation metric also relies on GPT-4o's triplet extraction and matching. If GPT-4o has a particular parsing bias, results on both sides (training and evaluation) could be systematically skewed. The paper should acknowledge this limitation and ideally include an independent validation set evaluated by a different LLM or human annotators.

- **Length-dependent confound in hallucination metrics is not controlled for.** Table 5 explicitly shows that stronger perturbation reduces caption length from 100 to 78 tokens. A shorter caption naturally reduces the opportunity to hallucinate objects, which inflates precision-heavy metrics like CHAIR and HalfScore precision. The paper reports the tradeoff (recall drops) but does not provide length-normalized hallucination rates or fixed-length evaluations to determine whether the model genuinely improves visual grounding versus merely becoming more conservative. This is central to the paper's core claim about reduced language bias.

### Minor

- **Mathematical explanation (Sec 4.2) is post-hoc and relies on unjustified independence assumptions.** Equations (6) and (9) apply Bayes' theorem to predict past/prefix tokens from future token $x_k$, and Eq (8) assumes $x_{<k}^p$ and $x_{<k}^{-p}$ are conditionally independent—an assumption unsupported for autoregressive sequences. The derivation does not connect to the actual SFT loss objective. The section is labeled "Mathematical Explanation" rather than being a core contribution, so this weakens the theoretical framing but does not invalidate the empirical results. Nonetheless, the paper should reframe this as intuition rather than formal derivation.

- **No variance reporting across random seeds.** All tables report single-point metrics. Gains of +1.6 on MMBench, +0.3 on SEED, and +0.6 on HalBench are modest and fall within typical SFT stochastic variance, making it difficult to distinguish true improvements from training noise. This is especially important given the methodological simplicity—if the method is truly robust, it should show consistent improvements across seeds.

- **"No additional computational overhead" claim is misleading.** The abstract and Section 1 claim the method incurs no additional overhead. While inference cost remains 1×, generating 160k high-quality perturbation texts with GPT-4o requires substantial API cost and latency for data preparation. The paper should clarify that the "free" overhead refers to the training/inference pipeline, not the total end-to-end workflow.

### Trivial

- Table 1 marks the "No extra data generation" column with ✗ for PerturboLLaVA, contradicting the "minimal additional cost" narrative and creating ambiguity about what "extra" means (vs. RLAIF-V's preference data collection vs. GPT-4o API calls).

- Section 4.2 conflates Clark et al. (2019) with the specific Bayes derivation without clearly stating which assumptions were borrowed from that work and which are novel additions.

## Nice-to-Haves

- A length-controlled evaluation (e.g., forced-decoding at fixed token counts) would cleanly separate genuine grounding improvements from length-based metric artifacts.
- Replacing GPT-4o-based perturbation generation with a lightweight, trainable or rule-based generator would validate scalability and break the circular dependency between data creation and evaluation.
- Attention or representation-similarity analysis (e.g., cosine similarity between image and text hidden states before/after training) could provide mechanistic evidence that the model actually shifts attention toward visual tokens.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic W1 (train-inference distribution mismatch as structural invalidity):** The claim that removing $x_p$ at inference is "structurally unsupported" is overstated. The method is essentially training-time data augmentation—a well-established technique. The ablation (Table 5) shows the random perturbation baseline still outperforms the plain LLaVA1.5 baseline, confirming that prefix robustness has value. Dismissing the approach as "noisy prefix augmentation" is precisely what data augmentation is. If the empirical gains are real, the mechanism is secondary to the results.

- **Harsh Critic W4 (unfair baseline comparisons):** The paper explicitly addresses this in Table 2's caption: it clearly separates foundation models (Qwen2-VL, InternVL2, etc.) trained on much larger datasets from LLaVA1.5 fine-tuning variants. The criticism ignores the paper's own transparent comparison design.

- **Harsh Critic on missing statistical significance testing, undisclosed hyperparameters, and trivial reproducibility details:** These are minor nitpicks not standardly expected for this type of submission. Removing them per the rules on reproducibility nitpicks.

- **Harsh Critic on missing appendix proofs/references:** The parser strips the appendix. Any criticism about missing details in appendix (e.g., GPT-4o prompts) should be disregarded as these exist in the original submission.

- **Any criticism doubting the existence or release status of cited models (GPT-4o, LLaVA-Next 34B, RLAIF-V, etc.):** Per hard rules, all cited entities are assumed real and released.

## Novel Insights

The paper makes a useful observation that hallucination mitigation via training perturbation and hallucination mitigation via decoding correction address complementary dimensions—combining them yields additive gains (Table 3). This suggests that hallucination has both training-time (distributional) and inference-time (decoding) components that can be independently optimized. However, the core methodological insight (adversarial prefix + SFT) overlaps substantially with adversarial instruction tuning, reducing the novelty. The HalfScore metric's value lies in its explicit decomposition of hallucinations into object/attribute/relation categories, revealing that relation hallucinations remain the most under-addressed—a finding unavailable from object-level (CHAIR) or holistic (MMHalBench) metrics.

## Suggestions

1. **Add length-controlled hallucination evaluation.** Evaluate CHAIR and HalfScore on captions generated with forced decoding length to decouple genuine grounding improvement from conservative verbosity suppression.
2. **Clarify the relationship to adversarial instruction tuning.** Position the method relative to AIT (sw6Wpx2LGr) explicitly, acknowledging the conceptual similarity while differentiating the application domain (dense captioning vs. dialogue) and contributions (HalfScore, complementarity).
3. **Report results across ≥3 random seeds.** Include standard deviations or confidence intervals for benchmark metrics to establish that gains are statistically significant rather than training noise.
4. **Reframe Section 4.2 as intuition.** Remove the formal mathematical derivation framing and instead present it as informal motivation, since the independence assumptions do not hold for autoregressive models.

## Score and Decision

**Calibration:** This paper was compared against several anchors:
- **Low-scoring (avg <5):** `aNYabH9Th4` (RITUAL, simple TTA for hallucination, 5,5,5,5 → withdrawn), `sw6Wpx2LGr` (AIT, adversarial prepending for hallucination, 5,5,6,6 → rejected). Both share the "simple prefix-perturbation + SFT" pattern but were rejected for limited novelty.
- **Medium-borderline (avg 5-6):** `Ww9rWUAcdo` (theoretical understanding of adversarial perturbations, 6,5,6,5 → accepted poster despite criticized assumptions), `xh3XUaB8M9` (visual evidence prompting, 5,6,6,5 → rejected). PerturboLLaVA is stronger empirically than both but similarly incremental.
- **High-scoring (avg 6-8):** `7lpDn2MhM2` (CHiP, DPO for hallucination, 5,6,8 → accepted poster), `aA33A70IO6` (mistake analysis SFT, 8,5,6,5 → accepted poster). These papers received higher scores due to clearer methodological novelty or stronger empirical separation from baselines.

PerturboLLaVA sits slightly above the rejected borderline papers (RITUAL, AIT) because HalfScore adds genuine evaluation value with human validation, and the complementarity with OPERA shows meaningful methodological breadth. However, it falls below the accepted poster anchors because the core perturbative training concept closely mirrors AIT's rejected approach, the GPT-4o circularity issue, and the length-dependent confound in the primary results. The paper is borderline—solid empirical work, but incrementally close to prior ideas.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>