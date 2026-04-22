Now I have a thorough understanding of the paper. Let me write the consolidated review.

## Summary

PerturboLLaVA proposes two contributions: (1) HalfScore, a concept-level hallucination metric for dense captioning that constructs language graphs from captions and measures precision and recall against ground-truth graphs, and (2) a perturbative visual training strategy that injects adversarially crafted misleading text into instruction tuning to reduce the model's over-reliance on language priors, forcing greater reliance on visual inputs. The method is evaluated on LLaVA1.5, showing improvements on HalfScore, CHAIR, HallusionBench, and general multimodal benchmarks with no additional inference cost.

## Strengths

- **Practical and simple method design**: PerturboLLaVA requires only prepending perturbation text to instructions during SFT, with no extra loss terms, reward models, or training stages. Table 1 shows it achieves 1× inference cost vs. 2–5× for OPERA/VCD.

- **Improvements on general benchmarks while reducing hallucination**: Unlike VCD (MMB drops 67.3→66.2) and RLAIF-V (MMB drops 67.3→63.7), PerturboLLaVA improves MMB (+1.6), SEED (+0.3), and CCBench (+1.2) over baseline (Table 3), addressing the common tradeoff where hallucination mitigation hurts general ability.

- **Complementary to decoding strategies**: Table 3 shows PerturboLLaVA + OPERA achieves further gains (Fscore 52.8 vs 52.2, CHAIRs 33.1 vs 36.1), demonstrating the training and decoding approaches address different aspects.

- **HalfScore provides more fine-grained evaluation than CHAIR**: Table 2 reports separate object, attribute, and relation hallucination rates, offering diagnostic insight that object-level CHAIR cannot. Table 6 shows HalfScore correlates better with human judgments (Pearson 80.7) than MMHalBench (71.7).

## Weaknesses

### Fatal

None.

### Major

- **Figure 2 showcase example contains a sport-level hallucination worse than baselines**: The paper's own cherry-picked comparison figure shows PerturboLLaVA outputting "two women playing **badminton**" with "badminton rackets" for an image of **tennis** (line 73). All three competitors correctly identify the sport as tennis. The caption claims "PerturboLLaVA describes rich image details more accurately" (line 77), but the method commits a more fundamental semantic error than any baseline shown. If the paper's best qualitative example demonstrates the method replacing attribute-level hallucinations (extra chairs) with semantic-level errors (wrong sport), this raises serious concern about what the method actually does to output quality. This does not invalidate the quantitative results but severely undermines the qualitative narrative and suggests the method may redistribute rather than reduce hallucination severity.

- **Random perturbation baseline undermines the claimed mechanism**: Table 5 shows that random (non-adversarial, non-image-relevant) text reduces HalfScore-measured hallucinations from 49.2 to 50.7 and improves HalBench from 46.9 to 49.0—a meaningful fraction of targeted perturbation's gains (52.2 and 47.5). If the mechanism is specifically about countering language priors with adversarial perturbations, irrelevant random text should not help much. The paper does not include baselines like dropout on text tokens or standard data augmentation to determine whether the effect is specific to "adversarial language prior perturbation" or is a generic regularization/noise phenomenon. Without isolating the mechanism, the "language prior reduction" framing is not empirically supported—it is equally consistent with a simpler noise-augmentation explanation.

- **Mathematical derivation in Section 4.2 relies on unjustified assumptions**: (1) The conditional independence assumption (Eq. 7→8) that $x_{<k}^p$ and $x_{<k}^{-p}$ are independent given $x_k$ and $I$ is stated without justification; in autoregressive models, language-prior and non-language-prior components are deeply entangled. (2) Dropping $p(x_k)$ because "in a sufficiently uniform dataset" it can be ignored is incorrect for language models with heavily skewed token distributions. (3) The claim that $p(x_k|x_{<k}^p, I)$ can exclude the image and become $p(x_k|x_{<k}^p)$ is self-defeating—if language priors make the image irrelevant, then perturbative training cannot fix this by the paper's own logic. While this section is presented as motivational rather than rigorous, the paper positions it as explaining *why* the method works (specifically reducing language prior reliance), and without these assumptions the claimed mechanism has no theoretical grounding.

### Minor

- **HalfScore validation study is small**: The human study uses only 12 pairwise comparisons per method (4 methods × 12 = 48 total). With this sample size, the reported Pearson correlations of 78.1 and 80.7 (Table 6) have wide confidence intervals, making it hard to confidently claim superior alignment with human judgment over MMHalBench (71.7).

- **Modest improvements on some established metrics**: HallusionBench improvement is only +0.6, SEED only +0.3. No variance or significance is reported, making it unclear whether these are real effects. The HalfScore Fscore margin over OPERA/VCD is only +0.2–0.3 on a self-proposed metric.

- **"No additional computational overhead" claim in abstract is misleading**: The method requires generating perturbation text using GPT-4o for 160k training samples, which involves non-trivial API cost and processing time. Table 1 honestly marks ✗ for "No extra data generation," but the abstract's claim of "without incurring additional computational overhead" contradicts this.

- **Version selection not fully justified**: Version1 is chosen as the primary method for having the best HalfScore Fscore and maintaining general capabilities, but Version3 achieves better CHAIR (32.3 vs 36.1). The choice of which version represents "Ours" in main results is consequential and the justification is implicit.

### Trivial

- None.

## Nice-to-Haves

- A simple data augmentation baseline (e.g., dropout on text tokens, random token shuffling) to isolate whether perturbative training's benefit is truly specific to adversarial language prior perturbation or is generic noise regularization.
- Cross-dataset evaluation beyond LLaVA1.5 training data.
- Error characterization analysis: what *types* of hallucinations remain or shift after perturbative training (e.g., does the method reduce attribute errors but introduce semantic errors as Figure 2 suggests?).
- Attention or representation analysis to directly verify the claim that the model shifts toward relying more on image tokens after perturbative training.
- Explicit reporting of GPT-4o API cost for perturbation generation.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"RLAIF-V comparison is potentially unfair"**: The harsh critic notes the RLAIF-V comparison is unfair because it uses a LLaVA-Next 34B reward model. However, this unfairness *advantages* RLAIF-V, not the authors' method. Per the hard rules, criticisms about unfair comparison where the asymmetry favors the baseline should be removed.

- **"Abstract claims 'no additional computational overhead' is inaccurate"**: Already captured in the minor weakness section with proper nuance. The harsh critic's phrasing was removed because Table 1 discloses the extra data generation cost, and the abstract's claim specifically refers to training/inference cost (which is accurate for inference), not data preparation cost. Downgraded to minor with precise framing.

- **"GPT-4o sees the answer and creates adversarial perturbations — contradictory to 'without disclosing the answer'"**: The paper states GPT-4o "views the image, question and answer" and is instructed to construct perturbations "without disclosing the answer." This means GPT-4o uses the answer as context to craft relevant perturbations but does not reveal the answer in the perturbation text. While the phrasing could be clearer, this is not truly contradictory—it describes different aspects of the prompt design.

- **"Missing cross-dataset generalization"**: Moved to Nice-to-Have. Testing on another training dataset would strengthen the paper but is not a core flaw for a method applied to the standard LLaVA1.5 pipeline.

- **"Statistical significance reporting"**: Moved to Nice-to-Have. While always desirable, single-run evaluation is standard in this community for large-scale benchmarks.

- **"Cost breakdown of GPT-4o generation"**: Moved to Nice-to-Have. This is a practical detail, not a scientific flaw.

- **"Missing appendix proofs/details"**: Removed per hard rules—appendix content is stripped by the parser and exists in the original submission.

- **Formatting/typo complaints**: Removed per hard rules.

## Novel Insights

The most revealing insight from this review is the tension between the paper's ambitious theoretical narrative (adversarial language prior perturbation as a mechanism) and the empirical evidence pointing toward a more generic regularization effect. The random perturbation baseline doing meaningfully well is not just a secondary ablation finding—it fundamentally challenges the paper's claimed mechanism and suggests the method may be better understood as input-level noise augmentation during SFT, similar in spirit to dropout or input perturbation techniques. This reframing would not diminish the practical value but would require the authors to abandon the "language prior reduction" framing that currently drives the paper's narrative, Figure 5's conceptual diagram, and Section 4.2's entire mathematical apparatus.

## Suggestions

- Replace or supplement the random perturbation baseline with standard regularization baselines (text dropout, token shuffling) to properly isolate the adversarial mechanism. If these achieve similar gains, reframe the contribution as a practical and effective noise-augmentation strategy rather than specifically targeting language priors.
- Remove or replace the Figure 2 example with one where PerturboLLaVA does not commit a more fundamental error than the baselines—showing a sport-level hallucination in the method's own showcase is counterproductive.
- Soften the abstract's "without incurring additional computational overhead" to acknowledge the one-time GPT-4o data generation cost, or specify that the claim applies to training and inference only.
- If keeping the mathematical derivation, clearly label the assumptions as approximations/motivational rather than rigorous, or remove it and acknowledge the method's effect may not be specific to language prior reduction.

## Score and Decision

**Calibration anchors:**

| Paper | Score | Comparison |
|-------|-------|-----------|
| TAME (zGb4WgCW5i) | 7.0 Accept Poster | Similar topic (MLLM hallucination reduction), similar practical method, but TAME has stronger theoretical grounding and better qualitative examples. PerturboLLaVA is weaker. |
| Prereq-Tune (UyU8ETswPg) | 7.0 Accept Poster | Similar spirit (synthetic adversarial data for LLM factuality), stronger mechanism isolation. PerturboLLaVA's mechanism is less well-isolated. |
| PATCH (ZPTHI3X9y8) | 6.0 Reject | Similar topic (LVLM hallucination, plug-and-play tuning), similar weakness in mechanism justification. PerturboLLaVA has more metrics but worse qualitative showcase. |
| HQM (kjVgyR3RFr) | 5.5 Reject | Similar (new hallucination benchmark + method), small human study validation. PerturboLLaVA is comparable. |
| SmoothLLM (xq7h9nfdY2) | 4.5 Reject | Shared pattern: random perturbation as defense, mechanism is essentially generic. PerturboLLaVA's situation mirrors this closely. |
| Memorisable Prompting (3viQDuclu0) | 1.7 Reject | Much weaker paper, not a fair comparison. |
| Mosaic-IT (DvU9ijSn1v) | 5.5 Reject | Claims specific mechanism but is essentially generic data augmentation. Very similar weakness pattern to PerturboLLaVA. |

The paper sits in a crowded space of hallucination mitigation methods with real but modest improvements. Its qualitative Figure 2 works against it, the claimed "language prior reduction" mechanism is not well-isolated from generic noise augmentation, and the math is poorly justified. However, unlike truly weak papers, it does provide honest multi-metric evaluation, shows no degradation in general ability (a genuine strength), and demonstrates complementarity with existing approaches. Compared to PATCH (6.0, Reject) and the noise-augmentation papers (4.5–5.5, Reject), the paper has genuine practical value but the mechanism claims overreach. I place it in the 4.5–5.5 range, slightly above SmoothLLM and Mosaic-IT because it has real hallucination reduction on standard metrics, but below PATCH because its qualitative showcase is actually counterproductive and the mechanism isolation is weaker.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>