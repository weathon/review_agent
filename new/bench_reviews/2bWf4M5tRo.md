## Summary

The paper proposes enhancing hallucination detection in LLMs by injecting uniform noise into intermediate layer representations (MLP outputs), arguing that this provides a complementary source of randomness to prediction-layer sampling because noise can disrupt token likelihood orderings whereas sampling preserves them. The method is simple and training-free: add noise ε ~ U(0, α)^d to selected layers during generation, combine with temperature-based sampling, and compute uncertainty metrics over multiple generations.

## Strengths

- **Clear conceptual argument for complementarity with empirical support**: The paper articulates a precise mechanistic distinction—prediction-layer sampling preserves token likelihood ordering regardless of temperature, while noise injection can reverse token orderings by perturbing intermediate representations (Section 1, Section 3.3). This is empirically validated by the Pearson correlation of 0.67 between the two uncertainty measures (Figure 3), confirming they are correlated but capture distinct information.

- **Consistent AUROC improvements for Answer Entropy**: Noise injection improves AUROC for Answer Entropy across all four datasets in Table 3 (+5.40 on GSM8K, +1.76 on CSQA, +1.26 on TriviaQA, +1.59 on ProntoQA), and across two additional model architectures in Table 6 (Llama2-7B-chat: 75.09→76.80; Mistral-7B: 77.03→82.95). This metric-specific improvement is noteworthy.

- **Thorough ablation studies**: Table 4 systematically varies temperature and noise magnitude; Table 5 ablates injection layers; Figure 4 varies number of generations; Section 4.5 tests alternative architectures. These ablations provide useful information about the method's sensitivity to design choices.

- **Method is simple and training-free**: Algorithm 1 requires only adding uniform noise to MLP outputs at selected layers—no additional models, fine-tuning, or architectural modifications needed.

## Weaknesses

### Fatal
None.

### Major

- **No comparison with existing hallucination detection methods**: The paper cites Semantic Entropy (Kuhn et al., 2023), INSIDE (Chen et al., 2024), and DoLA (Chuang et al., 2023) in its framing and related work, yet no experiment compares against any of these as detection baselines. The only comparison is noise injection vs. no noise injection within the same sampling framework. Without knowing whether Semantic Entropy or INSIDE already achieves AUROC levels that render the marginal gains from noise injection practically irrelevant, the paper cannot establish that its contribution represents an actual advance in hallucination detection. This is particularly important because the base AUROC levels without noise are modest (e.g., 62–73 across most Table 3 entries), and existing methods may already surpass these.

- **Overclaimed generality contradicted by inconsistent results**: The abstract claims noise injection "significantly improves detection accuracy" and Section 1 states it is "effective across various datasets, uncertainty metrics, and model architectures." Table 3 shows a different picture: Predictive Entropy on GSM8K degrades by 0.31 AUROC points (62.79→62.48); Normalized Entropy on GSM8K shows zero change (62.36→62.36); most improvements outside of Answer Entropy on GSM8K are under 1.1 AUROC points. The headline improvements (+5.40 on GSM8K with Answer Entropy) come from the metric–dataset combination where α was specifically tuned. The paper acknowledges α = 0.05 "is not the optimal noise magnitude for each dataset" (Section 4.1), but continues to present results as demonstrating general effectiveness. The actual evidence supports a much narrower claim: noise injection substantially helps Answer Entropy on GSM8K and provides modest gains for some other metric–dataset combinations.

- **One-sided noise U(0, α) without justification or comparison**: Throughout the paper, noise is sampled from U(0, α)^d, a non-zero-mean distribution that introduces a systematic upward bias of α/2 per dimension alongside stochastic variance. For a paper that frames noise injection as a form of "sensitivity analysis" to assess "coherence" (Section 1), zero-mean noise (e.g., U(−α/2, α/2) or N(0, σ²)) would be the natural choice. The paper never discusses this design choice and never tests alternatives. While the core mechanism claim (noise disrupts token orderings) holds for both one-sided and zero-mean noise, the systematic bias could independently affect model behavior—e.g., by shifting activation distributions in a way that benefits detection. Without testing whether zero-mean noise achieves similar or better results, it is impossible to determine whether the improvements arise from the claimed "complementary randomness" or partly from this unexamined bias effect.

### Minor

- **Missing noise-only (T=0) AUROC number**: Section 3.2 tests noise injection alone at T=0 and shows histograms (Figure 2a), but never reports the AUROC for this condition. This single number is critical for evaluating the complementarity claim—knowing whether noise-only AUROC is comparable to sampling-only AUROC (73.86 in Table 2) would establish whether combining both sources is genuinely better than either alone, or whether noise injection is simply doing most of the work.

- **Hyperparameter sensitivity reduces practical value**: Table 5 shows optimal α varies by layer group (0.01 for lower, 0.02 for middle, 0.05 for upper); Table 6 notes Mistral requires α = 0.02 vs. Llama2-7B's α = 0.05; Table 4 shows interactions between temperature and noise magnitude. The paper acknowledges this but provides no guidance for selecting α beyond per-dataset validation tuning.

- **Pearson correlation of 0.67 indicates substantial redundancy**: The paper presents 0.67 as evidence for complementarity, but R² = 0.45 means ~45% of variance is shared between the two sources. The paper never quantifies how much of the AUROC improvement is attributable to genuinely complementary information versus simply adding more perturbation overall.

### Trivial
None.

## Nice-to-Haves

- Testing zero-mean noise alternatives (U(−α/2, α/2) or Gaussian) to disentangle bias from variance effects
- Comparison with at least one established hallucination detection baseline (Semantic Entropy or INSIDE)
- Reporting noise-only AUROC to properly substantiate complementarity
- Evaluation on larger/more recent models to test scalability of the approach
- Per-question analysis of when and why combined method helps over sampling alone

## Removed Points

These points are flagged to be removed, treat them with caution.

- **MC Dropout connection absence**: The harsh critic claimed the connection to MC Dropout is "conspicuously absent." While discussing this connection could enrich the paper, the paper's scope is about enhancing sampling-based hallucination detection, not about Bayesian uncertainty estimation. The paper does cite related perturbation-based approaches (DoLA, INSIDE). This is a nice-to-have, not a weakness.

- **Single noise vector per generation across all layers and steps**: The critic argued that using the same ε across all decoding steps and layers is a "very specific design choice" not discussed or ablated. However, this is a reasonable design choice for consistency within a single generation—different noise per step would be a different approach, but the current design is not clearly wrong. This is more of an ablation suggestion than a weakness.

- **"Noisy label" concern about K=5 majority vote**: The critic argued that classifying hallucination based on majority of K=5 answers is a noisy label. This is standard practice in the hallucination detection literature and is not a unique problem with this paper.

- **Statistical significance testing**: The critic requested confidence intervals and p-values. With small improvements (0.2–0.31 AUROC), this is a valid concern, but single-run evaluation without confidence intervals is the norm in this field for large benchmarks. This is a nice-to-have.

- **Evaluation on more capable models**: The critic requested testing on Llama-3, Mistral-large, etc. This is scope creep—the paper tests on three model architectures already.

- **Missing appendix content**: The paper mentions "noise types" experiments in Section 4 and Appendix A for hyperparameter search. These sections were stripped by the parser; they exist in the original submission. Per instructions, missing appendix content is not a valid criticism.

## Novel Insights

The paper's most insightful observation is that prediction-layer sampling and intermediate-layer noise injection affect model outputs through fundamentally different mechanisms—sampling preserves token ordering while noise can reverse it. This distinction, while simple, has not been articulated in prior hallucination detection work and provides a principled reason to combine both sources. However, the practical significance of this insight is limited by the fact that the largest gains are concentrated in a specific metric (Answer Entropy) that was specifically designed for reasoning tasks and tuned on the primary evaluation dataset.

## Suggestions

- Report the noise-only (T=0, with noise) AUROC for GSM8K—this is a single number that would substantially strengthen or weaken the complementarity claim.
- Test at least one zero-mean noise distribution (e.g., U(−α/2, α/2)) to determine whether improvements come from the randomness or the systematic bias in U(0, α).
- Compare against at least one established detection method (e.g., Semantic Entropy) on the same benchmarks to establish practical significance.
- Temper the generality claims to match the evidence: the method works well for Answer Entropy on GSM8K and provides modest gains in some other settings, but degrades or has no effect for some metric–dataset combinations.

## Score and Decision

**Calibration anchors compared:**

| Paper | Score | Decision | Comparison |
|-------|-------|----------|------------|
| DoLa (Th6NyL07na) | 7.25 | Poster | Novel decoding method with 12-17% absolute improvements and baseline comparisons. Much stronger than this paper. |
| TRON (9WYMDgxDac) | 7.0 | Spotlight | Strong theoretical guarantees, comprehensive experiments. Much stronger. |
| Intervening Anchor Token (zGb4WgCW5i) | 7.0 | Poster | Good theory + strong empirical results with baselines. Stronger. |
| Semantic Embeddings UQ (N4mb3MBV6J) | 5.67 | Reject | Limited novelty, missing baselines (INSIDE), incremental improvements. Similar weaknesses, slightly better written. |
| Feeling of Knowing (YFOg1LUGG1) | 5.5 | Reject | Interesting idea but flawed experimental design. Comparable. |
| Scalable Hallucination Detection (GXzwq6waYb) | 4.25 | Withdrawn | Incremental over semantic entropy, missing baselines. Weaker than this paper but similar pattern. |
| AutoHall (LlG0jR7Yjh) | 3.67 | Reject | Missing essential baselines, marginal improvements. Weaker than this paper. |
| Stock Market Prediction (ICwdNpmu2d) | 1.5 | Reject | No baselines, minimal improvements, fundamentally flawed. Much weaker. |

The paper is clearly above the low-scoring anchors (1.5–3.67) which had fundamental flaws. It's comparable to the medium-scoring rejected papers (4.25–5.67) that had interesting ideas but were undermined by missing baselines and incremental contributions. It falls well below the accepted papers (7+) that had stronger empirical evidence and proper baseline comparisons. The one-sided noise issue and the overclaiming of generality push it below the borderline. The Answer Entropy improvements on GSM8K are meaningful, but the narrow scope of these gains and the absence of comparison with established detection methods prevent this from being a convincing contribution.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>