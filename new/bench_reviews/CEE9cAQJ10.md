## Summary

The paper introduces GSDP (Graph-based Synthetic Data Pipeline), a framework for synthesizing mathematical reasoning data at scale and low cost. GSDP extracts knowledge points (KPs) from 7.5K seed problems, constructs a Knowledge Point Relationships Graph (KPRG) that captures both explicit (one-hop) and implicit (multi-hop) relationships between KPs, and uses KP combinations drawn from different graph distances to prompt an open-source model (DeepSeek-Math-RL, LLaMA3.1-70B) into generating new problems and solutions. A multi-model scoring ensemble filters for quality. The resulting GSDP-MATH dataset contains 1.91M question-answer pairs, achieving a 255× expansion ratio at roughly 1% of the synthesis cost of GPT-4-based methods. Fine-tuning Mistral-7B on GSDP-MATH yields 37.7% on MATH and 78.4% on GSM8K, outperforming prior synthetic-data-trained models under comparable settings.

## Strengths

- **Novel graph-based expansion mechanism**: The KPRG construction and the taxonomy of one-hop, two-hop, three-hop, and community KP combinations provide a principled way to achieve high expansion ratios while introducing novel KP pairings not present in seed data. This is a meaningful advancement over prior methods that rely on seed rewriting or GPT-4-generated KPs (Section 2.3–2.4).

- **Strong empirical results across multiple base models and benchmarks**: Consistent improvements are shown on Mistral-7B, LLaMA3-8B, and Qwen1.5-7B across MATH, GSM8K, GAOKAO, and SVAMP. GSDP-7B's 37.7% on MATH and 78.4% on GSM8K are competitive or superior to prior work (Table 1).

- **Cost efficiency**: Using only open-source models for the entire pipeline is a clear practical advantage. Table 2 shows the synthesis cost per data point is approximately 1% of GPT-4-based methods, which is a compelling contribution for the community.

- **Internal ablation of KP combination types**: Figure 4 demonstrates a clear progression from one-hop-only (GSDP-One-Base: 17.1% MATH) to adding implicit combinations (GSDP-3: 33.1% MATH) to the full pipeline (GSDP-MATH: 37.7% MATH), providing evidence that adding diverse KP combinations helps.

- **Thoughtful filtering strategy**: The joint scoring approach (Section 2.5, Table 4) with weighted problem evaluation and single-vote veto for solutions is a sensible multi-model quality control mechanism with a systematic analysis of precision–retention tradeoffs.

## Weaknesses

### Fatal

None.

### Major

- **The core claim that graph-based implicit relationships drive performance is not validated against simpler baselines.** The paper's central contribution is the KPRG and its exploitation of implicit (multi-hop) KP relations. However, no experiment compares GSDP against a non-graph baseline—e.g., randomly sampling KP pairs at the same volume, or pairing KPs by embedding similarity rather than graph distance. Section 3.7 only compares subsets of GSDP's own data (one-hop vs. two-hop vs. three-hop), all of which already rely on the KPRG structure. The improvement from GSDP-3 over GSDP-One-Base could be explained by simply having more diverse KP pairings (regardless of graph structure) or by having more data. The impressive downstream results could be driven by "large-scale KP-based synthesis with strong models + aggressive filtering," not specifically by the graph mechanism. Without a random-combination or similarity-based baseline at matched data volumes, the evidence for the graph being the key ingredient is missing. This matters because the entire framing and contribution claim of the paper is centered on the graph.

- **Unfair comparisons with prior methods conflate data quality with data quantity and training recipe.** GSDP-MATH contains 1.91M pairs, vastly more than MetaMath (~395K), WizardMath (~96K), or MathCoder (~80K). Table 1 compares models across methods with different data volumes, different base models, different fine-tuning recipes, and different additional data sources. The paper's claim that "GSDP consistently shows superior performance across all three dimensions" (Figure 1) does not isolate whether improvements come from GSDP's data quality, from training on 10–20× more data, or from training recipe differences. The cost comparison (Table 2) also has an asymmetry issue: GPT-based methods' costs include generation of all data (including subsequently filtered-out data), but it is unclear whether GSDP's reported 1.23 cents per data point accounts for the substantial GPU time consumed by the multiple synthesis, scoring, and filtering models on the 4.24M raw samples before arriving at 1.91M final samples. A more apples-to-apples comparison would normalize by training data volume or report cost-per-accuracy-point.

### Minor

- **No human evaluation of generated data quality.** The 1.91M samples are verified only by an ensemble of open-source models calibrated against GPT-4 (Table 4). GPT-4 itself makes errors on non-trivial math; treating it as ground truth inflates quality claims. Even a modest human audit of 100–200 stratified samples would substantially strengthen confidence in the "high-quality" claim (Section 3.8).

- **Pre-training experiment is under-specified.** Table 3 shows GSDP-LLaMA3-8B results, but the paper does not report what "publicly available pre-training data" was combined with GSDP-MATH, in what proportions, or how many of the 3.5B tokens came from GSDP-MATH. Without a control that trains LLaMA3-8B on the same public data without GSDP-MATH for the same token budget, the MATH/GSM8K gains cannot be cleanly attributed to GSDP-MATH.

- **Decontamination from MATH benchmark is mentioned but not described.** The seed data is the MATH training set, and evaluation is on the MATH test set. One-hop KP combinations directly reproduce seed KP pairings, making indirect leakage plausible. The paper states decontamination is performed but provides no details on the method or its scope (string match? n-gram overlap? embedding similarity?).

- **The three-hop combination definition is ad hoc.** Restricting three-hop pairs to only those involving "core" KPs (highest-degree nodes) is an arbitrary design choice without justification or ablation. The paper does not explore whether non-core three-hop pairs are harmful, nor whether larger cliques yield different results.

### Trivial

- The naming in Figure 4 is confusing: "GSDP-2" corresponds to "GSDP-One-Base" rather than two-hop data, which obscures the ablation structure.
- The abstract claims "×100 lower costs" but Table 2 shows costs ranging from 23 to 220 (in 0.01 cents), compared to GSDP's 1.23—the ratio varies from ~19× to ~179× depending on the method, so "×100" is a rough approximation that could mislead.

## Nice-to-Haves

- A per-category breakdown of MATH results (Algebra, Counting & Probability, Geometry, etc.) to reveal which domains benefit most and where gaps remain.
- An ablation varying data volume (e.g., 500K, 1M, 1.91M) to determine whether gains plateau and whether full 255× expansion is necessary.
- Testing GSDP on a domain beyond mathematics (e.g., physics, coding) to assess generalizability of the pipeline.

## Removed Points

These points are flagged to be removed; treat them with caution.

- *(Harsh Critic)* "Comparisons are structurally unfair because different base models are used." — While true that Table 1 includes models on different base architectures, this is standard practice in this research area. The paper also reports results on three base models (Mistral-7B, LLaMA3-8B, Qwen1.5-7B) under identical fine-tuning settings, which provides the most informative comparison. The cross-architecture comparisons in the upper portion of Table 1 are not the primary claim. **Kept as a minor note under data quantity confounds, but removed the strongest version of this criticism.**

- *(Harsh Critic)* "Cost accounting is opaque for GPU-based synthesis." — The paper references Appendix B for cost calculation details. While the main text could be clearer, the claim is not unsubstantiated; it is just not fully visible in the main text. **Partially kept in the major weakness above (noting the asymmetry between reported and total synthesis costs), but removed the claim that it is entirely unsupported.**

- *(Harsh Critic)* "GPT-4 labels as ground truth without calibration." — This is a valid concern and is included as a minor weakness. However, the harsh critic treats it as undermining the entire quality claim, which overstates the issue: the downstream performance on standard benchmarks is the ultimate validation, and it is strong. **Kept as a minor weakness about data quality verification, not elevated to fatal.**

- *(Neutral Reviewer)* "Missing KP statistics (number of clusters, degree distribution, etc.)" — While useful for reproducibility, this is a detail-level request, not a substantive flaw in the contribution. **Moved to nice-to-have.**

- *(Spark)* "Statistical significance / variance across runs." — Single-run evaluation is the norm for large-scale fine-tuning in this community (see OpenMathInstruct-2, MAmmoTH, MathCoder). **Removed as not standard practice.**

- *(Spark)* "Cost-effectiveness (accuracy per dollar) should be reported." — This is a nice suggestion but not a flaw in the current evaluation. **Moved to nice-to-have.**

## Novel Insights

The observation that the ablation in Figure 4 actually cuts both ways is insightful: GSDP-One-Base (one-hop, no repetition) achieves only 17.1% on MATH, while GSDP-One (with edge-weight-based repetition) achieves 22.4%, and GSDP-4 (adding two-hop + three-hop) reaches 34.9%. The jump from 17.1% to 22.4% comes purely from duplicating data along high-frequency KP pairings, which is a data quantity effect unrelated to the graph structure. The further jump to 34.9% could similarly be driven by data volume and diversity rather than by graph-derived implicit relationships specifically. This underscores the need for a volume-matched random-combination baseline to disentangle these effects.

## Suggestions

1. **Run a random KP combination baseline at matched data volume.** Take the same KPs, randomly pair/triple them without the graph, generate the same number of problems with the same model and filtering, and fine-tune Mistral-7B. This is the single most important experiment for validating the core contribution.

2. **Report total synthesis cost (including discarded samples).** The 4.24M→1.91M pipeline means roughly 55% of generated data is discarded. Report the GPU cost of the full pipeline, not just per-accepted-sample cost, for a fair comparison to methods that may have higher acceptance rates.

3. **Add a human evaluation.** Even 100–200 samples would provide ground-truth error rates and validate the GPT-4-calibrated filtering.

4. **Clarify the pre-training setup and add a control.** Report the data mixture and include a "same public data, no GSDP-MATH" control to isolate the contribution of GSDP-MATH.

## Evaluation

- **Originality**: Moderate. The KPRG mechanism for exploring implicit KP relationships is a novel and principled idea for data expansion, though graph-based entity extraction for augmentation has appeared in related work (e.g., EntiGraph). The composition of known techniques (KP extraction, embedding clustering, multi-model voting) is sound but individually standard. The novelty is primarily in the KPRG perspective and the specific pipeline design.

- **Importance of research question**: High. Scalable, low-cost synthesis of high-quality math data is practically important for the LLM community.

- **Claims well-supported**: Partially. The downstream results are strong, but the central mechanistic claim (graph-based implicit relationships are the key driver) lacks the critical ablation. The cost comparison has asymmetries. The pre-training claim is under-specified.

- **Soundness of experiments**: Moderate. Strong fine-tuning results, good internal ablations, but missing the most important baseline (random KP combinations) and with an under-specified pre-training experiment.

- **Clarity**: Good. The paper is well-structured, clearly written, and the figures effectively illustrate the pipeline.

- **Value to research community**: High. The dataset (GSDP-MATH, 1.91M pairs) and the pipeline are useful contributions if released, and the low-cost open-source approach is valuable.

## Score and Decision

**Calibration anchors:**
- MathCoder2 (Accept Spotlight, avg ~7.3): Novel dataset creation method, strong results, clean experiments, well-validated.
- OpenMathInstruct-2 (Accept Poster, avg ~6.5): Large-scale useful dataset, systematic ablations, but process not very novel.
- EntiGraph/Synthetic continued pretraining (Accept Oral, avg 8): Strong methodology with theoretical backing, clear causal validation.
- ScaleQuest (Reject, avg ~5.5): Similar domain, similar concerns about pipeline complexity, fairness of comparisons, and overclaiming.
- MAmmoTH (Accept Spotlight, avg ~7.2): Strong dataset contribution but limited novelty (dataset curation + fine-tuning).

GSDP falls between ScaleQuest and OpenMathInstruct-2. It has stronger novelty than OpenMathInstruct-2 (the KPRG mechanism is a genuine methodological contribution) and stronger results than ScaleQuest, but has a more significant evidential gap than either: the core mechanistic claim (the graph drives the gains) is unvalidated against the most obvious alternative (random KP combination). This is not a minor ablation—it goes to the heart of what makes the paper's contribution "graph-based" rather than just "large-scale KP-based synthesis." The comparison fairness issues are real but not fatal. The lack of human evaluation and underspecified pre-training experiment are addressable in revision.

Given these considerations, the paper is above the acceptance threshold due to its practical contribution (a useful, low-cost pipeline and dataset) and reasonable novelty (KPRG), but the evidential gap on the core mechanism keeps it from a higher score. A score of **5.5** reflects solid work with a notable gap in validation of the central claim.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>