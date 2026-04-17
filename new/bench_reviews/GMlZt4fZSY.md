## Summary

The paper introduces MobileLLM-R1, a series of sub-billion-parameter reasoning models trained using a data-centric framework that combines leave-one-out dataset analysis, cross-capability influence-based data mixing, and iterative mid-training data-model co-evolution. The key claim is that ~2T curated tokens (trained as 4.2T with resampling) can produce sub-billion models that match Qwen3-0.6B (trained on 36T tokens) on reasoning benchmarks, substantially outperforming other fully open baselines like OLMo-2 and SmolLM-2. The authors commit to releasing models, code, and training recipes.

## Strengths

- **Strong empirical results for the fully open sub-billion model class**: MobileLLM-R1-950M achieves an AIME score of 15.5 vs. 0.6 (OLMo-2-1.48B) and 0.3 (SmolLM-2-1.7B), and 46.3% HumanEval vs. 30.5% for Qwen3-0.6B. These are substantial improvements under identical post-training (Table 2), convincingly demonstrating better pre-/mid-training quality. The consistent gains across 140M, 360M, and 950M models strengthen the evidence.

- **Systematic LOO analysis of data sources**: Section 2.1.2's leave-one-out experiments (Figure 3) provide genuinely interpretable findings—FineWeb-Edu acts as "glue" across all capabilities, StarCoder benefits math more than OpenWebMath benefits code, and Wikipedia contributes little to reasoning. This is useful practical knowledge for the community.

- **Controlled comparison under identical reasoning SFT**: Table 2 finetunes OLMo-2, SmolLM-2, and MobileLLM-R1 on the same reasoning corpora, isolating the contribution of pre-training quality. This is methodologically appropriate and strengthens the causal attribution of gains.

- **Full reproducibility commitment**: Release of models, code, data sources, and mixing ratios for a reasoning-focused sub-billion model is a genuine community contribution, especially given that comparable models (Qwen3) use proprietary training data.

- **Insightful mid-training convergence observation**: Figure 5's finding that influence scores compress toward zero across mid-training stages, and Figure 6's showing that subsampled data avoids performance dips, provide practical guidance for mid-training design.

## Weaknesses

### Major:

- **Influence-based data mixing is validated only against uniform sampling, not against simpler alternatives**: The core methodological claim—"benchmark-free, self-evolving data optimization"—is supported only by comparison with uniform sampling (Figure 4). There is no comparison to simpler, principled alternatives such as domain-stratified sampling with heuristic weights, loss-based data reweighting, or even manually upweighting math+code+knowledge domains based on the LOO findings. In the related Aioli work, the authors found that many data mixing methods converge to near-uniform weights and barely outperform stratified sampling. Without these baselines, it is impossible to determine whether the gains come from the influence machinery specifically or from any reasonable domain-weighting strategy informed by the LOO analysis. This significantly undermines the paper's central methodological contribution.

- **The token-efficiency claims against Qwen3-0.6B are confounded by model size differences**: The headline "11.7% of Qwen's tokens yet comparable performance" compares MobileLLM-R1-950M (950M parameters) against Qwen3-0.6B (600M parameters)—a ~58% parameter advantage. The paper does not disentangle whether the efficiency gains come from data curation, parameter count, architecture differences, tokenizer, or training objectives. There is also no data scaling ablation (e.g., 1T, 2T, 4.2T, 8T) to establish where diminishing returns actually occur. Without this, the "2T tokens suffice" claim is an inference from a single training run, not an empirically demonstrated finding. The paper should acknowledge these confounds more explicitly or provide scaling experiments.

- **Pre-/mid-training contributions are not individually ablated**: The training pipeline combines LOO dataset selection (Section 2.1), influence-based mixing (Section 2.2), and iterative mid-training co-evolution (Section 3), plus knowledge distillation. Table 1 ablates post-training stages, but no experiment isolates the contribution of each pre-/mid-training component. For instance: how much does LOO + uniform mixing yield vs. LOO + influence mixing? How large is the mid-training co-evolution benefit vs. a single round? Without these ablations, the paper cannot establish which components are load-bearing.

### Minor:

- **Capability-probing datasets introduce partial circularity**: The probing datasets D^P_{C,M,K} are constructed from the same source corpora (Table 5) used for training, via hierarchical rejection sampling with model-based filters. While they are distinct from downstream benchmarks, they inherit the distributional biases of the training data. The "benchmark-free" claim is technically accurate (no standard test sets are exposed), but the probing datasets function as implicit evaluation targets, making the claim somewhat overstated. The paper should clarify this distinction more carefully.

- **Computational overhead of the data curation pipeline is not reported**: The LOO experiments require training ~8 models from scratch (one per dataset exclusion), and influence computation requires 3 domain-specialized models trained to convergence with 10 checkpoints each, plus iterative mid-training refinement. This is substantial compute that may offset the claimed token-efficiency gains from using fewer training tokens. The paper should report the total compute (including curation) to allow readers to assess the real cost-benefit trade-off.

- **Evaluation breadth is limited to reasoning-centric benchmarks**: The paper focuses on MATH, GSM8K, AIME, HumanEval, LiveCodeBench, LCBv6, and MMLU. Broader generalization benchmarks (e.g., commonsense reasoning, multilingual, creative tasks) are absent, which is a particular concern for capacity-limited models where capability trade-offs are acute. Table 1 shows MMLU drops when reasoning data is added, but this trade-off is not explored in depth.

- **No reporting of variance or repeated runs**: Results on smaller benchmarks like AIME (30 problems) and HumanEval (164 problems × single pass) can be noisy. The paper reports single-run results without standard deviations, making it difficult to assess whether differences of a few percentage points are reliable.

### Trivial:

- The paper does not report the final dataset mixing weights from Eq. 5, making it harder to assess whether the influence-based method produces intuitive mixtures.

## Nice-to-Haves

- A controlled experiment comparing influence-based mixing against a simple heuristic mixture (e.g., manually upweighting code 3× and math 2× based on LOO findings) would substantially clarify the method's marginal value.
- A data scaling curve (1T, 2T, 4.2T tokens) would empirically ground the "2T suffices" claim.
- Evaluation on broader NLP benchmarks (HellaSwag, ARC, PIQA, WinoGrande) to verify the data curation does not disproportionately harm non-reasoning capabilities.
- Qualitative examples of model reasoning traces to demonstrate whether MobileLLM-R1 genuinely produces multi-step CoT reasoning versus short pattern-matched answers.

## Removed Points

- **"The comparison with Qwen3 is unfair because Qwen3's training details are proprietary"**: Removed per the rule that we do not question availability of cited models or references. The reviewer's concern about architectural/tokenizer differences is valid but the specific framing about unverifiability is removed.
- **"No qualitative examples of reasoning traces"**: This is a nice-to-have suggestion but not a substantive weakness; the paper demonstrates reasoning via benchmark scores which is the standard evaluation for this type of work.
- **"Formatting/style issues"**: Removed per the rule against formatting nitpicks.
- **"The paper should compare against models of exactly the same parameter count"**: While the 950M vs. 600M comparison is a valid confound (kept as a major weakness), demanding exact parameter matching is unreasonable—the paper's scope is sub-billion models and Qwen3-0.6B is the closest available model.
- **"Model-based data filtering (Ask-LLM) may introduce bias"**: This concern was raised by the human finder from ProX reviews. While theoretically valid, the paper uses Ask-LLM filtering as part of the probing dataset construction, not as the primary methodological contribution. The influence scoring is the key mechanism, and any bias introduced by Ask-LLM would affect the proxy evaluation, not the training itself. This is a minor concern, not a major flaw. Weakened to a brief note in minor weaknesses section.

## Novel Insights

The LOO analysis revealing that StarCoder benefits math more than OpenWebMath benefits code is a counterintuitive finding that challenges the common assumption of asymmetric math→code transfer. Additionally, the observation that FineWeb-Edu (web-scale general data) serves as cross-domain "glue" even for reasoning tasks suggests that broad distributional coverage complements domain-specific data in ways that smaller models particularly benefit from—possibly because their limited capacity makes interference between specialized domains more likely, and general web data provides shared representations that mitigate this.

## Suggestions

- **Add a comparison against a simple heuristic baseline for data mixing**: Even a single ablation (LOO-informed manual weighting vs. influence-based weighting) would dramatically clarify the marginal contribution of the influence machinery.
- **Report the actual mixing weights**: A table showing the final sampling proportions from Eq. 5 for each source dataset would improve interpretability and reproducibility.
- **Acknowledge the parameter-count disparity with Qwen3-0.6B explicitly** and frame the efficiency claim more precisely (e.g., "our 950M model with curated 4.2T tokens matches Qwen3-0.6B trained on 36T tokens, though architectural and parameter-count differences preclude attributing this solely to data curation").

## Score and Decision

Calibration against similar papers:
- **LokiLM** (1.4B model, no ablations, no release, scores 1–6, rejected): MobileLLM-R1 is clearly superior—stronger results, open release, systematic LOO analysis, and controlled Table 2 comparisons.
- **Arctic-SnowCoder** (1.3B code model, data curation focus, limited evaluation, scores 5–6, rejected): MobileLLM-R1 has broader evaluation, more systematic analysis, and open release, placing it above this.
- **ADO** (data mixing with heuristics, limited validation against baselines, scores 5–6, accepted poster): ADO has similar methodological weaknesses (heuristic choices without ablation) but less impressive results. MobileLLM-R1 has stronger empirical contributions.
- **Aioli** (unified data mixing framework, found many methods barely beat uniform, scores 5–8, accepted poster): Aioli's finding that data mixing methods often converge to near-uniform weights is directly relevant and somewhat undermines MobileLLM-R1's influence-based claims.
- **Smaller, Weaker, Yet Better** (compute-optimal sampling for reasoning, scores 6–8, accepted poster): Similar topic space but cleaner experimental design.

MobileLLM-R1 has genuinely strong empirical results and useful practical findings, but its central methodological contribution (influence-based data mixing) is not well-validated against simpler alternatives, and the efficiency narrative is overstated. The open-release commitment and LOO analysis are real contributions. I place it above ADO/Arctic-SnowCoder but below Smaller, Weaker, Yet Better due to the methodological validation gap.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>