Now I have enough calibration data. Let me write the final review.

## Summary

MobileLLM-R1 presents a data-centric framework for training sub-billion parameter reasoning models using only 4.2T tokens—~11.7% of Qwen3-0.6B's 36T corpus. The paper introduces benchmark-free, self-evolving data optimization through leave-one-out analysis for dataset selection and influence-based cross-capability weighting for data mixing, combined with a data-model co-evolution strategy for mid-training. MobileLLM-R1-950M achieves competitive reasoning performance (e.g., AIME 15.5 vs. OLMo-2-1.48B's 0.6), and the authors release models, code, and training recipes.

## Strengths

- **Comprehensive, fully open training recipe**: The paper discloses all data sources, mixing ratios, model architectures, and releases models and code (Abstract; Section A.3), which is genuinely valuable for the community and relatively rare for competitive reasoning models.
- **Strong empirical results among fully open-source models**: MobileLLM-R1-950M substantially outperforms OLMo-2-1.48B (57.8 vs. 53.0 on MATH) and SmolLM2-1.7B (57.8 vs. 41.4 on MATH) while being smaller, establishing a strong result in the fully open-source category (Table 2, Figure 9).
- **Insightful LOO analysis**: Figure 3 provides useful empirical data about cross-domain contributions (e.g., StarCoder benefiting math more than OpenWebMath benefits code), offering actionable insights for data curation.
- **Mid-training influence compression finding**: Figure 6 shows that influence-based subsampled data avoids a performance dip at ~30K steps that original data suffers, providing concrete evidence for a practical benefit of data curation during mid-training.
- **Honest scoping of "reasoning"**: The paper adopts a pragmatic stance (Section 2, first paragraph) about treating benchmark gains as proxies rather than claiming genuine cognitive reasoning, which is a healthy framing.
- **Staged post-training ablation**: Table 1 shows that decoupled alignment-then-reasoning (Tulu first, then M+C+S) achieves 68.5 GSM8K vs. 53.1 for joint training, and that science data transfers to math—providing practical training insights.

## Weaknesses

### Fatal
None.

### Major

- **Token-efficiency framing conflates data efficiency with model size advantages**: The paper's headline claim (Abstract, Introduction, Conclusion) repeatedly states the model "matches Qwen3-0.6B with only 11.7% of the tokens." However, MobileLLM-R1-950M has 950M parameters versus Qwen3-0.6B's 600M—a 58% parameter advantage. On a FLOPs basis (Size × Tokens), the gap narrows substantially (950M × 4.2T ≈ 4.0 vs. 600M × 36T ≈ 21.6 parameter-token units—a 5.4× ratio, not 8.5× as token-centric framing implies). The paper's own Figure 1 correctly visualizes the FLOPs comparison, but the text systematically privileges token counts. The claim should either include parameter-matched comparisons or use FLOPs-adjusted framing. Without this, the "data efficiency" narrative overclaims what the experiments support. **Why it matters**: The paper's core contribution claim is about token efficiency; if the 58% parameter advantage accounts for much of the gap, the claim is substantially weaker.

- **"Benchmark-free" claim is misleading**: The paper calls its approach "benchmark-free" (Abstract, Section 2.2, Section 6), but Section 2.1.1 describes constructing capability-probing datasets using FineWeb-EDU classifiers (score ≥ 4), Ask-LLM scoring with domain-specific prompts (code, math, knowledge), and semantic deduplication—these are explicitly designed to proxy the same capabilities measured by MATH, HumanEval, and MMLU. The influence scores (Equations 2–5) that drive all data mixing decisions are computed against these capability probes. The method does not directly use benchmark test sets during training, which is commendable, but it uses carefully constructed capability proxies targeting the same domains. Calling this "benchmark-free" misrepresents the degree of human prior knowledge built into the data pipeline. **Why it matters**: The paper positions itself as a self-evolving system, but the data curation is heavily guided by human-designed capability proxies.

- **Table 2 comparison starts from different training stages**: Baselines use "instruct checkpoints" while MobileLLM-R1 uses "intermediate Tulu3-SFT checkpoints." The paper acknowledges this asymmetry (Table 2 footnote with *), but does not address whether this systematically advantages or disadvantages either side. Different post-training trajectories (instruct fine-tuning, which may include RLHF/DPO, vs. SFT-only) can significantly affect reasoning capability. **Why it matters**: The claim that MobileLLM-R1 outperforms baselines "under identical fine-tuning" depends on starting points being comparable.

### Minor

- **AIME evaluation protocol unspecified**: The AIME score of 15.5 is a headline result, but the paper does not specify whether this is pass@1 vs. pass@k, sampling temperature, or number of generations. For a 30-question competition, pass@k with high k yields very different results than pass@1. This makes the headline number uninterpretable without these details.

- **No validation of influence score calibration**: The influence scores (Equations 2–5) underpin the entire data mixing framework. The paper computes influence via Hessian approximation (AutoMixer) but never validates that these scores are well-calibrated—e.g., by checking that influence-predicted rankings correlate with actual LOO performance differences from Section 2.1.2.

- **Influence convergence interpretation is overstated**: The paper claims (Section 3, Figure 5) that influence scores converging to zero indicates "the dataset's information has been largely exhausted." Standard optimization theory suggests gradient norms shrink as the model approaches a loss minimum, making influence estimates vanish. This is a statement about optimization dynamics rather than information content.

- **Missing data contamination analysis**: Several training datasets (FineMath, OpenWebMath, OpenMathReasoning) potentially overlap with MATH, GSM8K, and HumanEval. Given the emphasis on open-source data and reasoning benchmark performance, this omission is notable.

- **Figure 1 does not include Qwen3-0.6B**: The scatter plot compares MobileLLM-R1 against Qwen2.5, LLaMA3.2, Gemma-3, and SmolLM2, but not Qwen3-0.6B—the primary comparison in the text. This disconnect between figure and narrative makes the FLOPs-based comparison harder to verify.

### Trivial
None.

## Nice-to-Haves

- A parameter-matched comparison (e.g., training a ~600M variant on 4.2T tokens) would substantially strengthen the data-efficiency claim.
- Qualitative examples showing what the influence-based method selects as high vs. low influence samples would help readers understand what the method is optimizing for.
- Reporting total compute cost of the data curation pipeline (LOO experiments, influence computation, iterative mid-training) alongside the training efficiency would allow holistic assessment of the "efficiency" claim.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic claim that LOO only tests single-dataset removal and doesn't inform mixing ratios**: The paper uses LOO for dataset *selection* (Section 2.1) and influence scores for *mixing ratios* (Section 2.2). These are explicitly two different steps in the pipeline. The LOO analysis is not claimed to directly determine mixing ratios—it identifies which datasets are beneficial, and then influence scores determine weights. The critic conflates these.

- **Harsh critic claim that "the comparison is not simply misleading—it is the basis for the paper's most prominent contribution claim"**: While the token-centric framing is indeed misleading (kept as a major weakness), the paper also has substantive contributions beyond the Qwen3-0.6B comparison—namely the open recipe, LOO insights, influence-based mixing, and mid-training co-evolution. This is not a case where the overclaim invalidates all contributions.

- **Formatting nitpicks**: The harsh critic notes some table formatting issues in the extracted text (e.g., Figure 8 tables with "0.0" in the "Open-source weights, data source and training recipes" column for Qwen3 models). These appear to be parser artifacts.

- **Demand for ablation of the full curation pipeline against uniform sampling**: The paper already compares Datamix vs. Original (uniform sampling) in Figure 4, showing consistent PPL improvements. The critic's request for a full end-to-end ablation would require training entire models from scratch, which is an unreasonable additional compute request for a submission.

- **Missing appendix proofs/cost analysis**: The parser strips appendices, so these may exist in the original submission.

## Novel Insights

The paper reveals a practical and surprisingly effective finding: StarCoder benefits math more than OpenWebMath benefits code (reversing the common assumption from Lewkowycz et al., 2022), and FineWeb-Edu acts as cross-domain "glue." The mid-training influence compression finding—that subsampled data avoids a performance dip at ~30K steps (Figure 6)—is a non-obvious result with practical implications for training stability. However, the gap between the paper's ambitious "benchmark-free, self-evolving" narrative and the reality of heavily human-guided capability proxies is the most important meta-observation: the method's success may owe more to the quality of the human-designed capability probes than to the self-evolving nature of the influence optimization.

## Suggestions

- Reframe the core efficiency claim in FLOPs-adjusted terms rather than token-only terms, either providing a parameter-matched comparison or clearly discussing the model-size vs. token-count trade-off per Chinchilla scaling.
- Replace "benchmark-free" with a more precise term like "benchmark-test-set-free" or "indirect-benchmark-guided," and explicitly discuss how the capability-probing datasets encode prior knowledge about target capabilities.
- Report evaluation protocol details (pass@1 vs. pass@k, temperature, number of samples) for AIME and all benchmarks.
- Start all models compared in Table 2 from the same training stage (either all from base or all from instruct checkpoints) to make the comparison fair and transparent.

## Calibration

**Anchors used:**

| Paper | Avg Human Score | Topic | Comparison |
|-------|----------------|-------|------------|
| Why Less is More (Theory of Data Curation) | 7.50 | Data curation theory | Stronger theoretical grounding; MobileLLM-R1 has more engineering contribution |
| How to Train Data-Efficient LLMs | 6.80 | Data-efficient pretraining (AskLLM) | Similar scope; MobileLLM-R1 has stronger empirical results but more overclaimed narrative |
| Tina (Tiny Reasoning via LoRA) | 5.50 | Small reasoning models, open recipe | Similar "open recipe" contribution; Tina is more incremental, MobileLLM-R1 is more comprehensive |
| Fast-dLLM v2 | 6.00 | Overclaimed efficiency, strong engineering | Similar pattern of strong engineering with overclaimed efficiency framing |
| Go4RL | 4.00 | Data mixture for pretraining | Weaker methodology; MobileLLM-R1 is substantially stronger empirically |
| SPUS | 3.00 | Misleading parameter-matched claims, overclaimed efficiency | Notably overclaimed; MobileLLM-R1 is more honest |
| PREMISE | 2.67 | Overclaimed 84.3% token-reduction | Much more overclaimed than MobileLLM-R1 |
| ThinkDial | 5.00 | Open recipe, overclaimed novelty | Similar scope, moderate overclaim |

MobileLLM-R1 has genuinely strong engineering results and practical contributions (open recipe, LOO insights, mid-training co-evolution). Its main weakness is the overclaimed "data efficiency" and "benchmark-free" narrative. Compared to anchors like "How to Train Data-Efficient LLMs" (6.8) and Fast-dLLM v2 (6.0) which have similar patterns of strong engineering with some overclaimed framing, and Tina (5.5) which is a weaker but similar-topic contribution, MobileLLM-R1 sits in the same band. It is stronger than Go4RL (4.0) and ThinkDial (5.0) which have more fundamental methodological issues, but the overclaim on data efficiency holds it back from the 7+ range of purely well-calibrated papers.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>