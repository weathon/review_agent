## Summary

VideoJudge introduces a bootstrapped framework for training small (3B/7B) MLLM evaluators specialized for video understanding. A generator–evaluator pipeline iteratively produces and refines candidate responses across a 1–5 rating scale, yielding ~100K training examples that are used to fine-tune Qwen2.5-VL backbone models. The trained VideoJudge models match or outperform much larger baselines (32B/72B) on multiple meta-evaluation benchmarks, and a variant (VideoJudgeR-3B) generates instance-specific rubrics at inference time, improving interpretability.

## Strengths

- **Addresses a real gap**: Scalable evaluation for video understanding is genuinely underserved — traditional metrics are poor proxies and human annotation is costly. The paper tackles a practical and important problem.
- **Principled bootstrapping pipeline**: The iterative generate–evaluate–refine loop with acceptance thresholds (Algorithm 1) is clearly formalized and provides quality control over synthetic data. This is a well-designed extension of prior LLM-as-judge bootstrapping ideas to the video domain.
- **Strong results relative to model size**: VideoJudge-7B achieves Spearman correlations of 0.80/0.76 on VideoJudgeLLaVA/VCG, outperforming or matching Qwen2.5-VL-32B/72B. On pairwise evaluation, VideoJudge-7B reaches 98.6% accuracy on VideoJudge-Pairwise. On LongVideoBench, VideoJudge-7B achieves a PSup of 0.66 and ∆(C-D) of 1.16, outperforming all video-language baselines except Qwen2.5-VL-72B.
- **Rubric generation is a meaningful contribution**: VideoJudgeR-3B, trained on only 10% of data, achieves MAE comparable to Qwen2.5-VL-32B (0.59) and is preferred by human annotators over rubrics from much larger models (Figure 3).
- **Thorough analytical sections**: The frame-count ablation (optimal ~120–240 frames), temperature robustness analysis, and honest error analysis identifying overestimation bias add practical value beyond raw benchmarks.

## Weaknesses

### Major:

- **Closed-loop evaluation undermines headline claims**: Two of four pointwise benchmarks (VideoJudgeLLaVA-MetaEval, VideoJudgeVCG-MetaEval) and the pairwise VideoJudge-Pairwise are constructed using the same generator–evaluator pipeline used for training data creation. On these pipeline-derived benchmarks, VideoJudge outperforms the 72B model that produced the evaluation labels — which is expected for a model trained to approximate that labeler, not evidence of superior evaluation capability. The paper acknowledges "partial closed-loop effects" in §7, but this understates the issue: the majority of the reported "wins" come from these benchmarks. On the independent benchmarks (VATEX, LongVideoBench), results are more modest — VideoJudge-7B does not consistently outperform Qwen2.5-VL-72B. This does not invalidate the approach (distilling a large evaluator into a smaller specialist is itself valuable), but it means the claim that VideoJudge "matches or surpasses much larger models" requires significant qualification.

- **Severe overestimation bias limits practical reliability**: The error analysis (§6.2) reveals that 81.3% of rating-4 responses are incorrectly scored as 5, and 46.6% of rating-3 responses are inflated to 5. For a model whose primary function is evaluation, the inability to distinguish "good" from "excellent" is a critical failure mode. The paper identifies this problem but offers no mitigation, and it directly undermines the utility of the pointwise scores in practice.

- **Limited and narrow human validation**: Human evaluation covers only 250 pairs restricted to the 2-vs-3 rating region and only measures pairwise preference direction — not absolute calibration across the full 1–5 scale. Given that the 1–5 pointwise rating is the core supervision signal and the primary evaluation target on multiple benchmarks, this is insufficient to support claims of "alignment with human judgment." No human study validates whether the evaluator's 1–5 ratings correspond to human perceptual quality across the scale.

### Minor:

- **Generator/evaluator model dependency is unexplored**: The entire pipeline depends on specific models (Qwen2.5-VL-72B as evaluator, GPT-4o-mini descriptions as context). No ablation varies these components, so it is unclear how robust the framework is to different choices. This is relevant because the evaluator's biases (including the overestimation pattern) propagate directly into the training data and meta-evaluation labels.

- **Rubric evaluation methodology has partial circularity**: The claim that VideoJudgeR-3B rubrics win 92.7% against GPT-4o-mini is based on using GPT-4o-mini as the LLM judge, while the human evaluation (Figure 3) also shows strong results. The LLM-as-judge number is inflated by style alignment with the evaluating model; the human evaluation provides more legitimate evidence but lacks inter-annotator reliability details and specific evaluation criteria.

- **The LLM vs. MLLM comparison is confounded**: The claim that "providing video inputs is crucial" (Abstract) compares MLLM models (processing raw frames) against LLM models (processing text descriptions generated by another MLLM). The confound is that the descriptions may be lossy; the comparison only shows that this particular description channel is weaker than raw video, not that video modality is necessary for evaluation.

### Trivial:

- The pairwise feedback results are inconsistent (VideoJudge-7B with feedback drops on VideoAutoArena: 87.45→85.49), which the paper notes is "mixed" but does not analyze further.

## Nice-to-Haves

- Comparison with proprietary baselines (GPT-4o, GPT-4o-mini) in the main pointwise/pairwise tables, which would situate VideoJudge relative to the most widely deployed judge approach.
- Ablation of the feedback/refinement loop (single-pass vs. iterative) to demonstrate its contribution.
- Full-scale rubric model training (currently only 10% of data) to validate the scaling of the rubric approach.
- Per-rating calibration analysis (reliability diagrams, ECE per bucket) to complement the aggregate metrics.
- Cross-domain evaluation beyond general video QA/captioning (e.g., instructional, medical video).

## Removed Points

- **Formatting/style nitpicks**: Removed per instructions (e.g., garbled table rendering is a PDF extraction issue, not a paper problem).
- **Reproducibility concerns about proprietary models**: The paper cites GPT-4o-mini and Qwen2.5-VL-72B as pipeline components; these exist and are available per our rules. Removed.
- **Demand for confidence intervals/statistical significance on large-scale benchmarks**: Single-run evaluation is standard in this research area. Removed.
- **Demand for comparison with VideoScore/other video reward models**: Removed per instructions — missing related work is not a valid criticism without external source verification.
- **Claim that VideoJudge outperforming 72B on pipeline-derived benchmarks is "conceptually incoherent"**: This overstates the case. It is predictable (distillation effect) but not incoherent — it demonstrates efficient compression. The real issue is the overclaim about human alignment, not the math. Softened.

## Novel Insights

The observation that instance-specific rubric generation by a 3B model can produce rubrics preferred over those from much larger models (including GPT-4o-mini) is the most striking finding, suggesting that specialized training teaches evaluation *criteria* more effectively than scale alone. The temperature robustness result — base Qwen2.5-VL-3B degrades monotonically with temperature while VideoJudge improves — is also notable, indicating that rubric-guided training enforces more deterministic evaluation behavior. However, these findings are partially undermined by the closed-loop evaluation setup.

## Suggestions

1. **Separate and clearly distinguish results on pipeline-derived vs. independent benchmarks** in the main results tables and discussion. Report the "independent generalization" story separately from the "pipeline-approximation" story, and calibrate claims accordingly.
2. **Address overestimation bias** through targeted interventions: adding harder negatives near rating 5 during bootstrapping, or applying calibration correction (e.g., temperature scaling) post-hoc.
3. **Expand human evaluation** beyond 250 pairwise 2-vs-3 pairs to cover the full rating scale, including validating pointwise absolute scores against human ratings.
4. **Ablate the generator/evaluator models** in the bootstrapping pipeline (e.g., use a 7B evaluator instead of 72B) to characterize sensitivity and demonstrate that the framework is not tied to a specific teacher.

## Score and Decision

**Calibration**: Comparing against similar LLM-as-judge papers:
- JudgeLM (Spotlight, avg ~8): fundamentally novel self-instruction framework with strong human evaluation and generalization evidence.
- Self-Taught Evaluators (Reject, avg ~5.2): similar bootstrapping idea but with circularity concerns and limited human validation.
- "Is Your Video Language Model a Reliable Judge?" (Poster, avg ~6.5): systematic evaluation study with focused scope.
- Auto-J (Poster, avg ~5.3): automatic judge distillation with similar data-generation circularity.

VideoJudge is between Self-Taught Evaluators and JudgeLM in terms of both novelty and evaluation rigor. The closed-loop concern is significant and the overestimation bias is a practical limitation, but the framework is the first dedicated video-judge system with publicly released benchmarks and models. The independent-benchmark results (VATEX, LongVideoBench, VideoAutoArena) are modestly positive even if not dominant, meaning the core idea works but the headline claims require qualification. This positions the paper in the 5–6 range.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>