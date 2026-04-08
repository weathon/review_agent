=== CALIBRATION EXAMPLE 15 ===

# Final Consolidated Review
##Summary

This paper proposes Behavioral Embeddings, a quasi-dynamic framework for program representation that characterizes programs by their optimization sensitivity—how their static features change under diverse compiler optimization probes. The method extracts a "Behavioral Spectrum" (P×56 matrix of scale-invariant reaction vectors), discretizes it via Product Quantization into compositional sub-word codes, and pre-trains a multi-task Transformer (PQ-BERT) to learn contextual dependencies among these codes. Experiments on Best Pass Prediction and -Oz Benefit Prediction using CompilerGym benchmarks show substantial improvements over static, dynamic, and zero-shot LLM baselines.

## Strengths

- **Conceptually novel representation paradigm.** The idea of representing a program by its *reaction* to optimization probes, rather than by its static structure or runtime profile, is a genuine conceptual contribution. This "optimization sensitivity" framing is distinct from prior work and is well-motivated by the static/dynamic dilemma in compiler ML. The paper demonstrates that this signal is more predictive than raw hardware performance counters (Table 9: MAE 85.54 vs. 116.83 on runtime prediction), which is a striking result supporting the core thesis.

- **Compositional encoding via Product Quantization is well-suited to the problem.** The ablation in Table 2 shows that PQ (8 sub-spaces, 256 centroids each) outperforms monolithic K-Means on the regression task (MAE 8.19 vs. 8.24), and the gap would likely widen with more complex spectra. The analogy to RGB decomposition (Section 2.2.1) is clear and appropriate.

- **Strong empirical results with large effect sizes.** On the held-out curated test set, Behavioral-PQ achieves 64.48% Top-1 accuracy on Best Pass Prediction vs. 39.27% for the next-best embedding (inst2vec), and 8.19% MAE on -Oz Prediction vs. 16.23%. The margins are substantial enough to be convincing even without significance testing.

- **Outperforming dynamic baselines without execution.** The comparison in Appendix C.2 showing that Behavioral-PQ (85.54 MAE) outperforms a 28-dimensional hardware performance counter baseline (116.83 MAE) on runtime prediction is particularly noteworthy. It provides direct evidence that optimization sensitivity is a richer signal than single-point runtime snapshots.

## Weaknesses

### Major:

- **Probe–target entanglement risks inflating downstream performance.** The probes are 50-pass sequences explicitly optimized to reduce instruction count (Section 2.1.1). Downstream Task 1 predicts which single pass is best, and these passes appear within the probe sequences. Downstream Task 2 predicts instruction reduction under -Oz, which is the same objective used to construct the probes. The case study in Appendix A inadvertently illustrates this: the ground-truth best pass (-instcombine) appears 3 times within the single most reactive probe. The behavioral spectrum thus encodes direct evidence about the effectiveness of specific passes—this is not traditional label leakage (the model never sees the label), but it does mean the input features carry pass-specific signal that would be absent if the probes were, say, randomly selected or constructed without optimizing for the same metric. **The paper would be significantly strengthened by an experiment excluding probes containing the target pass, or by using probes optimized for a different objective (e.g., execution speedup), to demonstrate that the behavioral sensitivity signal generalizes beyond this entanglement.**

- **Ablation anomaly undermines claimed necessity of scale-invariant quantification for classification.** In Table 2, the No-Relative variant (absolute differences, no logarithmic ratio) achieves 94.33% Top-5 test accuracy on Best Pass Prediction, *outperforming* the full model at 89.55%. The paper's explanation—"coarser-grained signals… can provide reasonably strong predictive power"—is insufficient. If removing a core innovation *improves* performance on one of two tasks, the narrative that all components are essential is weakened. The authors should analyze why scale-invariance hurts classification generalization: does the log compression discard discriminative signal for certain program scales? Does it over-normalize rare but informative large reactions?

### Minor:

- **No continuous-spectrum baseline.** The paper does not evaluate a simple MLP trained on the raw continuous Behavioral Spectrum (P×56 = 5,600 dimensions) without PQ or Transformer encoding. If such a baseline performs comparably, the PQ-BERT architecture's contribution would be diminished, and the core contribution would be the spectrum itself rather than the encoding methodology. This experiment is straightforward and its absence leaves a gap in the ablation story.

- **Test set size is small (~184 curated programs).** While the CompilerGym curated/uncurated split is standard and the effect sizes are large, the total test set comprises only cbench (11) + mibench (40) + chstone (12) + npb (121) = 184 programs. This limits the granularity of conclusions about per-benchmark-family performance. A per-dataset breakdown of results would help readers assess robustness across code domains.

- **Overhead characterization lacks detail on program size scaling.** The paper reports ~0.2s per program for P=100 probes but does not specify whether this is averaged over the (typically small) competitive programming pre-training corpus or the larger downstream benchmarks. Since compilation cost scales with IR size, the overhead for modules from linux-v0 or larger real-world codebases could be substantially higher. A table reporting preprocessing time by benchmark suite would clarify the practical efficiency claims.

### Trivial:

- The term "quasi-dynamic" is used throughout but never given a sharp, formal definition distinct from "compile-time profiling." A one-sentence boxed definition early in the paper would aid reader anchoring.

## Nice-to-Haves

- **Permutation invariance test:** Shuffle the order of probes in the input and measure performance drop. Since probes are independently applied to the baseline IR, demonstrating that sequence order matters would validate the Transformer's role in learning cross-probe dependencies rather than acting as a bag-of-features encoder.

- **Probe count sensitivity analysis:** Plot accuracy vs. number of probes (P). If performance saturates at P=10, running 100 probes is wasteful; if it requires P=100, the computational cost may limit practical adoption.

- **Fine-tuned LLM comparison on main tasks:** The paper compares zero-shot LLMs (Table 6) and fine-tuned LLMs on DevMap (Table 10), but not fine-tuned LLMs on the main Best Pass and -Oz tasks. This would more completely situate the contribution relative to the strongest possible neural baselines.

- **Confidence intervals or bootstrap results** on the curated test set, given its small size, would strengthen the empirical claims.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Missing related works" (CodeBERT, GraphCodeBERT, RL-based optimization methods, hybrid static-dynamic approaches):** Per hard rules, we do not have external sources to confirm the existence or relevance of specific uncited works. The paper's comparison set (Autophase, InstCount, IR2Vec, inst2vec, ProGraML, zero-shot LLMs, dynamic HPCs) covers the main representation paradigms.

- **"Transcends the trade-off" overclaim in abstract:** While the wording is strong, the paper does demonstrate advantages on both efficiency-related and performance-related axes compared to both static and dynamic baselines. The limitations section partially tempers the claim. This is a minor framing issue, not a substantive weakness.

- **"Limited novelty because PQ and Transformers are existing techniques":** This generic argument could be applied to most applied ML papers. The novelty is in the *representation pipeline*—the quasi-dynamic spectrum concept, its extraction methodology, and the compositional encoding tailored to it—not in any single component.

- **"Comparison with dynamic baselines is apples-to-oranges regarding cost":** The paper's framing is explicitly about the trade-off between static efficiency and dynamic insight. Comparing against a dynamic baseline on accuracy (while noting cost differences) is fair and informative. The paper does not claim strict cost superiority over dynamic profiling.

- **Reproducibility concerns about hyperparameters or implementation details:** The paper provides a public code repository and describes key hyperparameters (M=8, k*=256, P=100, learning rates, batch sizes). Per hard rules, trivial implementation details are not a valid weakness.

## Novel Insights

The most striking empirical finding is that optimization *sensitivity* (how a program reacts to transformations) is a more predictive signal than runtime *observation* (what a program does during execution). Table 9 shows Behavioral-PQ outperforming 28-dimensional hardware performance counters on runtime prediction (MAE 85.54 vs. 116.83). This suggests that the space of "how programs change" contains richer information about optimization potential than the space of "what programs do"—a principle that could generalize beyond compilers to other domains where behavioral plasticity matters (e.g., adversarial robustness, domain adaptation). The probe–target entanglement concern, however, tempers how strongly this insight can be claimed without further decontamination experiments.

## Suggestions

1. **Run a leakage-mitigated experiment:** Construct probe sequences that explicitly exclude all 124 candidate passes used in Best Pass Prediction. If performance holds, it validates the generality of behavioral sensitivity; if it drops substantially, it clarifies what the model is actually learning.

2. **Add a raw continuous spectrum baseline:** Train an MLP directly on the P×56 vector. This is the single most informative missing experiment—it cleanly isolates the contribution of the PQ-BERT encoding from the contribution of the spectrum itself.

3. **Analyze the ablation anomaly:** Investigate why scale-invariant quantification hurts classification. Report per-benchmark-family results for the No-Relative variant vs. full model to determine if the degradation is concentrated in specific code domains.

4. **Report preprocessing time by benchmark suite** to clarify the 0.2s claim and enable readers to assess practical feasibility for their use case.

## Axis Evaluations

- **Novelty:** Moderate-to-good. The quasi-dynamic representation concept and the full pipeline are novel; individual components (PQ, BERT pre-training) are established techniques applied in a new context.

- **Technical soundness:** Acceptable with concerns. The methodology is well-designed, but the probe–target entanglement and the ablation anomaly on classification are substantive issues that require either experimental resolution or deeper analysis.

- **Empirical support:** Strong on main results, with gaps. The large performance margins are convincing, but the absence of a continuous-spectrum baseline and the ablation anomaly leave the contribution of the encoding methodology insufficiently validated.

- **Significance:** Good for the compiler optimization community. The quasi-dynamic paradigm offers a practical and empirically effective middle ground. Broader significance to representation learning depends on resolving the entanglement concern.

- **Clarity:** Good. The paper is well-structured, the methodology is clearly described with motivating intuition, and the experimental setup is transparent. The case study in Appendix A is a nice touch.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 4.0]
Average score: 4.7
Binary outcome: Accept
