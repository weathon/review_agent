=== CALIBRATION EXAMPLE 6 ===

# Final Consolidated Review
## Summary

Rex-Thinker reformulates object referring expression comprehension (REC) as an explicit Chain-of-Thought reasoning task with a structured Planning–Action–Summarization framework. It uses a two-stage architecture: an open-vocabulary detector generates candidate box proposals, and an MLLM performs step-by-step verification against the referring expression. To support this, the authors construct HumanRef-CoT (90,824 samples), a large-scale CoT-annotated dataset, and train via cold-start SFT followed by GRPO-based reinforcement learning, demonstrating state-of-the-art performance on HumanRef and improved hallucination rejection.

## Strengths

- **Novel and well-motivated task formulation.** The Planning–Action–Summarization CoT decomposition for REC is distinct from both direct coordinate prediction and simple retrieval-based selection. Each reasoning step is explicitly grounded to a specific candidate region via box hints, providing genuine interpretability that current methods lack. The framework directly addresses two under-explored properties—verifiability and trustworthiness—that are critical for real-world deployment.

- **Substantial improvement in rejection/hallucination reduction.** The 13.8-point improvement in Rejection Score (Rex-Thinker-Plain: 53.5 → Rex-Thinker-CoT: 67.3, Table 3) demonstrates that CoT supervision meaningfully enables the model to abstain when no matching object exists—precisely the capability that direct-prediction methods lack. This is the paper's most concrete and impactful empirical result.

- **Large-scale dataset contribution with quality controls.** HumanRef-CoT (90,824 samples) is the first large-scale CoT-style referring dataset, with a two-stage automated filtering pipeline (internal logical coherence + final accuracy matching) that removes 3–7.6% of generated samples depending on complexity (Table 1). Human evaluation on 600 samples confirms zero logical/summarization errors in the filtered data, with only 1.2% factual error rate in intermediate steps.

- **Strong ablation evidence for two-stage training necessity.** Table 12 demonstrates that CoT-based cold start is essential for GRPO convergence (Avg DF1: 83.5 with CoT SFT vs. 77.8 without), and Figure 5 shows qualitatively that GRPO without CoT initialization produces incoherent reasoning traces. Table 5 rigorously motivates the two-stage detector+MLLM architecture by showing standalone detectors lack language comprehension (25.7 precision) and standalone MLLMs lack localization (69.4 recall).

## Weaknesses

### Major:

- **Detector dependency creates a hard upper bound on recall that is not analyzed.** The retrieval-based architecture means any object missed by Grounding DINO is irrecoverable regardless of reasoning quality. While Table 5 shows the fine-tuned detector achieves 92.0 recall, the paper never quantifies the downstream impact of detector misses on final performance. This is a fundamental architectural limitation: the method's trustworthiness and completeness claims only hold within the candidate set provided. No failure analysis examines what fraction of Rex-Thinker's errors stem from detector misses versus reasoning failures.

- **Reasoning–answer inconsistency undermines the verifiability claim.** Appendix A.4.2 (Figure 13) acknowledges cases where the reasoning trace identifies N objects but the final output contains M ≠ N boxes. This directly contradicts the core contribution of "verifiable" and "grounded" predictions—if the reasoning chain and answer can diverge, the CoT trace does not faithfully explain the model's decision. The paper proposes a consistency reward as future work, but this gap exists in the current method and is not quantified: how often do these inconsistencies occur, and do they concentrate in specific expression types?

- **GPT-4o data generation with oracle answers raises questions about reasoning trace authenticity.** During annotation, GPT-4o is provided the ground-truth answer (Section 3.2, Figure 2), meaning the CoT traces may represent post-hoc rationalization rather than genuine inference. The paper's own evaluation (Table 8) shows GPT-4o without answer hints achieves only 53.2 DF1—a large gap from the annotated quality. This creates a distribution mismatch: the training data contains reasoning that GPT-4o cannot actually produce without answer supervision, which may limit the model's ability to learn authentically generalizable reasoning patterns.

- **Zero-shot generalization claims in the abstract need calibration.** The abstract states the model shows "strong generalization in out-of-domain settings," but Table 4 shows Rex-Thinker-CoT zero-shot achieves 81.2/80.3 on RefCOCOg, which is *below* baselines trained on RefCOCOg (e.g., RexSeek-7B at 84.0/84.4). While the zero-shot vs. supervised comparison is acknowledged in Appendix A.3.4, the abstract language overstates the result. Only after GRPO fine-tuning on RefCOCOg does the model reach SOTA (89.2/88.8), which is supervised adaptation, not zero-shot generalization.

### Minor:

- **Inference speed overhead is significant but underemphasized.** Table 15 (appendix) reports 6.68s per image for Rex-Thinker-GRPO vs. 1.13s for the Plain model—approximately 6× slower. For a method claiming practical trustworthiness, this latency is non-trivial and deserves discussion in the main text alongside the accuracy gains.

- **Limited quantitative evaluation of reasoning trace correctness.** The paper claims interpretable, grounded reasoning but primarily evaluates end-task accuracy. The human evaluation (A.2.2) assesses data quality, not model-generated reasoning quality at inference time. Without measuring whether the model's CoT traces are factually correct independent of final answer correctness, it remains unclear whether the reasoning is genuinely interpretable or merely formatted to appear so.

- **The Interaction category shows a recall decrease with CoT.** Section A.3.7 reports a -0.21% recall drop for Interaction with CoT (Table 14), attributed to occlusion and merged bounding boxes. This suggests the systematic per-candidate evaluation in CoT may be a liability for multi-entity relationship reasoning where context across candidates matters. This limitation deserves more prominence than a detailed appendix analysis.

### Trivial:

- The Reasoning subset had the highest removal rate (7.6%, Table 1) during data filtering, potentially indicating that the most complex reasoning cases are systematically underrepresented in the final training data. The impact is likely small given overall performance but worth noting.

## Nice-to-Haves

- A consistency reward term in GRPO that penalizes mismatches between the number of objects in the reasoning trace and the final answer output, directly addressing the inconsistency acknowledged in A.4.2.
- Quantitative analysis of reasoning trace correctness at inference time (e.g., human evaluation of model-generated CoT on a held-out set), not just data quality evaluation.
- Sensitivity analysis of detector recall on downstream performance (e.g., by artificially degrading the candidate set).
- Evaluation on additional standard REC benchmarks (RefCOCO, RefCOCO+) with in-domain training to contextualize performance against the broader REC literature.
- Ablation on the number of CoT stages to justify the three-stage Planning–Action–Summarization structure over alternatives.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Criticism that the IoU=1 reward in Equation 5 is unrealistic.** This misunderstands the architecture: the model selects from input box hints, not generating new coordinates. Requiring exact match with a hint box is the natural and correct design for a retrieval-based approach, not an artificial constraint. The predicted box must be one of the provided candidates.

- **Criticism demanding missing related works comparisons.** Per hard rules, I cannot confirm the existence of uncited works and should not flag their absence.

- **Reproducibility nitpicks about GRPO convergence criteria, unspecified ε value, and KL penalty implementation details.** The paper specifies β=0.04, learning rate, rollout samples, batch size, and training cost. The ε clipping parameter follows standard PPO convention. Convergence based on reward plateau is standard practice. These are not material reproducibility gaps.

- **Demand for confidence intervals / error bars on results.** Single-run evaluation is the norm for large MLLM benchmarks; requesting confidence intervals is a nice-to-have, not a core weakness.

- **Formatting/style nitpicks** about split training details between main text and appendix, or garbled figure references (parser artifacts).

## Novel Insights

The most insightful observation across the reviews concerns the tension at the heart of this paper's contribution: the CoT reasoning paradigm provides its greatest benefit exactly where it is hardest to validate. The 13.8-point rejection improvement is the paper's strongest result, and it stems from the model's ability to systematically evaluate and reject all candidates—a capability that emerges naturally from the Action phase's exhaustive per-candidate verification. However, this same exhaustive approach becomes a liability for Interaction-type expressions requiring cross-candidate relational reasoning, where context between candidates (not just within each candidate) determines correctness. This suggests a fundamental trade-off: the per-candidate decomposition that enables rejection and interpretability may conflict with the holistic reasoning needed for relational expressions. A promising direction would be augmenting the Action phase with cross-candidate comparison steps, or introducing a separate "relational verification" stage in the CoT structure.

## Suggestions

- **Quantify and address reasoning–answer inconsistency.** Measure how often the model's CoT trace and final output diverge on a held-out test set, and report this as a core metric alongside accuracy. If the rate is low, this strengthens the verifiability claim; if it is non-trivial, it must be discussed prominently.

- **Add detector failure analysis.** Run Rex-Thinker with both ground-truth boxes and detector-predicted boxes as candidates on a shared test subset, and report the performance gap. This directly quantifies the detector dependency ceiling and provides practical guidance on when the method can be trusted.

- **Tone down zero-shot generalization claims or reframe them.** The abstract should distinguish between zero-shot transfer (which is meaningful but below supervised baselines) and fine-tuned adaptation (which achieves SOTA). A phrase like "competitive zero-shot transfer and state-of-the-art fine-tuned performance" would be more accurate.

- **Move inference speed discussion to the main text.** A 6× latency increase is a significant practical consideration that readers should encounter alongside the accuracy improvements, not only in an appendix.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 4.0, 6.0]
Average score: 5.0
Binary outcome: Accept
