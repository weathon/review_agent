=== CALIBRATION EXAMPLE 20 ===

# Final Consolidated Review
## Summary
This paper introduces SURE, a test-time adaptation (TTA) framework for vision-language models that regularizes predictions via a dynamically evolving Prototype-Reliability Graph (PRG). The PRG jointly encodes semantic affinity from text embeddings and class-wise reliability based on temporal confidence stability, aiming to propagate reliable information and suppress noise. The method is evaluated across multiple domain-shift benchmarks, showing consistent improvements over prior TTA methods.

## Strengths
- **Novel integration of semantic structure and temporal reliability**: The core idea of a dynamic graph that fuses semantic similarity with time-varying prediction stability for TTA is a fresh and well-motivated contribution. It explicitly addresses the limitation of treating pseudo-labels independently, a clear advance over prior entropy or prototype-only methods.
- **Extensive and rigorous empirical evaluation**: The paper provides a comprehensive assessment across 15 datasets covering natural distribution shifts (ImageNet variants) and cross-dataset generalization, using two VLM backbones and comparing against numerous strong recent baselines (e.g., TPT, DPE, ZERO, BCA). The results show consistent gains, particularly on challenging datasets like ImageNet-Sketch and fine-grained domains.
- **Clear and informative ablation study**: Table 4 cleanly isolates the contribution of each component (prototype updating, graph structure, reliability weighting, logit propagation), demonstrating that the full system's gains stem from the synergistic combination of these elements.
- **Practical efficiency**: SURE achieves strong performance with low inference latency (0.067s/sample for ViT-B), making it significantly faster than gradient-based methods like TPT and competitive with lightweight baselines, enhancing its suitability for streaming deployment.

## Weaknesses
### Major:
- **Lack of empirical comparison to the most relevant graph-based baseline**: The paper explicitly differentiates SURE from PROGRAM (Sun et al., 2024), a graph-based TTA method, claiming advantages in "reliability-driven topology" and "VLM-specific design." However, no direct experimental comparison is provided, leaving these claims unsubstantiated. This omission undermines the paper's novelty argument.
- **Reliability mechanism lacks validation against ground-truth correctness**: The core reliability score \(R_j = \mu_j \cdot (1 - \sigma_j / \sigma_{\text{max}})\) is a heuristic based on confidence statistics. The paper does not analyze whether high \(R_j\) actually correlates with higher accuracy for samples assigned to class \(j\). Without this validation, the foundational assumption that "reliable" classes are semantically correct remains unverified.
- **Insufficient quantitative evidence for the central claim of error suppression**: The paper argues that SURE "prevents error amplification" and "suppresses unreliable classes," but provides only a qualitative visualization on a hand-picked set of five classes (Fig. 4). There is no quantitative analysis tracking error propagation (e.g., how often incorrect pseudo-labels are reinforced via the graph) or measuring the change in error rates for low-reliability classes over time.
- **Statistical significance of reported gains is unclear**: While the paper reports standard deviations for SURE across runs (Tables 7, 8), it does not provide comparable confidence intervals or significance tests for the baselines. The marginal gains over strong methods like ZERO (e.g., 66.23% vs. 66.10% on ViT-B natural shifts) are highlighted as consistent improvements, but without statistical grounding, it is difficult to judge their meaningfulness.

### Minor:
- **Increased hyperparameter burden**: SURE introduces several new hyperparameters (confidence threshold \(\theta\), neighbor size \(k\), buffer length \(L\), \(\sigma_{\text{max}}\)). Although sensitivity analysis is provided, the need to tune these parameters adds complexity compared to simpler baselines, and the paper does not discuss a strategy for setting them in a fully unsupervised test-time scenario.
- **Limited discussion of failure modes and limitations**: The method assumes that classes with stable high confidence are reliable. However, under severe domain shift, a model could become consistently but incorrectly confident. The paper does not analyze such pathological cases or discuss scenarios where SURE's graph-based regularization might fail (e.g., when semantic similarity is misleading).
- **Marginal improvements on some benchmarks**: While average improvements are consistent, absolute gains over the strongest baselines on certain benchmarks (e.g., ImageNet-ViT-B natural shifts) are very small. The paper could better delineate the conditions under which SURE's advantages are most pronounced versus negligible.

### Trivial:
- **Clarity of graph propagation mechanics**: The description of logit regularization (Eqs. 9-10) is somewhat terse. A more intuitive explanation or a small concrete example could improve accessibility.

## Nice-to-Haves
- **Evaluation on gradual or sequential distribution shifts**: Testing on benchmarks with continuously changing domains (e.g., increasing corruption severity) would further demonstrate the dynamic adaptation capabilities claimed.
- **Theoretical motivation for the reliability heuristic**: A more formal connection between the proposed \(R_j\) and established uncertainty measures (e.g., entropy, variance of a Beta distribution) would strengthen the method's foundation.
- **Analysis of prototype drift over time**: Tracking the cosine similarity between original text-based prototypes and adapted prototypes could reveal whether the updates cause beneficial alignment or harmful semantic drift, especially for low-reliability classes.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Claim that the reliability score is "unprincipled" and lacks theoretical foundation (Harsh Critic)**: While the heuristic nature is a valid concern, the paper explicitly states it is a "practical proxy for inverse uncertainty." Demanding a full theoretical derivation is outside the standard scope for an empirical systems paper in this area. This is softened to a "Nice-to-Have."
- **Criticism that graph propagation is merely "simple linear smoothing" and not "belief propagation" (Harsh Critic)**: This is a semantic nitpick. The paper's interpretation as one-step belief propagation in a Markov random field is a reasonable conceptual framing for the weighted averaging operation.
- **Request for per-dataset hyperparameter robustness analysis (Harsh Critic)**: The paper already includes a hyperparameter sensitivity analysis (Fig. 3) showing performance trends. Demanding per-dataset robustness curves is an excessive burden.
- **Complaint about missing inference time experimental details (Harsh Critic)**: The comparison in Table 3 provides a reasonable high-level efficiency assessment. Detailed hardware specs and batch sizes are impractical to include for all baselines and are not required for a meaningful relative comparison.
- **Suggestion to add "confidence intervals for large-scale benchmarks" (Spark Finder)**: While statistical significance is important (raised as a major weakness), providing full confidence intervals for all baselines across 15 datasets is not standard practice in the field. The request for significance testing is addressed in the major weaknesses.

## Suggestions
- **Add a direct empirical comparison to PROGRAM** (Sun et al., 2024) to substantiate the claimed advantages of reliability-driven topology and VLM-specific design.
- **Conduct an analysis correlating the reliability score \(R_j\) with actual class-wise accuracy** on the test stream to validate that the heuristic effectively identifies trustworthy classes.
- **Include a quantitative error propagation analysis**, such as tracking the fraction of incorrect pseudo-labels that are reinforced through the graph over time or measuring error rate changes for classes with low vs. high reliability.
- **Perform statistical significance testing** (e.g., paired t-tests across runs) for key comparisons to bolster claims of consistent outperformance.
- **Move the calibration analysis (ECE) from the appendix to the main experimental section**, as it directly supports the method's ability to maintain trustworthy predictions.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 6.0, 4.0, 4.0]
Average score: 4.4
Binary outcome: Reject
