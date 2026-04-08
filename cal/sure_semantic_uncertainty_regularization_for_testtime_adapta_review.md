=== CALIBRATION EXAMPLE 8 ===

# Final Consolidated Review
## Summary

SURE proposes a test-time adaptation framework for vision-language models that regularizes predictions through a dynamically evolving Prototype-Reliability Graph (PRG). PRG encodes both semantic affinity (from textual prototypes) and class-wise reliability (from temporal statistics of pseudo-label confidences), enabling structured propagation of reliable information while suppressing error amplification from noisy pseudo-labels. The method achieves state-of-the-art results across ImageNet variants and 10 cross-dataset benchmarks using ResNet-50 and ViT-B/16 backbones.

## Strengths

- **Principled structured regularization for TTA**: Unlike prior TTA methods that treat classes independently and rely on per-instance confidence, SURE explicitly models inter-class semantic dependencies and modulates them by temporal reliability. The ablation in Table 4 cleanly validates each component: the graph alone (+Graph w/o Rel) can hurt, but reliability gating (+Graph + Rel) rescues it, confirming the design is synergistic rather than additive.

- **Effective closed-loop adaptation design**: The co-evolution of predictions, prototypes, and graph structure (Section 4.3) is a coherent framework. Figure 4 provides tangible evidence that PRG suppresses spurious connections (e.g., 'Television'→'Tabby' weight drops from 0.75 to 0.13) while preserving valid semantic clusters—a concrete demonstration that the reliability mechanism works as intended.

- **Strong efficiency–performance trade-off**: Table 3 shows SURE achieves the highest accuracy (66.23%) while being 10× faster than TPT and only ~3× slower than the simplest baseline (BCA). The lightweight per-sample design (Algorithm 1) avoids gradient computation and multiple augmented views, making it genuinely deployment-friendly relative to entropy-based methods.

## Weaknesses

### Major:

- **Marginal gains over strong baselines raise statistical significance concerns**: On ViT-B (Table 1), SURE achieves 66.23% vs. ZERO's 66.10% (+0.13%) and DPE's 65.93% (+0.30%). The reported standard deviation for SURE is ±0.11–0.16% (Appendix A.3), meaning some gains over individual baselines fall within 1–2 standard deviations. Crucially, **no standard deviations are reported for any baseline**, making it impossible to assess whether these improvements are statistically meaningful. The claim of "consistent outperformance" (Abstract) needs either significance tests against baselines or acknowledgment that some margins are narrow. On RN50, the gains are more substantial (+4.79% over CLIP, +0.31% over DPE), which partially mitigates this concern but does not eliminate it for the ViT-B setting that dominates modern VLM usage.

- **Missing direct comparison with PROGRAM (Sun et al., 2024)**: PROGRAM is the most directly related prior work—it is a graph-based TTA method explicitly discussed in Section 2. The paper differentiates SURE from PROGRAM conceptually (reliability-driven topology, VLM-specific design), but never compares against it empirically. Without this comparison, it is impossible to determine whether SURE's gains come from the reliability-weighted graph specifically or from other design choices (e.g., prototype initialization, prompt ensemble). A graph-based TTA baseline is essential for validating the core contribution.

### Minor:

- **Inconsistent temporal modeling between prototype updates and reliability estimation**: Equation 12 uses a cumulative moving average for prototype updates (weight 1/N_i for new samples), while Equation 13 uses a fixed-size sliding window for reliability statistics. This means prototypes are anchored to early test samples with diminishing plasticity, whereas reliability tracks only recent history. If the test distribution drifts over time, the cumulative prototype update could lag behind the reliability estimate, creating a mismatch. The paper should justify this asymmetry or adopt consistent temporal modeling.

- **Reliability score formulation is heuristic and untested against alternatives**: The specific form R_j = μ_j · (1 − σ_j/σ_max) is presented with "information-theoretic intuition" but no derivation or comparison against alternatives (e.g., entropy-based, confidence-only, variance-only). The ablation in Table 4 validates that reliability gating helps, but does not isolate whether this particular functional form is necessary versus simpler alternatives. Given that σ_max = 0.5 is a hard hinge (values of σ_j > 0.5 produce a negative term before clipping), a smoother formulation might be more robust.

- **Semantic similarity matrix S drifts from textual priors as prototypes are updated**: Equation 3 computes S from current prototypes t_i, but these are updated with visual features via Equation 12. The paper claims edges encode "semantic affinity derived from textual prototypes" (Abstract), yet S evolves toward a visually-driven topology. Under systematic domain bias (e.g., occluded dogs resembling cats), the graph structure itself could drift, undermining the semantic prior. The paper should either compute S from frozen text embeddings (as a stable prior) or explicitly acknowledge and analyze this drift.

- **Hyperparameter sensitivity varies across domain types**: Figure 3 shows optimal k differs between natural shifts (k=4) and cross-dataset (k=3), and optimal θ varies from ~0.3 to ~0.5. While performance is smooth around optima, the "generalizable" claim (Abstract) is somewhat undermined if different shift types benefit from different hyperparameters. The paper should clarify whether a single hyperparameter setting works reasonably across all benchmarks or whether per-domain tuning is needed.

### Trivial:

- **5-class toy visualization (Figure 4)**: While illustrative, graph dynamics in a 5-node graph (dense, highly connected) differ qualitatively from a 1000-node sparse graph. A visualization of degree distribution or sparsity patterns for the full ImageNet graph would be more convincing.

- **Memory overhead of adjacency buffer**: Storing L=5 matrices of size C×C for C=1000 is ~20MB—manageable but not discussed. For C=10,000+ this becomes prohibitive.

## Nice-to-Haves

- **Evaluation on larger VLMs (ViT-L, ViT-H)**: All results use RN50 and ViT-B. Testing on current-scale backbones would strengthen generality claims.

- **Long-term adaptation stability analysis**: Plot accuracy vs. test-step to verify the reliability mechanism prevents error accumulation over extended streams, which is the core failure mode SURE targets.

- **Adaptive weighting between local and graph predictions**: Equation 10 uses a simple sum. A learned or confidence-gated combination could improve robustness when the graph is noisy early in adaptation.

- **Breakdown of per-component latency**: Table 3 reports total time but doesn't profile graph construction vs. propagation vs. prototype updates, which would inform optimization priorities.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: "Missing evaluation on DTD, Cars, fine-grained datasets"** (from Spark Finder) — Factually wrong. Table 2 explicitly evaluates on DTD, StanfordCars, Flowers102, and FGVC-Aircraft.

- **Weakness: "GraphAdapter not compared"** (from Harsh Critic) — GraphAdapter requires labeled supervision and offline training, which the paper explicitly scopes out (Section 2). Comparing a TTA method with a supervised adaptation method is unfair and scope creep.

- **Weakness: "Parser artifacts in Equation 1"** (from Harsh Critic) — Formatting nitpick; the prompt itself notes these are parser artifacts.

- **Weakness: "Batch size sensitivity"** (from Spark Finder) — Algorithm 1 processes samples individually (per-sample loop), so batch size does not directly affect the method's operation. This is not a relevant concern.

- **Weakness: Transferable weaknesses from unrelated papers** (from Harsh Critic's "Additional transferable weaknesses") — These are about CH-divnorm normalization, concept bottleneck models, and segmentation uncertainty quantification. They are from completely different papers and not applicable here.

- **Weakness: "Missing comparison with recent graph-based adaptation methods beyond PROGRAM"** (from Spark Finder) — The spark finder could not specify which methods, and without confirming their existence this is speculative.

- **Weakness about "EuroSAT performance degradation"** — Not substantiated by the data; SURE achieves 53.60% vs DPE's 55.79% on ViT-B (Table 2), which is actually a slight decrease, but this is one dataset out of ten and SURE wins on average. Worth noting but not a standalone weakness.

## Novel Insights

SURE's ablation (Table 4) reveals a non-obvious finding: adding semantic graph structure without reliability gating can *hurt* performance (ImageNet-A drops -0.24%), making the reliability mechanism not merely an enhancement but a *necessity* for safe graph-based TTA. This suggests that the common intuition—semantic structure should always help—is false under distribution shift, and that the real contribution is the *gating* of structure by uncertainty rather than the structure itself. This has implications beyond SURE: any method that injects semantic priors into TTA must account for the possibility that those priors become misleading under domain shift.

## Suggestions

- **Run significance tests (paired t-test or bootstrap) against DPE, ZERO, and BCA** using at least 3–5 seeds for baselines. This is the single most impactful change for strengthening the empirical contribution, given that key gains are <0.5%.

- **Add PROGRAM as a baseline** in Tables 1 and 2. Even if it was designed for uni-modal classifiers, adapting it to the VLM setting (or reporting its original numbers where applicable) would directly validate the reliability-driven topology contribution.

- **Ablate the reliability formulation**: Compare R_j = μ_j · (1 − σ_j/σ_max) against simpler alternatives (μ_j only, 1 − σ_j only, entropy of the confidence buffer) to justify the specific multiplicative form.

- **Clarify the calibration framing**: The method name ("Uncertainty Regularization") and Abstract imply improved uncertainty estimation, but Table 10 shows SURE's ECE (7.48) is worse than unadapted CLIP (6.29). The paper should explicitly state that the regularization claim is *relative to adapted methods*, not to the zero-shot baseline, and discuss why some calibration degradation is inherent to any adaptive method.

- **Report whether S in Equation 3 is computed from frozen or updated prototypes** — if from updated prototypes, add an ablation comparing frozen-text S vs. evolving S to quantify semantic drift's impact on the graph.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 6.0, 4.0, 4.0]
Average score: 4.4
Binary outcome: Reject
