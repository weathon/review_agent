Now I have all the information I need. Let me write the consolidated review.

## Summary

TULiP proposes a post-hoc OOD detection method grounded in linearized training dynamics (NTK framework). The key idea is to bound the effect of hypothetical mid-training perturbations on the converged network, yielding an uncertainty score that can be estimated via weight perturbation at test time without accessing training data. The method demonstrates strong near-OOD performance on OpenOOD benchmarks and generalizes to non-classification tasks via surrogate ensemble construction.

## Strengths

- **Novel conceptual framework for epistemic uncertainty**: The idea of modeling epistemic uncertainty as the sensitivity of converged networks to hypothetical mid-training perturbations, then bounding this via the NTK gradient-embedding distance (Theorem 3.1, Eq. 5–6), is genuinely novel and provides an appealing conceptual connection between training dynamics and OOD detection.

- **Strong near-OOD performance**: TULiP achieves the best or second-best near-OOD AUROC on CIFAR-10 (89.67%) and ImageNet-200 (83.84%) benchmarks (Table 1), providing a genuine practical contribution for the harder near-OOD detection setting.

- **Practical advantages over training-data-dependent methods**: As a post-hoc method requiring no training data, TULiP avoids the storage and computation overhead of methods like ViM and MDS (Table 1, †-marked), while maintaining competitive performance. The paper notes TULiP is ~3× faster than ViM's extraction step (Sec. 5.2).

- **Generality beyond classification**: The surrogate ensemble construction (Alg. 1, lines 14–18) produces posterior samples rather than relying on logit-specific transformations, enabling extension to regression (Fig. 2a) and composition with logit-based methods like GEN (Table 1, TULiP+GEN).

- **Architecture generalization**: Experiments across MobileNet, VGG, and RegNet architectures (Fig. 3) demonstrate that TULiP's advantages persist beyond the standard ResNet backbone.

## Weaknesses

### Fatal
None.

### Major

- **Theory-practice disconnect undermines the "theoretically-driven" framing**: The paper's central claim is to be a "theoretically-driven post-hoc uncertainty estimator" (abstract), but Algorithm 1 does not compute the bound from Theorem 3.1. The chain from Theorem 3.1 → Algorithm 1 requires multiple heuristic substitutions that each break the theoretical connection: (i) θ_{t_s} is replaced with 0 (Sec. 4, acknowledged), (ii) 𝔼_x[Θ(x,x)] is dropped because it is "intractable and irrelevant to z" (Sec. 4.3), (iii) the constant K in Lemma 3.2 is replaced by the hyperparameter λ (Sec. 4.2: "λ acts as a proxy to the constant K"), and (iv) layer-wise scaling Γ (Eq. 12) is inserted—the authors themselves call it "highly heuristic" (Sec. 4.1). Most critically, the ablation study confirms that layer-wise scaling is *essential* ("our method failed to achieve consistent performance across various datasets without layer-wise scaling," Sec. 5.2), yet it has no theoretical justification. The component that makes the algorithm work is the very component that severs its connection to the theory. The paper is better described as *theory-inspired* than *theoretically-driven*, and the abstract's claim that the bound "results in an uncertainty score computable by perturbing model parameters" implies a more direct connection than exists.

- **The closeness assumption (Eq. 8) is load-bearing but only empirically verified**: Equation 8 is the key step allowing replacement of the intractable inf_{x∈X} with a tractable expectation. The paper provides only Fig. 1d as empirical support, using 256 samples from one ResNet-18 model on ImageNet. There is no proof, no conditions under which it holds or fails, and no systematic study across architectures, datasets, or perturbation scales. If this assumption fails, Proposition 3.3 collapses and the theoretical chain breaks. A core theoretical step cannot rest on a single empirical figure.

- **The theoretical bound is never validated on practical architectures**: Figure 2 validates Theorem 3.1 on synthetic data using infinite-width networks and exact NTK computation. The experiments in Section 5 use ResNet-18/50—finite-width networks where the NTK evolves during training (which motivates the layer-wise scaling in the first place). No experiment shows that the bound from Eq. 5 or Eq. 9 bears any quantitative relationship to actual epistemic uncertainty on the architectures used in practice. Without this bridge, the theory-to-practice connection is entirely unsupported by evidence.

### Minor

- **"State-of-the-art" claim is not uniformly supported**: On ImageNet-1K (ResNet-50), ASH outperforms TULiP on both near (78.17 vs 77.52) and far (95.74 vs 88.03) AUROC. On CIFAR-100 near-OOD, GEN slightly outperforms TULiP on AUROC (81.31 vs 81.29). The claim of "state-of-the-art performance, particularly for near-distribution samples" (abstract) is directionally correct but overstated for these benchmarks. The paper's discussion of ASH's advantage via "redundant representations" is plausible but not rigorously established.

- **Figure 3 comparison is too narrow for the architectural generalization claim**: Fig. 3 only compares TULiP against MLS and ODIN across architectures, omitting EBO, GEN, and ASH. If the claim is that TULiP generalizes across architectures *and* remains superior, the comparison should include the strongest baselines.

- **Hyperparameter sensitivity and near/far trade-off**: Fig. 4 shows a significant near/far trade-off, and the optimal ε varies substantially across datasets. The method requires ID validation data for tuning, which is a practical requirement not shared by all baselines (e.g., MLS requires no tuning).

- **The constant C in Theorem 3.1 may be vacuous in practice**: C = α·Θ̄_X^{1/2}·(e^{(T−t_s)Lλ_max} − 1)/λ_max has exponential dependence on (T−t_s)·L·λ_max, which could be enormous for realistic training durations. While C is absorbed into the heuristic λ in the algorithm, this means the algorithm's connection to the bound becomes even more tenuous for practical settings.

### Trivial
None.

## Nice-to-Haves

- A direct comparison between the theoretical bound (Eq. 5 or 9) and Algorithm 1's output on a small-scale realistic setup, to quantify how much the heuristic approximations degrade the bound.

- A formal analysis or more systematic empirical study of when the closeness assumption (Eq. 8) holds and when it fails.

- Comparison with a simple weight-perturbation baseline (Gaussian noise perturbation + prediction variance as OOD score, without the bound estimation, variance matching, or entropy computation) to isolate what the theoretical machinery adds over naive perturbation.

- Per-sample qualitative analysis showing when TULiP succeeds vs. fails on near-OOD samples that other methods miss.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh critic: "ASH failed with different weights" as selective reporting**: The harsh critic implies the paper selectively reports ASH's failure in the appendix while dismissing ASH's superior ImageNet-1K performance. However, the paper does mention this in the main text (Sec. 5.2) and it's a relevant observation about ASH's instability, not selective reporting. The criticism about ASH outperforming TULiP on ImageNet-1K is kept as a minor weakness above, but the "selective reporting" framing is removed.

- **Harsh critic: A4 is a strong assumption for cross-entropy loss**: The argument that "a perturbation large enough to produce meaningful uncertainty signals would plausibly prevent the perturbed network from interpolating" is speculative. The paper cites Zhang et al. (2017) and Du et al. (2019) for interpolation of overparameterized networks, and while the perturbed network's convergence is not guaranteed, this is an assumption common to much NTK analysis. This is a reasonable theoretical concern but not a fatal flaw.

- **Harsh critic: Squeezing factor γ can become zero**: While true that γ = 0 when S ≤ 0, the algorithm handles this via the max(S, 0) operation (line 14), which simply reduces to using the unperturbed prediction. This is a design feature, not a bug, though it could be discussed more explicitly.

- **Strength Finder: "Principled theoretical derivation" as a core strength**: While the theory provides conceptual motivation, calling the derivation "principled" conflicts with the verified major weakness that multiple heuristic steps break the theory-practice chain. The theoretical insight is kept as a strength (novel conceptual framework), but the claim of "principled derivation" from theory to algorithm is weakened.

- **Strength Finder: "Empirical validation of assumptions" (Fig. 1a-c)**: The empirical justification for layer-wise scaling (Fig. 1a-c) is suggestive but far from establishing that Eq. 11 approximates the true NTK during training. This is kept as partial support but downgraded from a full strength.

- **Harsh critic: Computational cost comparison is insufficient**: The paper does mention TULiP is ~3× faster than ViM and that it requires O(M) forward passes. While a systematic wall-clock comparison would be nice, this is a nice-to-have rather than a substantive weakness.

- **Harsh critic: Missing ablation isolating each heuristic modification**: This is a reasonable request but is more of a nice-to-have than a major weakness. The paper does provide an ablation on layer-wise scaling (which is the most important component).

## Novel Insights

The most insightful observation across the reviews is that TULiP exhibits a fundamental tension between its theoretical and empirical contributions: the theory (Theorem 3.1) provides an appealing conceptual connection between training dynamics and epistemic uncertainty via gradient-embedding distance, but the algorithm that works in practice operates in a regime where this theory's assumptions (lazy training, constant NTK, closeness) are known to be violated. The layer-wise scaling—explicitly acknowledged as heuristic—exists precisely to compensate for the failure of the NTK constancy assumption on practical networks, which means the algorithm's success may be *because of* rather than *despite* its departure from the theory. This suggests the empirical contribution (a well-tuned weight perturbation scheme with principled variance matching) may be more significant than the theoretical one, and the paper would be stronger if it embraced this rather than framing the theory as the primary contribution.

## Suggestions

- Reframe the contribution: Position TULiP as a *theory-inspired* method rather than *theoretically-driven*, and be explicit that Algorithm 1 is a heuristic approximation motivated by but not directly derived from the bound. This honest framing would strengthen rather than weaken the paper.

- Add one experiment bridging theory and practice: Even on a small-scale CNN (not infinite-width), compute Eq. 5 approximately and compare with Algorithm 1's output to quantify the gap introduced by the heuristic steps.

- Formalize the layer-wise scaling: Connect the |θ_l|^{-1/2} scaling to how parameter norms change during training in different layers (the empirical support in Fig. 1a-c is a starting point), or at minimum, provide a more systematic study of alternative scaling schemes.

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Red Teaming Game | BrtOzgElD7.md | 2.50 | Much worse: didn't formally define its game; TULiP has a correct theoretical framework within its assumptions |
| Extreme Distribution RL | BgZzJISvpY.md | 2.33 | Much worse: methodology "questionable without justifications"; TULiP has genuine theory and stronger experiments |
| Normalizing Flows OOD | 6Z8rZlKpNT.md | 3.40 | Worse: missing baselines and unclear method description; TULiP is more complete but has deeper theory-practice gap |
| Hierarchical Overlapping Clustering | oHSXRy29tj.md | 5.60 | Comparable: theory-practice disconnect with practical speed-up not preserving guarantees; TULiP has stronger empirical results but a more central theory-practice gap |
| Riemannian GD | 6YZmkpivVH.md | 5.50 | Comparable: theory relies on heuristics for implementation; similar pattern |
| SCALE | RDSTjtnqCg.md | 6.25 | Accepted poster; more incremental but no overclaimed theory; TULiP has more novelty but also more overclaim |
| ImOOD | am7BPV3Cwo.md | 5.75 | Rejected; theoretical bias analysis for OOD on imbalanced data; comparable depth |
| R-EDL | Si3YFA641c.md | 7.20 | Much stronger: theoretical grounding directly connects to method; TULiP doesn't reach this level of theory-practice alignment |
| NegLabel | xUO1HXz4an.md | 7.50 | Much stronger: clean theoretical analysis with SOTA empirical results; TULiP's theory is less rigorous |

TULiP sits in the medium band. It has genuine contributions (novel conceptual framework, strong near-OOD results) that clearly exceed the low-scoring anchors (2-3), but the theory-practice disconnect is significant enough—and central enough to the claimed contribution—that it falls below the high-scoring anchors (7+). The paper is comparable to theory-practice disconnect papers in the 5.5–5.6 range but with stronger empirical results, suggesting a score slightly above those. However, the overclaimed "theoretically-driven" framing pulls it back down. The most appropriate score is 5.5.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>