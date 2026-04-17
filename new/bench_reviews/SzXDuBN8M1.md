Now I have enough calibration information. Let me write the final review.

## Summary

TD-JEPA introduces a temporal-difference (TD) variant of latent-predictive representation learning for zero-shot unsupervised RL. Rather than predicting one-step latent transitions from on-policy data, it conditions multi-step predictors on policy embeddings z, enabling off-policy, multi-policy training from offline reward-free data. The method trains separate state (ϕ) and task (ψ) encoders, policy-conditioned predictors Tϕ and Tψ, and parameterized policies πz end-to-end, with the predictor approximating successor features for zero-shot policy optimization. Theoretical analysis in a linear/tabular setting shows non-collapse guarantees and that the latent-predictive gradients match those of explicit successor measure approximation. Empirically, TD-JEPA matches or outperforms state-of-the-art zero-shot baselines across 65 tasks from ExoRL and OGBench, particularly on pixel-based domains.

## Strengths

- **Novel and well-motivated formulation**: The core idea of connecting TD learning with policy-conditional latent prediction for multi-step successor feature approximation is a genuine conceptual contribution. The off-policy formulation (Eq. 7, 9) that avoids on-policy sampling from successor measures is clean and practical, providing a meaningful departure from prior JEPA-style approaches that are either one-step or on-policy.

- **Strong empirical performance with breadth**: Evaluation across 65 tasks spanning 13 datasets (locomotion, navigation, manipulation; proprioceptive and pixel-based) is substantial. TD-JEPA consistently performs at or near the top, particularly on DMCRGB (628.8 vs. next-best 582.4) and DMC proprioception (661.2 vs. 645.4). The probability-of-improvement analysis (Fig. 2) provides a principled aggregate comparison showing TD-JEPA as the most consistently strong method.

- **Fair experimental protocol with transparent baseline improvements**: The authors re-implement baselines with shared architecture and explicit state encoders, reporting that this substantially improves existing methods (1.3×–2.4× over originally reported results). This level of fair comparison is commendable and increases confidence in the claimed improvements.

- **Useful ablations**: The comparison with BYOL/BYOL-γ variants (Fig. 3, left) directly tests the contribution of multi-step policy-conditional prediction vs. one-step/behavioral prediction, and the symmetric vs. asymmetric encoder comparison (Fig. 3, right) provides concrete architectural guidance. The fine-tuning experiments (Fig. 4) show practical value of learned representations.

- **Theoretical contribution**: The gradient-matching arguments (Thm. 1, 3) establishing that latent-predictive gradients align with explicit successor measure approximation losses is a valuable formal insight. The non-collapse guarantee (Thm. 2) for the "doubly latent-predictive" TD setting extends existing non-collapse results to a genuinely more complex case.

## Weaknesses

### Major

- **Theory-practice gap is significant and understated**: Theorems 1–4 assume tabular states, linear predictors, symmetric transition kernels (A3), and uniform state distributions (A2). Assumption A3 (symmetric P^πz) is particularly unrealistic—most RL domains have irreversible dynamics (locomotion with friction, gravity; manipulation with contact). The paper acknowledges assumptions "can be relaxed at the price of more involved proofs" but does not provide these relaxations, nor discuss *qualitatively* what happens when they fail. For the actual deep RL algorithm, these guarantees provide at best an analogy. The abstract and introduction frame the theoretical results as validating that TD-JEPA "learns encoders that capture a low-rank factorization of long-term policy dynamics" and "recovers their successor features in latent space" — this overstates what is proved, which applies only under assumptions far removed from the experimental setting.

- **Core conceptual claim lacks direct empirical validation**: The paper's central claim is that Tϕ learns something that approximates successor features, enabling zero-shot optimization for rewards in the span of ψ. However, no experiment directly verifies this: there is no measurement of ‖Tϕ(ϕ(s),a,z) − MC estimate of F_ψ^{πz}(s,a)‖, no value prediction error vs. ground truth, and no test of linear generalization to novel reward combinations. Good performance on a finite set of benchmark tasks, while encouraging, is compatible with alternative explanations (e.g., the latent policies πz aligning well with the task distribution). This is a significant evidential gap for a paper that makes the successor-feature mechanism its core conceptual contribution.

- **Missing ablations on key design parameters**: Several important design choices are not ablated: (1) the number/dimension of latent policies |Z| and z, which controls the resolution of successor measure approximation; (2) the covariance regularization coefficient λ — Theorem 2 claims non-collapse, but the algorithm includes explicit regularization, and without a λ=0 ablation it is unclear whether the theoretical guarantee or the regularizer prevents collapse in practice; (3) no analysis of off-policy stability or sensitivity to dataset coverage. These omissions make it harder to assess robustness and understand what drives performance.

### Minor

- **Mixed per-domain results obscured by aggregation**: While aggregate metrics favor TD-JEPA, per-domain results in Table 1 show non-trivial weaknesses. On OGBench proprioception, PSM achieves 51.60 on antmaze-me vs. TD-JEPA's 20.20; on cube-double, TD-JEPA achieves only 3.00%. The paper does not analyze when or why TD-JEPA underperforms, which limits the practical understanding of the method's scope.

- **Modest gains from asymmetric encoder design**: The ablation (Fig. 3, right) shows the symmetric variant (shared ϕ=ψ) is quite competitive, sometimes even preferable. Given that the asymmetric design is the paper's more architecturally complex contribution, the empirical case for separate encoders is not strong enough to fully justify the added complexity.

- **No comparison of computational cost**: TD-JEPA trains four networks (ϕ, ψ, Tϕ, Tψ) plus a policy and their target networks. No wall-clock time, memory, or FLOPs comparison is provided against simpler baselines like FB. Since computational overhead is a practical concern for foundation model pre-training, its absence weakens the practical assessment.

## Nice-to-Haves

- Direct measurement of successor feature fidelity (comparing predictor outputs to Monte Carlo SF estimates) would validate or falsify the paper's core mechanism claim.
- Ablations on |Z| dimension and λ=0 to connect theory to practice.
- Visualization of learned representations (t-SNE/PCA of ϕ and ψ) to investigate what the separate encoders capture.
- Analysis of failure cases (e.g., cube-double, antmaze-me) to understand method limitations.
- Discussion of what the theoretical results *qualitatively* imply for the neural network setting, even without formal extensions.

## Removed Points

- **"BYOL/ICVF baselines are retrofitted and thus not a fair comparison"** (Harsh Critic point 4): The paper explicitly marks these with ⋆ and clearly states they are representation methods adapted to a zero-shot framework, not standard zero-shot algorithms. The comparison is designed to isolate representation quality, and the paper is transparent about this. This is not an unfair comparison — it's a well-motivated controlled experiment. The paper also re-implements all methods with shared architectures for fairness. *Removed because the paper explicitly addresses this and the comparison is informative for the claimed contribution about representation quality.*

- **"Expressiveness of continuous latent z may be insufficient for combinatorial task structure"** (Human Finder point 2): This is a generic concern applicable to any successor-feature method (FB, HILP, PSM all use continuous z). TD-JEPA shows strong empirical results across diverse tasks, providing indirect evidence that the parameterization is sufficient for the tested benchmarks. This is a scope concern, not a weakness of this specific paper. *Removed because this is a generic concern about the entire family of zero-shot SF methods, not specific to TD-JEPA.*

- **"Evaluation limited to DMC and OGBench"** (Human Finder point 3): The paper evaluates on 65 tasks across 13 datasets covering locomotion, navigation, manipulation with both proprioception and pixels. This is a broader evaluation than most comparable papers. *Removed because this is a generic "add more experiments" request that does not identify a specific flaw.*

- **"Presentation accessibility for readers unfamiliar with FB/Successor Features"** (Human Finder point 6): This is a formatting/style nitpick. The paper provides sufficient preliminaries (Sec. 2) and related work (Sec. 5). *Removed as formatting nitpick.*

- **"No ablation on covariance regularization λ"** is kept as a minor point but not elevated to fatal — Theorem 2's guarantee is about the continuous-time limiting case with optimal predictors; the algorithm uses explicit regularization as practical stabilizer. Still, an λ=0 run would illuminate whether collapse is a real concern.

## Novel Insights

The gradient-matching perspective (Thm. 1, 3) — that latent-predictive losses produce the same representation gradients as explicit successor measure approximation losses — is a genuinely novel unifying insight. It provides a principled reason why self-predictive representations are useful for value-based RL beyond the standard "good representations capture dynamics" intuition: they specifically optimize for representing the information needed for successor feature decomposition. This connection between the JEPA paradigm and successor measure approximation is a conceptual bridge that, to my knowledge, has not been made before. However, this insight operates within the strong assumptions of the theoretical framework; its empirical validity in the deep RL setting remains an open question.

## Suggestions

- Add a direct experimental validation that the predictor approximates successor features: measure ‖Tϕ(ϕ(s),a,z) − MC-F_ψ^{πz}(s,a)‖ during training, and show it decreases. This would directly validate the core mechanism claim.
- Add λ=0 and |Z| ablations to connect theory to practice and establish the sensitivity of the method to these parameters.
- Include a brief discussion of failure cases (domains where TD-JEPA underperforms) — even speculative analysis would improve the paper.
- In the abstract/introduction, qualify theoretical claims more carefully: instead of "learns encoders that capture a low-rank factorization," write "in an idealized linear/tabular setting, learns encoders that capture a low-rank factorization; we demonstrate empirically that the resulting algorithm performs well on pixel-based benchmarks."

## Score and Decision

**Calibration**: I compared against papers in this family:
- **Fast Imitation via BFM** (Accept spotlight, scores 8/6/8/8): Strong application of successor-feature zero-shot RL to imitation, well-motivated, extensive experiments, similar theoretical grounding from prior work.
- **Bridging State and History Representations** (Accept poster, scores 8/8/8/3): Strong theoretical unification of self-predictive RL representations with solid experiments, but acknowledged limitations in empirical conclusiveness.
- **PSM** (Reject, scores 5/8/6/8): Novel successor-measure idea but limited experiments and poorly written.
- **DVFB** (Accept poster, scores 6/8/6): Novel dual-value FB variant with empirical improvements, similar theory-practice gap concerns.
- **CDPC** (Accept poster, scores 6/8/6/8): TD version of contrastive predictive coding — similar conceptual move (applying TD to existing framework), with strong empirical results.
- **MR.Q** (Accept spotlight, scores 8/8/8/6): General-purpose RL with model-based representations, strong theory-practice gap but excellent empirical breadth.
- **FB-CPR** (Accept poster, scores 6/8/6/6): Extension of FB to humanoid whole-body control, applied contribution with similar zero-shot RL foundation.

TD-JEPA has a genuine conceptual contribution (TD-based policy-conditional latent prediction for SF approximation), strong and broad empirical results (especially on pixels), and meaningful theoretical insights (gradient matching). However, the theory-practice gap is substantial and the core mechanism claim (SF approximation) is not directly validated. These are notable but not fatal weaknesses. Compared to CDPC (which made a similar "apply TD to existing predictive coding" move and was accepted) and DVFB (similar FB-family extension, accepted), TD-JEPA has broader empirical evaluation and more theoretical depth. Compared to Fast Imitation via BFM (spotlight), TD-JEPA is less polished in connecting theory to practice and lacks direct validation of its mechanism. The paper is above the acceptance threshold but with clear room for improvement.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>