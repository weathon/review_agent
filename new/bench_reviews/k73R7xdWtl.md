Now let me look at the calibration papers mentioned by the reviewers.Now I have enough information to write the consolidated review. Let me verify one more key point about Algorithm 3's RFF resampling behavior.Now I have sufficient information to write the consolidated review. Let me synthesize.

---

## Summary

This paper proposes a contextual bandit framework—Shared-Context Kernel UCB (SCK-UCB) and its RFF-accelerated variant RFF-UCB—for the online selection of the best prompt-based generative model (T2I, image captioning, T2V) conditioned on the input text prompt. The key insight is that a single model with the highest average score need not be optimal for every prompt category, and an online learner exploiting prompt embeddings can improve over any fixed model selection. The authors establish O(√GT) regret bounds for both algorithms and demonstrate results across several experimental setups.

---

## Strengths

- **Timely and well-motivated problem.** The premise that generative model rankings vary across prompt categories is compelling, concretely motivated by Figure 1, and is a genuinely underexplored setting relative to standard best-model-identification literature.
- **Theoretical contribution.** The paper delivers formal regret bounds for both SCK-UCB (O(√GT)) and RFF-UCB. The extension of kernel-UCB regret analysis to the RFF approximation regime, where features are drawn adaptively, is a non-trivial analytical step.
- **Computational scalability.** The O(t³) → O(t) reduction via RFF is directly motivated and formally established in Lemma 2. The design choice of a fixed-size s×s gram matrix is clean.
- **Breadth of experimental coverage.** Experiments span T2I (Setups 1–3), image captioning (Setup 4), and T2V (Setup 5), with an adaptation scenario (Setup 2) that tests new arms and new prompt types—a practical setting not commonly addressed in bandit literature.
- **Outperforming One-arm Oracle.** In Setup 1, SCK-UCB-poly3 demonstrably outperforms the oracle that knows the single best average model, supporting the core claim that prompt-conditioned routing adds value.

---

## Weaknesses

### Fatal
*None that fully invalidate the core claim, though the major weaknesses below are collectively serious.*

### Major

- **RFF-UCB—the paper's primary efficiency contribution—fails in the real-model experiment.** Figure 2 shows RFF-UCB achieving OPR ≈ 0.55 and O2B ≈ 0 on real SD vs. PixArt-α, while SCK-UCB-poly3 reaches OPR ≈ 0.68 and O2B ≈ 0.5. The abstract claims "RFF-UCB performs successfully in identifying the best generation model," but the real-model evidence flatly contradicts this. The paper offers no analysis of *why* the polynomial-3 SCK-UCB succeeds where RFF (with RBF kernel) fails—whether it is a kernel mismatch, insufficient feature dimension, the resampling behavior of Algorithm 3, or something else. Given that computational efficiency is one of the paper's headline contributions, this unexplained failure on the only real-world benchmark directly undermines a core claim.

- **Real-model experiments are narrow.** Setup 1 and Setup 2 (the only real-model evaluations) use only 2–3 generative models, and Setup 1 uses a single partition of prompts into roughly "dog vs. car" categories. The appendix adds uni-Diffuser and DeepFloyd IF in pairwise comparisons (Figures 14–16), which helps but still does not test selection among ≥4 simultaneously deployed real generators. Setups 3–5 construct "non-expert" arms by adding Gaussian pixel noise—an artificial structure where the CLIP embedding of the noised image predictably degrades in a category-separable manner, making the routing problem far easier than realistic multi-model heterogeneity. Success on these setups provides limited evidence for the practical claim.

- **Baseline suite is insufficient for the claimed contribution.** The comparisons are Lin-UCB, One-arm Oracle, Naive-KRR, Greedy, and Random. These are either trivially weak or poorly matched. Missing are: (a) kernel/neural Thompson Sampling, (b) a simple supervised few-shot classifier trained on early observations and then frozen, (c) nearest-neighbor retrieval over prompt embeddings. Without these, it is impossible to determine whether the gains over One-arm Oracle come from the bandit methodology specifically or simply from any prompt-conditioned predictor.

- **No computation time measurements despite efficiency as a headline claim.** The paper's primary practical motivation for RFF-UCB is its O(t) vs. O(t³) cost. However, no wall-clock timing comparison between SCK-UCB and RFF-UCB is reported. The computational savings claim rests entirely on asymptotic analysis; without empirical runtime figures, users cannot assess the practical payoff.

### Minor

- **Notation inconsistency in Assumption 1.** The paper writes `s_g(y) = ⟨y, w_g*⟩_H` with y ∈ ℝ^d and w_g* ∈ H. Since y lives in the input space, this should be `⟨φ(y), w_g*⟩_H`. The intent is clear from context but the formal statement as written is ill-typed and should be corrected.

- **RFF resampling in Algorithm 3 is non-standard and under-explained in the main text.** At every call to COMPUTE_UCB_RFF, fresh random weights are drawn (line 4). This differs from the standard practice of fixing random features once. The paper defers the statistical justification to Appendix B.1, but the main text provides no intuition for why resampling at each round is valid or desirable. This creates ambiguity about the statistical independence assumptions the analysis relies on.

- **Realizability (Assumption 1) is not empirically validated.** The entire regret guarantee rests on score functions being linear in the RKHS. No held-out KRR prediction error or goodness-of-fit check is provided. In the real experiment, the kernel that works (poly3) differs from the one used by RFF-UCB (RBF), hinting that kernel choice matters substantially in ways the theory does not guide.

- **O2B can be positive even with modest per-prompt accuracy.** The metric measures improvement over the best single model's *average* score. In a two-model setting where one model dominates 60% of prompts, an algorithm with 68% correct arm selection will report positive O2B even if it misclassifies a substantial fraction of prompts. Reporting cumulative regret relative to the true per-prompt oracle (computable in synthetic settings) would be more informative and is the standard metric in bandit literature.

### Trivial

- Notation inconsistency in the main text of Algorithm 2 (index sets referred to as both Ψ_g and Φ_g).

---

## Nice-to-Haves

- **Practical justification of online over batch.** Running exploratory queries to suboptimal generative models is expensive. The paper would benefit from a brief discussion of regimes where the online framework is worthwhile (e.g., streaming prompt distributions, budget-limited APIs).
- **Experiments with ≥4 real simultaneous generative models** to test scaling in G.
- **Per-category breakdown** of OPR in the real-model setup to confirm the algorithm is learning routing structure, not exploiting accidental correlates.
- **Ablation on RFF feature size s**, kernel bandwidth σ, and exploration coefficient η—key hyperparameters whose practical sensitivity is unstudied.
- **Investigate multi-task/shared representation** across arms, which the "shared context" framing naturally suggests but the current per-arm independent KRR does not exploit.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "Only Setup 1 uses real generators."** Setup 2 explicitly uses Stable Diffusion, PixArt-α, and uniDiffuser as real models. This claim is factually incorrect; the criticism was overstated.

- **Harsh Critic: "Informal theorem statements are a presentation problem."** Theorems 1 and 2 are stated informally with full proofs deferred to the appendix—this is completely standard in theory-empirical ML papers and is not a weakness.

- **Harsh Critic: "KRR regularization written with ‖w‖ not ‖w‖²"** — The reviewer acknowledged this may be a parser artifact. Per hard rules, removing parser-artifact criticisms.

- **Neutral/Spark: Missing related works.** Per instructions, not including criticisms about missing related works since we cannot independently verify existence.

- **Harsh Critic: Algorithm 3 RFF resampling fully invalidates the analysis.** The paper explicitly states "To derive statistical guarantees, the number of features varies according to both the input data and error thresholds, which we will specify in Appendix B.1." The formal treatment is deferred but claimed to exist; this weakens (but does not eliminate) the concern as a main-text presentation issue rather than a soundness failure.

---

## Novel Insights

The most insightful observation across the reviews is the *disconnect between the theoretical framing and the empirical resolution*: the paper introduces the shared-context CB setting specifically to route to the best model per prompt, yet the per-arm independent KRR structure of SCK-UCB does not actually share any statistical information across arms. Notably, when one arm accumulates more observations its estimate improves, but this does not accelerate learning for other arms even though all arms share the same prompt context. Bridging this gap—perhaps through multi-task kernel learning or a shared low-rank RKHS structure—would make the "shared context" label genuinely load-bearing algorithmically, and would likely yield data efficiency improvements especially when G is large. The failure of RFF-UCB on real data (while SCK-UCB-poly3 succeeds) also hints at a meaningful interaction between kernel choice and the CLIP embedding geometry that is unexplored.

---

## Suggestions

1. **Diagnose and fix RFF-UCB on real data**: before claiming it as a viable contribution, the authors should understand and address why it underperforms SCK-UCB-poly3 by ~0.13 OPR on Setup 1. Report ablations on kernel type and feature dimension for RFF-UCB.
2. **Add wall-clock runtime comparisons** between SCK-UCB and RFF-UCB to substantiate the computational efficiency claim empirically.
3. **Expand real-model experiments** to at least 3–4 simultaneously available models drawn from public T2I families, tested over semantically diverse prompt categories.
4. **Add a simple supervised-classifier baseline** (train a prompt-to-best-model classifier on the first K rounds) to establish that the bandit exploration mechanism adds value beyond any prompt-conditioned predictor.
5. **Fix Assumption 1 notation** to use φ(y) rather than y in the inner product.

---

## Score and Decision

**Calibration:**

- *2Chkk5Ye2s (Mixture-UCB for generative model selection)*: Accepted as poster, scores 6,6,5,6,6 (avg ≈ 5.8). That paper addresses a related but different problem (mixture rather than prompt-conditioned routing), has broader real-model experiments, and its algorithms both work. This paper is comparable in motivation and theory but weaker empirically—RFF-UCB fails on real data and the real-model experiments cover fewer simultaneously active generators.

- *U0c2IaQhHk (RKHS-RL)*: Rejected, scores 3,6,6. Rejected mainly for incomplete proofs and oversimplified experiments. This paper's theory appears more complete, but the experimental limitations overlap.

- *5ep85sakT3 (Neural CB)*: Accepted, scores 8,6,5,5,6. Substantially stronger—tighter theory, broader experiments, and the efficient variant actually works.

**Assessment:** The paper identifies a real problem, provides a theoretically grounded solution, and shows initial positive results on real data for SCK-UCB. However, the main efficiency contribution (RFF-UCB) is empirically broken on real data with no explanation offered; the experiments are conducted on too few real models; and the baseline comparisons are insufficient to establish practical superiority. These are not formatting issues—they directly undercut the paper's stated contributions. Against the Mixture-UCB accepted paper as the primary anchor, this paper is below that bar due to the RFF failure and narrower real-model validation.

**Score: 4.5**  
Borderline rejection. The core idea and SCK-UCB are promising, but the paper's current form does not sufficiently support its main claims, particularly regarding RFF-UCB.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>