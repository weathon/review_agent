# ICLR Benchmark Results

Date: 2026-04-13 00:47
Critic/Merger: claude:claude-sonnet-4-6 (OpenRouter)
Neutral: qwen/qwen3.5-plus-02-15, Related Work: qwen/qwen3.5-flash-02-23:online (OpenRouter)

## OpNMWVDdKS

- GT: Reject (avg 5.8)
- Predicted: Accept (5.3/10)
- Match: NO

### Final Review

## Summary
ALDA (Associative Latent DisentAngLeMent) proposes combining a Quantized Latent Autoencoder (QLAE) for disentangled representation learning with a Hopfield-network-inspired associative memory mechanism to achieve zero-shot visual generalization in vision-based RL without data augmentation. The key technical contribution is replacing QLAE's hard argmin quantization with a Softmax separation function (yielding soft retrieval dynamics) while dropping the codebook update loss, framed as a learnable associative memory over fixed codebook "memories." The authors additionally prove that data augmentation is a necessary condition for a form of "weak disentanglement" and introduce a framestacking fix that allows disentanglement models to operate on individual frames while still integrating temporal context via a 1D CNN.

---

## Strengths

- **Competitive performance without external data:** ALDA consistently outperforms all non-augmentation baselines (DARLA, SAC+AE, RePo) on both "color hard" and DistractingCS, and matches or approaches SVEA despite SVEA having access to 1.8 million externally sourced real-world images. Demonstrating parity on a task where the baseline has a massive data advantage is a genuinely strong empirical finding.

- **Practical framestacking solution with demonstrable impact:** The insight that disentanglement models fail when presented with frame stacks — and the specific fix of folding frames into the batch dimension before encoding, then recombining via a 1D CNN — is a concrete, reproducible architectural contribution that the field has not previously addressed. The ablation in Figure 3 (BioAE degrades while QLAE does not) provides supporting evidence specific to this paper's setting.

- **Identifiable disentanglement in latent traversals:** Figure 6 provides qualitative evidence (with the full traversal set in appendix A.3) that individual ALDA latents track single factors of variation — e.g., torso orientation vs. background color — and, crucially, that task-irrelevant factors are encoded *separately* rather than discarded. This is a distinguishing design choice from task-centric methods (RePo) and is illustrated directly in a way most RL papers do not attempt.

- **Theoretically grounded framing of data augmentation:** Theorem 1 establishes a formal logical connection between successful data augmentation and a specific factorization property of the latent space (Eq. 2), motivating why explicit disentanglement can achieve a similar outcome. The probabilistic implication (Eq. 3) — that augmentation must cover all task-irrelevant source combinations — provides an intuitive and practically actionable argument for the approach's efficiency advantages.

---

## Weaknesses

- **Theorem 1 overstated in abstract and introduction:** The theorem proves a *necessary condition*: if data augmentation leads to an optimality-invariant Q-function, then the latent space must exhibit weak disentanglement. It does not prove that standard augmentation schemes (random crops, color jitter, image overlay) *produce* or *cause* this factorization in practice. The abstract's claim that the paper "formally shows data augmentation is a form of weak disentanglement" reverses the conditional. A more precise statement would be: "successful data augmentation is logically equivalent to weak disentanglement under the stated optimality assumptions." This matters because the theorem's practical force depends on assumptions about the Q-function that are never verified empirically.

- **Technical novelty of the "association" contribution is modest:** The core change relative to plain QLAE is: (1) replace the argmin (Eq. 4) with a Softmax (Eq. 7), and (2) drop L_quantize while retaining L_commit. This is functionally similar to soft vector quantization with a temperature parameter. While the Hopfield framing provides useful theoretical context, the paper does not benchmark against soft VQ variants (e.g., FSQ) or ablate the temperature β systematically in the main text, making it unclear whether the performance gains arise from the "associative memory" property per se or simply from smoother gradients due to soft quantization.

- **Ablations are restricted to Walker Walk:** Figures 3 and 4 — the ablations comparing BioAE vs. QLAE and QLAE vs. ALDA — are shown only on a single task. Given that the paper tests four tasks with different observation complexities, the scope of ablations is insufficient to establish that the design choices generalize. An ablation failure on Ball in Cup or Finger Spin could significantly change the conclusions.

- **Codebook initialization is unexplained:** The paper drops L_quantize and treats the codebook as "predetermined memories" that do not adapt toward encoder outputs. However, no explanation is given for how the initial codebook values are set, whether random initialization is sufficient, or how to prevent dead or collapsed codebook entries during long training runs. This is a material reproducibility concern for a central design choice.

- **|z_d| = 12 heuristic lacks justification in main text:** The paper selects the disentangled latent dimensionality based on the proprioceptive state size ("ballpark"), a heuristic without theoretical grounding. The paper acknowledges this and refers to Section A.4 in the appendix for sensitivity results, but this is a critical hyperparameter — in real-world deployments where proprioceptive state dimensions may not be defined or accessible, the method has no principled selection strategy. If A.4 shows robustness, this should be summarized in the main text.

- **DistractingCS results undermine the scope of the zero-shot claim:** The paper honestly reports that "performance degrades severely for all algorithms" on DistractingCS and attributes this partly to camera shake affecting learned dynamics. However, DistractingCS is specifically designed as a more realistic deployment condition than "color hard," and the title broadly claims "zero-shot generalization." The title's claim is substantially scoped by this failure, which is not adequately reflected in the framing.

- **Key augmentation baseline (DrQ/DrQ-v2) absent:** The paper references DrQ in Section 2.3 as a prominent augmentation method but does not include it as a baseline in Figure 5. SVEA is included, which is more recent and stronger, but comparing only against SVEA leaves readers unable to assess where ALDA sits relative to the broader landscape of augmentation-based approaches. Notably, SADA (Almuzaire et al., 2024) is discussed in the introduction but also absent from Figure 5 despite being the most recent augmentation baseline mentioned.

---

## Nice-to-Haves

- **Fair data budget comparison:** A comparison against SVEA restricted to the same number of training images (no external Places dataset) would isolate whether ALDA's advantage is architectural or data-driven. As currently designed, ALDA matching SVEA is encouraging but conflated with SVEA's massive external data advantage.

- **Quantitative disentanglement on a controlled synthetic benchmark:** Evaluating the ALDA encoder on a dataset with known ground-truth factors (e.g., 3DShapes or CLEVR) would allow computing standard disentanglement metrics (MIG, DCI) and would provide objective evidence that the qualitative latent traversals in Figure 6 reflect genuine disentanglement rather than incidental feature separation.

- **Attention/retrieval visualization for OOD inputs:** Plotting the Softmax attention distribution over codebook entries for in-distribution vs. OOD inputs would directly validate whether the association mechanism is performing meaningful retrieval or simply acting as a smooth approximation to the argmin.

- **Compute and memory overhead analysis:** The motivation emphasizes that augmentation methods are computationally expensive, but no empirical comparison of training wall-clock time or memory usage is provided. Quantifying this trade-off would strengthen the paper's practical case.

- **Sensitivity of |z_d| to the heuristic:** A plot of "color hard" performance vs. |z_d| from, say, 6 to 24 across at least two tasks would quantify how brittle the heuristic is and help users apply ALDA to new environments.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "DrQ-v2 citation inconsistency / DrQ-v2 is absent"** — The paper does cite DrQ in the background section and uses SVEA as its primary augmentation comparison, which is a stronger and more recent method. The complaint that the most dominant baseline is missing is partially addressed by SVEA's inclusion, though the concern about DrQ/DrQ-v2 is kept as a genuine (weaker) weakness above.

- **Harsh Critic: Citation inconsistency between "Almuzaire" and "Almuzzareca"** — Pure formatting/typo nitpick; removed per policy.

- **Harsh Critic: SVEA comparison is unfair to ALDA's claims** — The paper explicitly states "We do not expect to outperform SVEA since it uses additional data…" The comparison is intentionally asymmetric and favors the baseline, making ALDA's competitive results stronger, not weaker. This is not a flaw.

- **Harsh Critic/Spark Finder: Dynamics shift generalization (gravity, friction) not tested** — The paper's stated scope is visual generalization under distribution shifts in DMControl. Criticizing the absence of dynamics generalization is scope creep. Removed.

- **Spark Finder: Sim-to-real validation demanded** — The paper does not claim real-world deployment; requiring sim-to-real experiments is outside the paper's stated scope. Removed.

- **Spark Finder: Temporal disentanglement as a contradiction** — The paper explicitly acknowledges this as a limitation and future direction in Section 6. Removing as a criticism; it is already addressed honestly.

- **Reviewer 2: "Limited benchmark diversity" as a core weakness** — While broader benchmarks would strengthen the claims, the four DMControl tasks tested are the standard generalization benchmark in this community. Flagging because this criticism is generic ("test on more benchmarks") and does not uniquely undermine the core contribution; moved to nice-to-haves rather than a substantive weakness.

---

## Novel Insights

The most novel insight across the three reviews — one that goes beyond the paper's explicit contributions — is the observation that the "association" mechanism and soft VQ are not clearly distinguished, raising the question of what the *associative* framing specifically purchases beyond smoother gradient flow. If the Hopfield interpretation is correct, one would expect the model to perform *qualitatively different* retrieval on OOD inputs compared to ID inputs (attending to different "memories"). The fact that this is never visualized or measured means the paper's theoretical framing could be recast as a pragmatic soft-quantization trick, which is a lesser but still valid contribution. Designing an experiment that distinguishes the Hopfield interpretation from plain soft VQ — e.g., by showing the Softmax distribution shifts toward different codebook entries under distribution shift — would be a genuinely informative scientific test of whether the "associative" claim is mechanistically supported.

---

## Suggestions

1. **Restate Theorem 1 accurately:** Change "formally show that data augmentation is a form of weak disentanglement" to "formally show that successful data augmentation implies weak disentanglement" throughout the abstract and introduction. This is a minor but important precision fix that removes an overstatement.

2. **Run the ALDA vs. QLAE ablation on all four tasks**, not just Walker Walk, to establish that dropping L_quantize and using Softmax consistently helps rather than hurts.

3. **Explain codebook initialization and provide dead-codebook statistics** (e.g., percentage of codebook entries actively used at convergence). If initialization is random and all entries remain active, this is reassuring; if not, it is a reproducibility risk.

4. **Report |z_d| sensitivity in the main paper**, even as a single figure. If A.4 already shows robustness, a one-panel summary in Section 4 would substantially address a key reproducibility concern without requiring additional experiments.

5. **Add a retrieval visualization** (Softmax weights per codebook dimension for an ID vs. OOD sample pair) to Section 4.2 or Figure 6. This would directly validate or challenge the associative memory mechanism's contribution.

6. **Include training compute and wall-clock comparison** with SVEA in a table. Since a major motivation is computational efficiency over augmentation, quantifying this is important for the practical narrative.

---

**Novelty:** Moderate. The Hopfield interpretation of QLAE and the specific Softmax modification are novel, as is the framestacking fix for disentanglement models. The theoretical connection between augmentation and disentanglement is insightful but narrower than claimed. Most components are assembled from existing building blocks.

**Technical soundness:** Adequate, with notable gaps. The theoretical claim (Theorem 1) is logically valid but directionally overstated. The codebook initialization and sensitivity analysis are incomplete in the main text.

**Empirical support:** Moderate. Results on 4 tasks with 5 seeds and 95% CI are solid for the community standard. Key ablations are underscoped (Walker Walk only), and DistractingCS failure limits the scope of the zero-shot claim.

**Significance:** Moderate-to-good. Demonstrating that a non-augmentation approach can match SVEA on tasks where SVEA has a 1.8M-image data advantage is a meaningful result. If the approach were more thoroughly ablated and the theoretical framing tightened, the contribution would be more impactful.

**Clarity:** Good overall. The framestacking fix, association derivation, and objective are clearly described. The relationship between z_obs, z_d, z_q, and z_π in Figure 2 and the codebook initialization details could be more explicit.

MY FINAL SCORE: <pineapple>5.3</pineapple>

---

## TvfkSyHZRA

- GT: Accept (Poster) (avg 7.0)
- Predicted: Accept (6.8/10)
- Match: YES

### Final Review

## Summary

This paper proposes that grokking (delayed generalization after prolonged overfitting) is primarily driven by two linked mechanisms: *Softmax Collapse* (SC), a floating-point absorption error in the Softmax that zeros out gradients from correctly-classified samples when logits grow too large, and *Naïve Loss Minimization* (NLM), the tendency of gradients to align with the weight-scaling direction after overfitting, which continuously inflates logits until SC occurs. To validate these hypotheses, the authors introduce *StableMax*, a piecewise-linear replacement for Softmax that avoids absorption errors and enables grokking without regularization (converting complete overfitting into grokking), and *⊥Grad*, an optimizer that projects out the NLM component of the gradient to achieve rapid generalization without the characteristic delay (converting grokking into fast generalization). The paper further shows that these mechanisms explain the efficacy of existing grokking-inducing methods such as weight decay and MSE loss.

---

## Strengths

- **Specific, testable causal hypothesis about floating-point errors.** The core claim — that SC is a *numerical*, not merely a statistical, phenomenon — is tested with a particularly clean causal intervention: training with float16/float32/float64 and observing that SC onset shifts proportionally with precision (Fig. 2). This is direct evidence rather than correlation, and represents a perspective on grokking largely absent in prior work focused on implicit bias and circuit formation.

- **Interventions serve as causal probes.** StableMax (converting complete overfitting to grokking, Fig. 4) and ⊥Grad (converting grokking to fast generalization, Fig. 6) are not merely useful tools — they are causal demonstrations of the hypothesized mechanisms. Each reverses exactly the expected transition in Fig. 1. The fact that StableMax induces grokking with *increasing* weight norms (Fig. 4, middle) is a notable result that directly disentangles weight norm from the necessary condition for grokking, challenging prior accounts.

- **Unified explanation for disparate prior observations.** Section 5.2 provides a clean, parsimonious account of why weight decay induces grokking (it opposes the NLM scaling direction), why MSE loss works on shallow networks (logit overshoot prevents indefinite NLM), and why input dimensionality matters (low-dimensional inputs prevent easy overfitting). This synthesis goes beyond describing the phenomenon to providing predictive explanations.

- **Empirical evidence of NLM in non-homogeneous architectures.** Fig. 5 shows gradient-weight cosine similarities approaching 0.9 in the output layers of MLPs *with* bias terms and in transformers, providing empirical grounding for the NLM concept beyond the theoretically clean homogeneous case.

---

## Weaknesses

- **StableMax conflates numerical stability with optimization geometry change.** Proposition 1 shows that StableMax is equivalent to Softmax with log-compressed inputs, meaning it defines a *different loss function*, not merely a numerically repaired version of the original. The success of StableMax could partly arise from the modified gradient geometry — specifically, the piecewise-linear tail imposes a fundamentally different margin structure than the exponential. The paper does not disentangle these effects. A comparison with float64 Softmax on the same tasks (especially the 40% dataset size shown in Fig. 2a) is notably absent: if float64 alone achieves grokking in that setting, it would confirm the purely numerical story; if not, the loss-geometry change in StableMax is doing additional work. This matters for the core claim.

- **The gap between NLM theory and ⊥Grad application in non-homogeneous models.** The theoretical motivation for ⊥Grad relies on the homogeneity property (Def. 6) under which $\alpha\theta$ is a provable NLM direction. However, ⊥Grad is applied to models with bias terms (Fig. 6b) and transformers (Fig. 6a), which are non-homogeneous. The paper acknowledges this and provides empirical evidence of gradient alignment (Fig. 5), but the projection is still applied to the *entire concatenated weight vector* $\theta$ globally. Whether this global projection removes the correct component in non-homogeneous models is not theoretically characterized. The limitations section mentions quasi-homogeneity but does not close this gap. If ⊥Grad works for the wrong reasons in non-homogeneous settings, the mechanistic interpretation weakens.

- **The paper explains the delay and termination of grokking but not its abruptness.** A defining feature of grokking is the *sudden* transition from near-chance to near-perfect test accuracy. The paper provides an account of why generalization is delayed (NLM dominates gradient) and why it eventually halts (SC), but not why, when generalization does occur, it happens over a very short window. The limitations section acknowledges this gap with a brief mention but it represents a substantial incompleteness in the mechanistic account, particularly for an ICLR audience where prior work has extensively studied the transition dynamics.

- **Narrow experimental scope for the strength of the claims.** The core results are demonstrated on modular arithmetic (addition/subtraction/product mod 113) with a 2-layer MLP, with transformer results only for subtraction mod 113 in Fig. 6a. Sparse parity and a tiny MNIST subset appear in supporting roles. The introduction explicitly invokes grokking in "vision and language" settings, but no evidence is provided that SC or NLM manifests in those settings. The causal claims (SC stops grokking; NLM causes SC) are presented as general explanations yet are validated on a narrow slice of the phenomena.

- **StableMax requires extremely long training (60–80k epochs) in Fig. 4 left.** The delay in grokking with StableMax, while avoiding complete overfitting, is still very substantial. The paper does not discuss what dynamics govern this remaining delay after SC is prevented. If NLM continues under StableMax (the weight norm in Fig. 4 middle continues rising substantially), then a large portion of the NLM dynamics persist even without SC, raising the question of whether StableMax's mechanism is distinct from simply tolerating the NLM-induced logit growth for longer.

---

## Nice-to-Haves

- **Layer-wise projection in ⊥Grad.** The current global projection (Eq. 12) over the entire concatenated weight vector is extremely mild for large models and does not correspond to the per-layer NLM directions. A layer-wise variant — projecting out the $\alpha W_\ell$ direction for each layer $\ell$ — would more directly align with the homogeneity argument and potentially be both more principled and more scalable.

- **Explicit analysis of weight norm behavior under ⊥Grad.** Since $\nabla_\perp \mathcal{L}$ has no component in the $\theta$ direction, ⊥Grad approximately preserves the weight norm (to $O(\eta^2)$). This connects ⊥Grad to normalized gradient descent or Riemannian gradient descent on a sphere; making this connection explicit would situate the method in a known literature and clarify its implicit regularization properties.

- **Ablation on alternative absorption-resistant functions.** The specific functional form of StableMax ($s(x) = x+1$ for $x \geq 0$, $s(x) = 1/(1-x)$ for $x < 0$) is well-motivated informally but not ablated. A comparison with Softplus-based cross-entropy or other linear-tail alternatives would clarify whether the specific design is critical.

- **Validation of ⊥Grad on standard benchmarks.** Showing that ⊥AdamW does not degrade performance on tasks where grokking is not the concern (e.g., CIFAR-10 classification) would strengthen the case that ⊥Grad is a practically safe intervention.

- **Quantitative analysis of the fraction of zero-gradient samples over time.** Visualizing what percentage of training samples yield zero gradient (due to SC) as a function of epoch would provide a more direct and quantifiable measure of SC severity, making the narrative more concrete.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"No comparison between StableMax and ⊥Grad combined" (Harsh Critic):** Figure 7 explicitly shows a combined ⊥SGD + StableMax trajectory (labeled "LSGD + StatMolux" — apparently a rendering artifact). This experiment exists in the paper.

- **"Statistical significance / multiple seeds" (Harsh Critic):** Single-run evaluation is standard practice in the grokking literature and in the broader algorithmic generalization setting. Demanding multi-seed statistics with confidence intervals for this type of controlled mechanistic study would be imposing a norm not standard in this field.

- **"Why wrong-class gradients don't compensate" (Harsh Critic) — partially:** The paper explicitly validates this in Appendix B.1 via a causal intervention (artificially zeroing correct-class gradients replicates the SC-induced plateau). The main text is brief on intuition but the empirical addressal is provided. This could be improved with more intuition in the main text, but it is not an unaddressed concern.

- **"Lack of external related work analysis" (general):** Per review guidelines, missing related works are not flagged as we cannot verify their existence.

---

## Novel Insights

The most genuinely novel insight in this paper — largely absent from prior grokking literature — is that the failure to grok without regularization in standard settings is a *computational* rather than a *statistical* phenomenon: the exponential nature of Softmax, combined with the ease of overfitting in these tasks, guarantees that floating-point arithmetic will eventually collapse the gradient signal before generalization can occur. This reframes regularization's role: weight decay is not primarily reshaping the loss landscape toward simpler circuits (though it may do this too) but is operationally countering the NLM scaling direction that would otherwise drive logits into a numerical dead zone. The causal evidence from precision manipulation is particularly clean and the connection between the known gradient-weight alignment result (Lyu & Li, 2020) and finite-precision failure is a substantive new link. A secondary insight is that grokking is representation-contingent even for the same algorithmic task (Fig. 4, right): changing input representation from one-hot to compact binary removes the overfitting-induced delay entirely, suggesting that what makes a task "a grokking task" is not the algebraic structure but the ease of memorization induced by the input dimensionality.

---

## Suggestions

- Add a float64 experiment on the 40% modular addition setting (currently absent from Fig. 2a, which only shows float16 and float32) to determine whether precision alone can induce grokking in the hardest setting, or whether StableMax's linear tail is doing structural work beyond numerical stabilization.
- Provide a causal experiment that clips logits at a fixed threshold (preventing SC without changing the loss function) to cleanly isolate the numerical-stability component of StableMax's benefit from its loss-geometry change.
- Expand the limitations discussion to explicitly address the sudden-generalization puzzle: the paper explains the *delay* via NLM and the *termination* via SC, but not the *abruptness* of the eventual transition. Even a qualitative hypothesis (e.g., referencing the weight-decay rotational equilibrium of Section 5.2 as a potential sudden-unlock mechanism) would strengthen the mechanistic completeness.
- Consider a brief evaluation of ⊥Grad behavior on weight norm dynamics over training, and whether exploding norms under ⊥Grad (since NLM is removed but no explicit regularization is added) become a concern in longer training runs.

---

**Novelty**: High — the floating-point absorption mechanism is a specific, testable, and largely novel contribution to the grokking literature.

**Technical soundness**: Good — formal definitions are precise, proofs are provided, and the empirical methodology is clean. The main theoretical gap (non-homogeneous case for ⊥Grad) is acknowledged but not resolved.

**Empirical support**: Moderate-to-good — the precision-modulation experiment is a particularly strong causal test, but the experimental scope is narrow (modular arithmetic with one architecture as the primary testbed), and the connection to broader grokking settings claimed in the introduction is asserted rather than demonstrated.

**Significance**: Good — the insights are actionable, the unified explanation is parsimonious, and StableMax/⊥Grad provide new experimental tools for the community. The significance for large-scale practice remains an open question.

**Clarity**: Good — formal definitions are clear, the narrative arc of Fig. 1 is well-constructed, though some figures (e.g., Fig. 2's ambiguous vertical dashed line, Fig. 7's abbreviation artifacts) could be improved.

MY FINAL SCORE: <pineapple>6.8</pineapple>

---

## Ah3n8U3kRT

- GT: Reject (avg 5.5)
- Predicted: Accept (5.5/10)
- Match: NO

### Final Review

## Summary
This paper introduces median-clipping-based zeroth-order algorithms (ZO-clipped-med-SSTM and ZO-clipped-med-SMD) for non-smooth convex optimization, and extends the technique to the stochastic multi-armed bandit (MAB) problem under symmetric heavy-tailed noise with any κ > 0. The key innovation is a novel oracle model (Assumption 3) that encodes both symmetry and power-law tail behavior, enabling the construction of unbiased gradient estimators with bounded second moment even when the noise distribution has unbounded expectation (κ ≤ 1). For ZO optimization, the methods achieve $\tilde{O}(d^2\varepsilon^{-2})$ iterations — matching optimal rates for bounded-variance problems — for any κ > 0, whereas prior work (ZO-clipped-SSTM/SMD) degenerates as κ → 1 and is undefined at κ = 1. For MAB, the proposed Clipped-INF-med-SMD achieves $\tilde{O}(\sqrt{dT})$ regret, matching the optimal lower bound for bounded-variance settings.

## Strengths

- **Genuine extension of the heavy-tail frontier to κ ≤ 1.** Prior ZO methods ([19, 20]) achieve high-probability convergence only for κ ∈ (1, 2] and degenerate as κ → 1. This paper is the first to handle κ ∈ (0, 1] in the zeroth-order setting, including Cauchy noise with undefined expectation. This is a non-trivial barrier that is cleared by the combination of Assumption 3 (power-law envelope on the noise density) and the component-wise median estimator.

- **Technically non-trivial Lemma 1.** The unbiasedness of the median estimator under Assumption 3's symmetry, and the derivation of bounded second moment $\sigma^2 = O(dM_2^2 + d^2\Delta^2(4/\kappa)^{2/\kappa})$ requiring only $m > 2/\kappa$ samples, is a substantive technical result that goes beyond straightforward application of prior median analysis. The proof approach is explicitly noted as distinct from earlier works.

- **Rates matching bounded-variance optimal for ZO optimization.** Theorem 1 (Lipschitz oracle) achieves $\tilde{O}(\max\{d^{3/2}M_2R/\varepsilon,\; d(M_2^2 + d\Delta^2/\kappa^{2/\kappa})R^2/(b\varepsilon^2)\})$, which for fixed κ has the same ε and d scaling as the optimal ZO bound under bounded variance. Table 1 provides a clear, honest contrast against the baseline's $(\sqrt{d}\varepsilon^{-1})^{\kappa/(\kappa-1)}$ factor that blows up at κ = 1.

- **Empirical validation of the median clipping effect under extreme tails.** Figure 3 cleanly shows that for κ ≤ 1, median-clipping methods significantly outperform non-median counterparts, while matching them for κ > 1. This directly validates the core theoretical claim where it matters most.

## Weaknesses

- **Potential theoretical gap in the MAB section: symmetry under importance weighting.** The MAB algorithm applies the median operator to importance-weighted estimators $\hat{g}_{t,i} = g_{t,i}/x_{k,i}$ (for chosen arm) and $0$ otherwise. Even if the raw noise $\xi_t$ satisfies Assumption 3's symmetry, the importance-weighted estimator $\hat{g}_t$ has a manifestly asymmetric distribution (zero with probability $1 - x_{k,i}$, heavy-tailed with probability $x_{k,i}$). The paper does not show that $\hat{g}_t - \mathbb{E}[\hat{g}_t]$ satisfies Assumption 3, which is the premise on which Lemma 1's bounded second moment and unbiasedness are derived. If this gap is not addressed in the appendix proof, Theorem 3's regret bound rests on an unjustified application of Lemma 1. This is the most serious concern in the paper.

- **Experimental conclusion in §5.1 appears inconsistent with Figure 1.** The paper claims "HTINF and APE do not have convergence in probability, while our Clipped-INF-med-SMD does." Yet Figure 1 (per the figure caption) shows HTINF achieving the *highest* probability of best-arm selection (~0.9) and the *lowest* average expected regret (~0.1), while the proposed method stabilizes at probability ~0.6 and regret ~0.2. Claiming that HTINF "does not have convergence in probability" while it empirically dominates both metrics in Figure 1 is logically inconsistent with the displayed results, unless "convergence in probability" is being used in a narrow theoretical sense that must be made explicit. This casts doubt on the reliability of the paper's empirical narrative.

- **Growing constant $(4/\kappa)^{2/\kappa}$ is obscured in abstract and Table 1.** Lemma 1 and Theorem 1 explicitly include the factor $(4/\kappa)^{2/\kappa}$ in $\sigma^2$, which grows without bound as κ → 0, and $m = 2/\kappa + 1$ means per-iteration oracle cost also grows as $O(1/\kappa)$. The abstract's claim that methods "require $\tilde{O}(d^2\varepsilon^{-2})$ iterations for any κ > 0" and Table 1's non-degenerating rates absorb this exploding factor into the $\tilde{O}$. While the theorems themselves are honest, the top-level framing obscures the practical degradation for very small κ. At minimum the abstract or Section 6.2 should note that the hidden constant grows as $(4/\kappa)^{2/\kappa}$.

- **Off-by-one in Algorithm 3.** Line 6 computes $\sigma_{med}^{k+1}$ but line 7 clips $\sigma_{med}^k$ (without +1 subscript). At iteration k=0, the update would reference $\sigma_{med}^0$ which has not been computed. If intentional (e.g., the update is lagged), this must be explained; if a typo, it should be corrected since index consistency is critical for algorithm correctness and proof validity.

- **Contribution §1.1 references "Assumption 4" that does not appear in the main text.** The paper lists "Theory I: ...Assumption 4 (our novel theoretical zeroth-order oracle)" but the main text only defines Assumptions 1–3. Readers cannot evaluate the stated contribution without tracking down the appendix. This appears to be a numbering inconsistency from a revision; it should be corrected.

- **ZO experiments use only 3 runs, and the unexplained acceleration failure is not discussed.** Figure 3 shows that SGD-based variants (ZO-clipped-med-SGD) consistently outperform the accelerated SSTM variants (ZO-clipped-med-SSTM) across all κ values tested. This is theoretically surprising — acceleration should help for convex problems — and is practically important since the paper's primary theoretical algorithm is the SSTM variant. No explanation is provided.

## Nice-to-Haves

- **MAB experiments with d > 2 arms.** The sole MAB benchmark uses d = 2 arms, far too small to validate the claimed $\tilde{O}(\sqrt{dT})$ dimension scaling. Adding experiments with d ∈ {10, 50, 100} would substantiate Theorem 3's claims about dimension dependence.

- **Comparison on total oracle queries, not iterations.** Since each iteration of ZO-clipped-med-SSTM requires $(2m+1)\cdot b$ calls, and $m = 2/\kappa + 1$ grows with decreasing κ, comparing on total oracle calls (not just iterations) would give a more complete picture. Table 1 already notes the per-call overhead ($b/\kappa$ calls), but a sample-complexity plot would be informative.

- **Adaptive selection of m.** The theoretically optimal m = 2/κ + 1 requires knowledge of κ. The paper discusses using m = 3 as a fallback for κ ≥ 1, but an adaptive or data-driven scheme for unknown κ would improve practical usability and is noted as future work.

- **Theoretical or empirical characterization of symmetry tolerance.** The method relies on Assumption 3's symmetry, but Section 6.1 argues robustness to mild asymmetry. Quantifying how much skewness degrades performance (at least empirically, supplementing the appendix results §D.2.1) would strengthen the applicability claims.

- **ML-relevant ZO benchmarks.** The ZO experiments use only a synthetic least-squares problem. Including a standard zeroth-order benchmark (e.g., black-box hyperparameter tuning or an adversarial attack scenario) would broaden appeal and relevance for the ICLR community.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic: "The abstract's 'any κ > 0' is technically wrong."** The bound does hold for any κ > 0 — the $(4/\kappa)^{2/\kappa}$ factor is inside the constant, not in the asymptotic class. The theorems are honest. The issue is framing, not correctness. Moved to the "obscured constant" weakness above.

- **Harsh critic: Assumption 3 does not cover "majority of distributions."** The claim that the assumption is "strictly stronger than bounded κ-th moment with symmetric density" and the demand for a counterexample are overly picky for this type of paper. The characterization is indeed imprecise, but the paper explicitly provides worked examples (Cauchy) and notes reductions to standard assumptions for κ ∈ (1, 2].

- **Harsh critic: Cryptocurrency experiment should compare against HTINF/APE.** The cryptocurrency experiment is explicitly described as a real-world illustration in the full-feedback setting (disclosed by the authors), not a head-to-head MAB algorithm comparison. The disclosure and framing are clear; demanding HTINF/APE comparison here is scope creep for a demonstration task.

- **Harsh critic: Broader impact statement is two sentences.** Pure formatting/style criticism with no bearing on scientific content.

- **Positive reviewer: Lipschitz oracle is hard to verify in black-box settings.** This is a common limitation of any Lipschitz-type assumption in ZO theory and is not specific to this paper. Not a substantive weakness here.

- **Spark finder: Provide a lower bound matching the proposed rates.** Proving matching lower bounds for zeroth-order optimization with symmetric heavy-tailed noise is a significant open theoretical problem outside this paper's scope. Appropriate as a future direction, not a weakness.

## Novel Insights

The most insightful observation emerging from this review is the subtle relationship between the median's variance-reduction mechanism under Assumption 3 and the structural cost hidden in $(4/\kappa)^{2/\kappa}$: the paper achieves "optimal-rate matching" at the price of a super-exponentially growing constant as κ → 0, and the improvement over prior work is most dramatic not as κ → 0 (where the constant explodes) but around κ ≈ 1, where prior rates are actually *undefined* (due to the $\kappa/(\kappa-1)$ exponent) while the proposed method remains well-behaved. The regime κ ∈ (0.8, 1.2] is therefore the paper's true sweet spot, not the extreme heavy-tail regime. This observation, which is not clearly stated in the paper, would help set realistic expectations for practitioners.

## Suggestions

1. **Clarify the symmetry-under-importance-weighting question** (most critical): either prove in the main text that the centered importance-weighted estimator satisfies Assumption 3, or provide a separate lemma for Theorem 3's proof that does not rely directly on Lemma 1. This is required to establish the MAB result.

2. **Fix or explain the off-by-one in Algorithm 3 line 7** (σ_med^k vs. σ_med^{k+1}).

3. **Reconcile §5.1's conclusion with Figure 1**: either re-examine whether "convergence in probability" in the theoretical sense applies here, or add explanation distinguishing theoretical guarantee from empirical figure. If HTINF empirically dominates, this should be acknowledged honestly along with the point that it lacks theoretical guarantees for the experimental κ.

4. **Add a brief discussion on why SSTM (accelerated) underperforms SGD (non-accelerated)** in Figure 3, as this phenomenon is practically important and potentially expected under heavy-tail settings.

5. **Add a quantitative note on the $(4/\kappa)^{2/\kappa}$ constant** to the abstract or Section 6.2, specifying numerically for e.g. κ ∈ {0.25, 0.5, 1} what the effective constant is, so readers can calibrate when the proposed method is truly superior in total oracle calls.

---

**Axis evaluations:**
- **Novelty:** High — extending median estimation to ZO optimization and MAB under κ > 0 is a genuine first, and Assumption 3 is an original oracle formulation.
- **Technical soundness:** Moderate-to-good for the ZO optimization part; uncertain for the MAB part due to the unresolved importance-weighting symmetry question.
- **Empirical support:** Weak-to-moderate — the ZO Figure 3 is convincing; the MAB experiment (d=2 only, apparently contradictory conclusions) and crypto experiment (full-feedback, non-algorithmic baselines) are insufficient.
- **Significance:** Moderate-to-high for the heavy-tailed optimization community; the κ ≤ 1 regime is genuinely important.
- **Clarity:** Moderate — theorems are clearly stated, but the Assumption 4/3 numbering inconsistency, Algorithm 3 index issue, and §5.1 narrative weaken overall clarity.

MY FINAL SCORE: <pineapple>5.5</pineapple>

---

## HmwneoGoy9

- GT: Reject (avg 5.2)
- Predicted: Accept (6.8/10)
- Match: NO

### Final Review

## Summary

SeerAttention proposes a learnable sparse attention mechanism that augments standard attention with a lightweight gating module (AttnGate) to adaptively select significant blocks in attention maps. The key enabler is a customized FlashAttention kernel that extracts block-level max-pooled attention maps during training without materializing the full O(n²) matrix, circumventing the memory bottleneck that has limited prior learned sparsity approaches. The method is evaluated in post-training calibration and long-context fine-tuning settings (with YaRN), demonstrating superior accuracy-efficiency tradeoffs over MoA and MInference on perplexity and LongBench.

---

## Strengths

- **Custom FlashAttention training kernel is a genuine engineering contribution.** The technique of storing row-max values (r_ij) during FlashAttention's online pass and rescaling them post-iteration (Equation 2) to recover block-level attention statistics avoids the quadratic memory cost of naïve attention while providing supervision for the gate. Figure 8 confirms near-identical memory usage to FlashAttention-2, enabling training at 64k+ sequence lengths that were previously infeasible for learned sparse attention.

- **Learned sparsity outperforms handcrafted heuristics across most settings.** Table 1 and Table 2 show SeerAttention at equal or higher sparsity achieving lower perplexity and higher LongBench scores than both MoA and MInference in nearly every configuration up to 64k context, demonstrating that the learned gate captures head-specific, input-dependent sparsity patterns that static patterns miss.

- **Fine-tuning integration with YaRN delivers compelling results.** Table 3 shows that YaRN+SeerAttention at 50% sparsity achieves perplexity of 8.81/2.47 vs YaRN baseline 8.79/2.46 (PG19/Proof-pile) — effectively lossless. Even at 90% sparsity, perplexity is 9.16/2.60, a ≤5% relative increase. Figure 1a shows loss curves at both sparsity levels tracking the dense baseline through 400 training steps, indicating stable joint optimization.

- **End-to-end TTFT dominates both competitors.** Table 4 shows SeerAttention achieving 13.37s TTFT at 128k vs MInference's 14.38s despite similar sparsity (0.95), and MoA running out of memory entirely at 128k.

- **Ablations are targeted and informative.** The RoPE ablation (Figure 9) provides convincing evidence: without the re-scaled RoPE in AttnGate, perplexity degrades catastrophically beyond the training context length. The pooling ablation (Figure 10) over 49 combinations identifies a principled best configuration (Qavg, Kmaxmin).

- **Flexibility via a single checkpoint.** Because the gate is trained to predict a distribution and top-k is applied at inference time, users can adjust sparsity ratio post-hoc without retraining — a practical advantage over MoA's per-sparsity search and MInference's fixed patterns.

---

## Weaknesses

- **Figure 1b presents a cross-dataset comparison that is visually misleading.** The orange dashed line labeled "YaRN Baseline" is evaluated on PG19 (perplexity ≈ 10), while the red solid line "YaRN w/ SeerAttention" is evaluated on Proof-pile (perplexity ≈ 3). These are different datasets with inherently different absolute perplexities. Table 3 confirms this: PG19 baseline is 8.79 and Proof-pile baseline is 2.46. The figure should compare both methods on the same dataset; as presented, the apparent dramatic reduction in perplexity is an artifact of the dataset switch, not a model improvement.

- **The 5.67× speedup figure in the abstract and Figure 1c refers to kernel-level computation only, not end-to-end inference.** Table 4 (TTFT) shows that at 128k the end-to-end speedup is 35.54 / 13.37 ≈ 2.66×, and at 32k it is approximately 4.63 / 3.60 ≈ 1.29×. The abstract should clearly label the 5.67× figure as kernel-level and present the corresponding end-to-end figure alongside it to avoid overclaiming. This distinction matters because other LLM components (MLP, normalization, etc.) dilute the attention speedup at the system level.

- **Block size B=64 is a central hyperparameter that is never ablated.** This single choice controls the coarseness of the sparsity approximation and has a first-order effect on the accuracy–efficiency tradeoff. The paper fixes B=64 throughout without justification. Since block size also determines hardware tiling granularity, an ablation over B ∈ {32, 64, 128} is necessary to understand the design space and whether the current choice is optimal.

- **Evaluation is limited to perplexity and LongBench aggregates; Needle-in-a-Haystack (NIAH) and retrieval-intensive tasks are absent.** High attention sparsity could cause disproportionate failure on tasks requiring precise long-range retrieval (e.g., NIAH, multi-hop QA) even when aggregate perplexity changes are small. Without these evaluations, the claim of "minimal loss" is incomplete — perplexity is insensitive to local retrieval failures that matter in practice.

- **No downstream task evaluation for the fine-tuned model.** Section 5.2 evaluates the YaRN+SeerAttention model only on perplexity (Table 3). Given that fine-tuning is presented as a key contribution, the absence of any LongBench or instruction-following results for the fine-tuned model leaves a significant gap in validating that the fine-tuned model retains general capability.

- **Post-training perplexity degradation at 128k/90% sparsity is substantial, not "minimal."** Table 1 shows perplexity rising from 10.03 (dense) to 13.20 at 90% sparsity and 128k context — a 31.6% relative increase. The "minimal perplexity loss" claim in the abstract is accurate for the fine-tuning scenario (which is explicitly scoped to 32k), but the paper should be clearer that post-training at very high sparsity and very long context is a different and weaker regime.

- **The linear projection layer in AttnGate is central to the architecture but not ablated.** It is unclear whether the linear layer is necessary, or whether a direct pooled-Q × pooled-K dot product (analogous to standard attention on pooled tokens) would perform similarly. Given that the gate design is one of the paper's core contributions, this is a meaningful gap.

---

## Nice-to-Haves

- **Per-head variable sparsity.** The paper acknowledges (Table 1 discussion) that MInference's per-head sparsity is likely why it outperforms at 128k in post-training. Extending SeerAttention to learn per-head sparsity budgets would be a natural enhancement and could close this gap.

- **Ranking-based training objective.** The gate is trained with MSE loss against the row-normalized max-pooled attention map. Since inference uses top-k block selection, a ranking or top-k recall loss would more directly optimize the downstream objective. This is not a fatal flaw given the strong empirical results, but it is worth investigating.

- **Calibration cost reporting.** The paper states post-training calibration completes in "hours" on 4 A100 GPUs with 500 steps. A clear breakdown of FLOPs or GPU-hours relative to inference savings would help practitioners assess the trade-off for new models.

- **CUDA kernel implementation.** The Triton kernel is compared against a CUDA FlashAttention-2 baseline, so the speedup numbers reflect both algorithmic and implementation differences. The authors acknowledge this and note CUDA as future work; flagging this more explicitly would strengthen credibility of the efficiency claims.

- **Gate entropy / confidence analysis.** Measuring the entropy of AttnGate outputs across different heads and context lengths would reveal whether some heads are systematically uncertain (high entropy), which would indicate instability in block selection at high sparsity.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Causal masking not explained" (Harsh Critic).** The paper's block-sparse kernel explicitly follows FlashAttention-2's dataflow, which already correctly handles causal masking. There is no evidence the causal structure is mishandled.

- **Statistical rigor / confidence intervals (Harsh Critic).** For large-scale LLM benchmarks (LongBench, perplexity), single-run evaluation is the established norm. Demanding confidence intervals is not standard in this community and would impose a non-standard rigor requirement. Removed per reviewer calibration rules.

- **Missing comparisons with H2O, SnapKV, StreamingLLM (Harsh Critic).** These are KV-cache eviction/compression methods, which operate on a different problem (reducing KV cache memory during decoding) from this paper's focus (sparse prefill computation). The comparison set of MoA and MInference, which are direct sparse prefill competitors, is appropriate. Additionally, per review rules, missing related work comparisons should not be flagged when external sources cannot be confirmed.

- **MoA TTFT at 8k slower than FlashAttn-2 (Harsh Critic).** The critic notes MoA is slower than FlashAttn-2 at 8k (1.29s vs 0.90s) and suggests this needs explaining. However, this reflects a genuine weakness of MoA, not of SeerAttention — the comparison is intentionally asymmetric in favor of the baseline. Including this makes the paper's method look *stronger*, not weaker, and per rules, such comparisons should be removed as "unfair comparisons beneficial to the baseline."

- **The K-outlier hypothesis being speculative (Harsh Critic).** The paper presents this as a possible explanation ("may relate to"), not a claim. This is appropriate scientific hedging and is not a weakness.

- **"Well-structured and clearly written" (Positive Reviewer strength).** This applies to any competently written paper and provides no differentiation.

- **"Topic is important / industry push toward 128k+" (Positive Reviewer strength).** Generic significance claim that applies to any long-context LLM paper.

---

## Novel Insights

The most genuinely novel insight in this paper is the decoupling of *supervision extraction* from *attention computation*: by instrumenting FlashAttention's tiled pass to store intermediate row-max values and rescale them post-iteration, the paper shows that block-level attention statistics can be recovered at near-zero overhead without a separate O(n²) forward pass. This is not merely an engineering trick — it enables a training paradigm where a learned gating module receives dense attention supervision at 64k+ contexts, something that was previously impractical. A secondary insight is that applying a block-rescaled RoPE (θ′=θ/B) to the pooled Q/K positions inside AttnGate is both principled and practically critical: Figure 9 shows it is the difference between smooth extrapolation to 128k from 8k training data versus catastrophic perplexity collapse, suggesting that positional encoding fidelity at the block level is a non-trivial concern for any block-sparse attention scheme that needs to generalize across context lengths.

---

## Suggestions

1. **Fix Figure 1b** to show both YaRN Baseline and YaRN w/ SeerAttention on the *same* dataset (e.g., PG19 or Proof-pile). The current figure is misleading and will be flagged by any reviewer who checks the dataset labels.

2. **Add a block size ablation** (B ∈ {32, 64, 128}) in Section 6. Given that B controls both approximation coarseness and tiling granularity, this is essential for understanding the paper's design space.

3. **Add NIAH evaluation** for the post-training models, ideally at both 32k and 128k context lengths at several sparsity ratios. This would provide evidence that the gate does not miss critical positional targets that perplexity cannot capture.

4. **Add LongBench results for the YaRN+SeerAttention fine-tuned model** in Section 5.2 to demonstrate that fine-tuning with sparsity does not degrade instruction-following on real tasks.

5. **Clarify end-to-end vs kernel-level speedup in the abstract** — present both figures, or at minimum label the 5.67× as kernel-level and cite the corresponding end-to-end TTFT numbers.

6. **Ablate the linear projection layer** in AttnGate by testing a no-linear-layer baseline (direct pooled-Q × pooled-K matmul). If the linear layer is necessary, this experiment will make a stronger case for the current architecture.

---

**Axis Evaluations:**

- *Novelty:* Moderate-to-strong. Learning block sparsity is not new, but the memory-efficient customized training kernel and the RoPE-in-AttnGate design are genuinely novel mechanisms.
- *Technical soundness:* Generally sound. The core kernel design is well-explained and the RoPE ablation is rigorous. Some design choices (block size, linear layer) are unvalidated.
- *Empirical support:* Adequate for post-training claims; weaker for fine-tuning claims (perplexity only, no downstream tasks). Absent for NIAH-type retrieval evaluation.
- *Significance:* Meaningful for the long-context LLM inference community. The fine-tuning result (90% sparsity, ~5% perplexity increase) is practically useful.
- *Clarity:* Good overall structure with targeted ablations; Figure 1b and the abstract speedup presentation introduce avoidable ambiguity.

MY FINAL SCORE: <pineapple>6.8</pineapple>

---

## 0YkZe9nwiC

- GT: Reject (avg 3.0)
- Predicted: Reject (4.4/10)
- Match: YES

### Final Review

Now I have sufficient understanding of the paper to synthesize the reviews. Let me produce the final consolidated review.

---

## Summary

SIGnAL (Self-Informed Generative Active Learning) proposes a reinforcement learning-based framework for active learning when the initial unlabeled pool is scarce or biased. The key idea is to fine-tune a large language model (Qwen2.5-7B-Instruct) using PPO, guided by a novel acquisition function that rewards both informativeness (KL divergence between a candidate's and its neighbors' classifier predictions) and relevance (proximity in embedding space). This acquisition function doubles as the PPO reward, so the generator is iteratively steered to produce more informative in-distribution data. Experiments on three text classification benchmarks (SST-2, AGNEWS, QNLI) in two low-data regimes (0.1% and 1% of training data) show consistent improvements over pool-based baselines.

---

## Strengths

- **Relevance-aware acquisition function for generative AL.** The joint KL/distance acquisition function (Eq. 3) directly addresses the well-known failure mode of generative AL—producing uncertain but irrelevant OOD instances—by penalizing high embedding distance to labeled neighbors. This is a specific and concrete design choice that distinguishes the method from naive uncertainty-based generation.
- **Adaptive label-balance recovery demonstrated empirically.** Section 5.4 documents and explains an emergent adaptive behavior: the generator initially biases toward entailment examples on QNLI but, through RL feedback, gradually shifts to producing underrepresented non-entailment instances as entailment data becomes less informative. This qualitative insight about the self-correcting dynamics of RL-guided generation is the paper's most interesting empirical observation.
- **Clean integration of the RLHF paradigm with active learning.** The paper provides a coherent mapping of the generative AL problem onto the RLHF objective (Eq. on line 115), including a KL-regularization term against the pretrained policy to prevent reward hacking. The formalization in Section 3 is crisp and the algorithm (Algorithm 1) is easy to follow.

---

## Weaknesses

- **Missing critical ablation: RL vs. simple generate-and-pool.** The paper's central claimed contribution is that PPO-based generator optimization improves over a fixed generator. Yet there is no baseline that uses the same LLM to generate data and then applies any standard pool-based acquisition function *without* RL fine-tuning. Without this comparison, it is impossible to determine whether observed gains come from the RL optimization or simply from having access to more diverse LLM-generated data. This gap directly undermines the core empirical claim and is the most damaging missing experiment in the paper.

- **Structurally asymmetric comparison.** As stated in Section 5.3, pool-based baselines halt at 100% of the real data budget, while SIGnAL extends to 200%. Although this asymmetry reflects the intended use case (SIGnAL is designed for the regime where the real pool is exhausted), the paper does not provide any analysis at a *fixed annotation budget* where all methods can be compared on equal footing. At 100% budget, baselines have used up all their ground-truth-labeled real data while SIGnAL's 200% includes synthetically annotated instances from a ~91–94% accurate oracle—making the comparison multi-dimensional and hard to interpret. Providing a fixed-budget comparison is essential to establish the method's practical value.

- **Noisy oracle confound is unquantified.** Section 5.3 acknowledges that synthetic instances are labeled by a fine-tuned classifier with 91.3%/93.75%/90.99% accuracy. Because this oracle is trained on the *full* training set, it implicitly injects global label distribution information into the synthetic annotations—information that pool-based baselines do not have access to. The paper notes this but makes no attempt to quantify the performance impact, measure label error accumulation across iterations, or bound how much the oracle advantage contributes to SIGnAL's gains.

- **Acquisition function numerical stability.** Eq. 3 computes the ratio KL(·‖·) / d(Φ(xᵢ), Φ(xⱼ)). When a generated instance lands very close to a labeled neighbor in embedding space (near-zero denominator), the score can grow arbitrarily large and dominate selection. No smoothing, clamping, or minimum-distance threshold is mentioned. This is a potential failure mode in practice that is not addressed.

- **Acquisition function terms not ablated.** The two components of Eq. 3—the KL informativeness numerator and the distance relevance denominator—are never tested individually. It is therefore unknown whether both are necessary or whether the distance term alone (a diversity criterion) drives the gains.

- **Early-stage underperformance in the target regime.** Section 5.4 acknowledges that "SIGnAL tends to underperform compared to pool-based methods during the early stages of training." The paper's stated motivation is precisely the low-budget, early-stage regime. This consistent early underperformance—attributed to the generator initially producing repetitive near-in-context-example instances—is a practically significant failure mode that the paper discusses only briefly without a concrete fix.

- **No PPO hyperparameter details or reproducibility information.** The KL penalty coefficient β, PPO clip range, learning rates, number of RL epochs per AL iteration, and reward scaling are not reported anywhere in the main paper. For a system that relies critically on stable RL training, this severely limits reproducibility.

- **No computational cost analysis.** The abstract claims SIGnAL is "cost-efficient," but the method requires iterative PPO fine-tuning of a 7B-parameter model at every AL round, on top of BERT fine-tuning. No wall-clock time, GPU hours, or cost comparison with pool-based baselines is provided. The claimed cost-efficiency is unsubstantiated.

- **Limited experimental scope.** Experiments cover only text classification on three datasets in two data-scale conditions. The paper claims SIGnAL is a "general framework" for other tasks and modalities, but presents no supporting evidence for this generality. The scope restriction is acknowledged in the conclusion but not presented as a limitation in the body.

---

## Nice-to-Haves

- **Fixed annotation budget comparison.** Evaluating all methods at the same number of total annotations (e.g., 500 labels) with SIGnAL's budget split between real and synthetic data would give a cleaner picture of practical cost-benefit trade-offs.
- **Qualitative examples from early vs. late RL iterations.** Side-by-side displays of generated instances at initialization vs. after RL training would vividly illustrate the claimed adaptive behavior described in Section 5.4.
- **Embedding space visualization.** t-SNE plots of real vs. synthetic data over training iterations would verify that synthetic data fills distribution gaps rather than clustering near existing labeled points.
- **Sensitivity to β.** A plot of performance across values of the KL penalty coefficient would show whether the method is robust to this hyperparameter or requires careful tuning.
- **Comparison with simpler RL alternatives.** Comparing PPO to REINFORCE or rejection sampling (best-of-N) would help justify the engineering cost of full PPO training.
- **Domain generalization test.** Testing on a specialized domain where the LLM's pretraining distribution is weak (e.g., medical or legal NLP) would stress-test the method's dependence on strong LLM priors.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Self-informed" title is misleading.** The harsh reviewer argues the generator is "externally informed" by the classifier. However, the system forms a closed loop where the generator's own outputs are evaluated and fed back as training signal; "self-informed" is a defensible framing and this is a semantic nitpick.
- **PPO is wrong for a single-step bandit.** While technically PPO is designed for sequential MDPs, applying PPO to single-step text generation is the dominant convention in RLHF (e.g., InstructGPT). Criticizing this as methodologically inappropriate ignores community norms.
- **Temporal mismatch in reward computation.** The harsh reviewer flags that PPO updates use rewards from the current iteration while the next iteration's classifier will differ. This is standard in online RL and iterative RLHF, not a paper-specific flaw.
- **Notation overloading (y as response vs. label).** A real notational annoyance but a pure style issue.
- **Comparison unfair because baselines use real labels.** One reviewer flags that baselines have ground-truth labels and SIGnAL has noisy oracle labels as an unfair disadvantage *to SIGnAL*. If anything, this asymmetry favors the baselines, so it cannot be called unfair to the baselines. Per the meta-review rules, comparisons where asymmetry benefits the baseline are not a flaw.

---

## Novel Insights

The most genuinely interesting observation in this paper—and one not explicitly highlighted as a core result—is the *emergent self-correcting label rebalancing* documented in Section 5.4 for QNLI. The RL-trained generator initially collapses onto the mode of its prior (generating predominantly entailment examples), but as the classifier becomes saturated on entailment instances, the acquisition score for entailment examples drops, pushing the generator toward underrepresented non-entailment examples. This demonstrates that an RL-optimized generator can implicitly perform curriculum rebalancing without explicit supervision of the class distribution—a property that pool-based methods, which are limited to whatever distribution exists in the real pool, cannot exhibit. This behavior also reveals the method's key dependency: the mechanism requires enough RL training steps to shift the generation policy, meaning it is slow to correct strong prior biases, and performance in early iterations in biased-prior scenarios will suffer.

---

## Suggestions

1. **Add the no-RL baseline.** Run the same generation loop but replace PPO with either fixed LLM generation or best-of-N rejection sampling using the acquisition function. Report this in Figure 3. This single experiment would substantially strengthen the paper's core claim.
2. **Provide a fixed-budget analysis table.** At a fixed number of total annotations (e.g., N = 200, 500), compare all methods including SIGnAL with its real + synthetic mix, and clearly report the oracle accuracy on synthetic labels. This addresses the 100% vs. 200% comparison ambiguity.
3. **Add a smoothing floor to the acquisition function.** Replace d(Φ(xᵢ), Φ(xⱼ)) with max(d(·,·), ε) for some small ε > 0 and report sensitivity. This is a simple fix that guards against numerical instability.
4. **Ablate KL numerator and distance denominator separately.** One can run SIGnAL with KL-only acquisition (equivalent to CAL in the generative pool) and distance-only acquisition to isolate what each term contributes.
5. **Report all PPO hyperparameters** in a dedicated table in the appendix (β, learning rate, clip range, epochs, batch size, reward normalization scheme).
6. **Quantify oracle error propagation.** Track disagreement between the oracle classifier and ground truth on a held-out synthetic validation set across iterations to show whether label errors accumulate or stabilize.
7. **Report training wall-clock times.** Include a table comparing GPU hours per AL iteration for SIGnAL vs. pool-based baselines. If the overhead is large, discuss lightweight RL alternatives.

---

**Evaluation along key axes:**

- **Novelty:** Moderate. Adapting RLHF to steer generative AL is incremental but applied to a new setting. The relevance-aware acquisition function is the most specific novel contribution.
- **Technical soundness:** Weak-to-moderate. The RL formulation is standard and correct, but the acquisition function has an unaddressed numerical stability issue, and the choice of hyperparameters is opaque.
- **Empirical support:** Weak. The critical RL-vs-no-RL ablation is absent; the comparison budget asymmetry and oracle confound are unresolved; only three datasets in one task type.
- **Significance:** Moderate potential, but current evidence is insufficient to confidently establish that the RL component is responsible for the observed gains.
- **Clarity:** Generally good. Algorithm 1 is clear; Section 5.4 is informative. Missing reproducibility details are a notable gap.

MY FINAL SCORE: <pineapple>4.4</pineapple>

---

## 8bjspmAMBk

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: Accept (5.4/10)
- Match: YES

### Final Review

## Summary
This paper proposes a quality metric for evaluating continuous-time dynamic graph (CTDG) generative models by applying Johnson-Lindenstrauss (JL) random projections directly to dynamic graph event sequences, bypassing the need for static snapshot instantiation. The method embeds variable-length per-node event sequences via two-stage random projection matrices and computes cosine distance between the resulting fixed-dimensional graph representations. A comprehensive empirical benchmark is also introduced, adapting fidelity, diversity, sample efficiency, and computational efficiency evaluations from vision/static-graph literature to CTDGs.

---

## Strengths

- **Directly addresses the snapshot discretization bottleneck.** Existing CTDG metrics universally require explicit static snapshot construction (reported at 8–12 s/100 events); the JL-Metric avoids this entirely, achieving 1.05 s/100 events while remaining sensitive to temporal structure — a concrete improvement for practical model development pipelines.

- **First joint topology+feature metric for CTDGs.** All reviewed baselines (node degree, LCC, NC, PLE, activity rate) are blind to edge features; the JL-Metric is, by construction, sensitive to both. The event permutation experiment in Section 4.1 demonstrates a case where every existing metric returns flat response while the JL-Metric produces a near-perfect correlation of 0.988 — confirming the capability gap.

- **Time perturbation sensitivity is non-trivial.** On the time perturbation task (altering temporal ordering while preserving event identity), the JL-Metric achieves Spearman 0.944 versus the next-best topological metric at 0.927, and beats all other methods — demonstrating that temporal ordering is meaningfully encoded in the representation even without explicit recurrent modeling.

- **Unified scalar output.** The paper correctly identifies an underappreciated practical problem: model rankings produced by competing topological statistics (degree, LCC, PLE) often disagree with no principled tie-breaking. Providing a single scalar is a concrete usability benefit, not just a rhetorical claim.

- **Comprehensive empirical benchmark design.** Adapting fidelity, diversity (mode drop/collapse), and sample efficiency evaluations from Thompson et al. (2022) to the CTDG setting across five datasets and 10 seeds constitutes a reusable benchmark infrastructure that the community currently lacks.

---

## Weaknesses

- **Missing random temporal GNN baseline — this is the most important empirical gap.** The paper's stated motivation is to explain *why* random networks work via the JL lens, then exploit this to build a more efficient version. Yet it never includes a randomly initialized temporal GNN (e.g., a random TGN or TGAT) as a baseline. Without this, it is impossible to tell whether the JL projections are doing something genuinely better than random temporal GNNs, or whether the JL narrative just rebrands the same phenomenon. This absence significantly limits the paper's central empirical claim.

- **Variable-length handling breaks the JL guarantee.** The paper handles nodes with fewer events by "ignoring unused rows of the matrix where necessary" (Section 3), which is functionally zero-padding. The JL lemma (Eq. 2) guarantees distance preservation for fixed-dimensional vectors in ℝ^N; zero-padding shorter vectors changes their norms and distances non-uniformly (sparse nodes are systematically mapped closer to the origin). No corrective argument or citation is provided for why distance preservation still holds in this regime. This gap is not merely pedantic: for graphs with high-degree hubs alongside low-activity nodes (a common real-world pattern), the distortion could be large and systematic. The theoretical contribution of the paper rests substantially on this unvalidated step.

- **Single-sample distribution estimator without variance characterization.** Equation 3 compares the entire real and generated graphs via a single Frobenius cosine similarity. While the stationarity assumption in Section 2.1 is invoked to justify this, no analysis of the estimator's variance is provided. In non-stationary or hub-heavy graphs, a single-sample cosine distance could be noisy, and the paper provides no confidence intervals or theoretical error bounds.

- **No evaluation on actual DGGM outputs.** All experiments use synthetic perturbations of real data rather than outputs from TagGen, TIGGER, Dymond, or TG-GAN (the models motivating the work in Section 1 and 2.2). It is therefore unknown whether the metric produces sensible rankings of real generative models — which is the ultimate application. A controlled generative-model ranking experiment would substantially strengthen the practical claim.

- **Mode-detection protocol depends on a specific trained model (TGN).** Section 4.2 defines diversity modes by clustering TGN memory-bank embeddings via affinity propagation. Different trained models, or different training runs, may yield different clusterings, making the diversity benchmark non-canonical and potentially biased toward the TGN's learned inductive biases. A model-free clustering (directly on event features) would be more principled.

- **Hyperparameter sensitivity not ablated.** Dimensions n and o are selected via grid search (mentioned in the experimental setup, Appendix D), but no ablation of their impact appears in the main paper. Since the JL lemma prescribes n > 8 ln(q)/ε², it is unclear whether the selected values are near the theoretical threshold or if performance degrades sharply below it. This makes robustness of the method difficult to assess.

- **Two-stage projection cascade is unanalyzed.** W₁ projects event sequences to ℝⁿ and W₂ aggregates node embeddings to ℝᵒ. The accumulated approximation error through two composed random projections is never discussed, nor is the choice of n and o's joint impact on the overall distortion bound.

---

## Nice-to-Haves

- Including a randomly initialized temporal GNN (e.g., random-weight TGN) as a baseline in Table 1 would directly address the core theoretical claim and is the most impactful single addition.
- A brief discussion of feature preprocessing for categorical or high-cardinality edge features (Section 3 or Appendix B), since JL distance preservation assumes Euclidean geometry over the raw concatenated vectors.
- A sensitivity ablation of n and o (potentially in the appendix) showing performance vs. JL-theoretic bounds.
- Validation of the multi-graph aggregation setting mentioned in Section 2.1 with even a toy experiment, since the paper claims generality there.
- Correlation of metric rankings with downstream task performance (e.g., link prediction accuracy on generated graphs) would help establish external validity.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **"Efficiency claim is undermined because JL-Metric is 9× slower than Activity Rate"** (Harsh Critic): The paper's efficiency claim is explicitly against *snapshot-based* topological metrics (8–12 s/100 events), where the JL-Metric is 8–11× faster. Activity Rate is a univariate scalar statistic that cannot jointly model topology and features, making it a strawman for the efficiency comparison. The nuance that the JL-Metric is slower than Activity Rate is worth noting (already mentioned as nuance in Table 1), but framing this as undermining the efficiency contribution misrepresents the paper's context.

- **"Event permutation comparison is vacuously favorable"** (Harsh Critic, flagged for removal): The claim that this experiment is "vacuous by construction" overstates the concern. Event permutation specifically tests the capability the paper claims — joint feature-topology sensitivity — and it is methodologically appropriate to include such a targeted test. However, the concern that this experiment cannot distinguish whether the JL-Metric is generally better (vs. being the only metric *designed* to detect this perturbation) has some merit and could be clarified in the paper with a caveat.

- **"Paper is incremental extension of Thompson et al."** (Harsh Critic): The extension to CTDGs with variable-length event handling, structured random matrices, and the full CTDG benchmark infrastructure is substantive. Thompson et al. operates on static graph collections and does not address temporal ordering, variable-length node histories, or the absence of the snapshot assumption. The lineage is clearly acknowledged by the authors.

- **"JL connection is largely heuristic for GNNs"** (as a removal point — the paper itself explicitly acknowledges this limitation): The authors write "no formal theoretical extension of the JL lemma to the static graph domain has been established" (Section 3). The speculative nature of the GNN-JL link is an acknowledged limitation, not a hidden weakness. However, the *variable-length padding* issue (kept above) is a separate and less-acknowledged gap.

---

## Novel Insights

The most substantive novel observation synthesized across reviews is the following: the JL-Metric's theoretical foundation contains a structural asymmetry that creates an interesting tension. The paper uses zero-padding to handle variable-length node histories, which means nodes with very few events (low-degree nodes) are treated as having long zero-tailed histories. This is not merely a theoretical gap — it implies that in graphs with heavy-tailed degree distributions (extremely common in social and biological networks), the metric's geometry is dominated by high-degree hubs, because the zero-padding systematically shrinks low-activity node vectors toward the origin. A rigorous analysis (or even an empirical ablation) of how metric sensitivity scales with degree heterogeneity would both strengthen the theoretical story and expose whether this is a practical concern. The structured random matrix (Hadamard + Rademacher) used here could potentially be adapted to normalize for degree, which would be a natural and theoretically grounded extension.

---

## Suggestions

1. **Add a random temporal GNN baseline.** Initialize a TGN or TGAT with random weights (no training), use its node memory/embeddings as function descriptors, and compare with the JL-Metric in Table 1. This is the most direct test of the paper's theoretical claim and the most impactful addition.
2. **Formally address the zero-padding/variable-length distortion.** Either (a) derive an adjusted distortion bound for zero-padded vectors of varying lengths, (b) adopt degree normalization before projection to correct for this, or (c) clearly frame this as an empirical heuristic with no formal JL guarantee and test its impact via an ablation on datasets with varying degree heterogeneity.
3. **Run the metric on outputs of at least one real DGGM** (e.g., TagGen or TIGGER on any of the four benchmark datasets). Even a single qualitative case study showing the metric produces a sensible ranking relative to human judgment would substantially improve the practical credibility of the work.
4. **Replace TGN-dependent mode clustering with a model-free alternative** (e.g., k-means directly on event feature vectors) and compare diversity benchmark results to assess sensitivity to clustering method.
5. **Add an appendix ablation over n and o** showing performance near and below the JL theoretical threshold n = 8 ln(q)/ε², making it possible to assess whether the empirical behavior aligns with the theoretical prescription.

---

**Evaluation Summary:**
- *Novelty:* Moderate. The CTDG adaptation of the random-network evaluation paradigm, the JL framing for variable-length data, and the comprehensive benchmark design are genuine contributions. The JL-for-GNNs argument remains speculative.
- *Technical soundness:* Weak-to-moderate. The empirical methodology is careful and well-calibrated against prior work, but the key theoretical claim (JL guarantees for zero-padded variable-length vectors) is unsubstantiated.
- *Empirical support:* Adequate for the benchmark comparison but incomplete without real DGGM outputs and the random temporal GNN baseline.
- *Significance:* Moderate-to-good. Evaluation methodology is a bottleneck for the CTDG generation field and this paper provides a practical tool backed by reasonable evidence.
- *Clarity:* Good overall; the variable-length handling explanation (Section 3) is the main dense/under-explained passage.

MY FINAL SCORE: <pineapple>5.4</pineapple>

---

## CvGqMD5OtX

- GT: Accept (Poster) (avg 6.2)
- Predicted: Accept (6.4/10)
- Match: YES

### Final Review

## Summary
CHASE-SQL is a multi-agent Text-to-SQL framework that combines three diverse candidate generation strategies—Divide-and-Conquer CoT, Query Plan CoT, and Online Synthetic Example Generation (OS ICL)—with a fine-tuned pairwise binary selection agent to identify the best SQL query from a candidate pool. The system achieves 73.0% execution accuracy on the BIRD benchmark test set, outperforming the previous best published method by ~5.8% (dev) and all undisclosed leaderboard entries, and generalizes to Spider (87.6%) without any target-domain retraining.

---

## Strengths

- **Large, well-validated SOTA margin on BIRD.** The 73.01% dev / 73.0% test results exceed the next-best published method (Distillery+GPT-4o, 67.21% dev / 71.83% test) by nearly 6 points, and top all undisclosed leaderboard entries. The performance on Spider (87.6%) without any Spider-specific training or prompt tuning is further evidence of robust generalization.

- **Query Plan CoT is a creative, well-motivated reasoning strategy.** Translating the database engine's EXPLAIN output into a human-readable format and using it as a reasoning scaffold directly exploits the structure of the task in a way no prior Text-to-SQL CoT method has done. Appendix Fig. 21 provides a concrete case where this method uniquely succeeds where others fail.

- **Online Synthetic Example Generation (OS ICL) is a genuinely novel ICL contribution.** Rather than retrieving from a fixed pool of demonstrations, the system synthesizes many-shot examples conditioned on the target schema and SQL feature distribution *at inference time*. The approach achieves 68.02% single-candidate accuracy with Gemini 1.5 Pro—the best of the three generators—and is shown to be complementary to the CoT methods via the Venn diagram in Fig. 3a.

- **The pairwise selection agent is rigorously validated.** Table 6 documents a consistent ~6% gain over self-consistency across all three generators and two temperatures, and Table 7 shows the ranker-agent alternative underperforms by 7.5%, directly supporting the design choice of pairwise comparisons. The selection agent's robustness to temperature variation (while self-consistency degrades) is an insightful finding about the interaction between diversity and selection quality.

- **Open-source reproducibility path.** Using Mistral Large + fine-tuned Qwen-2.5-coder, the framework reaches 70.33% on BIRD dev—exceeding all prior published work—providing a meaningful community contribution independent of expensive frontier model access.

- **Generator complementarity is empirically demonstrated.** Fig. 3a's Venn diagram concretely shows that each generator solves questions the other two cannot (35, 38, and 38 exclusive successes respectively), justifying the complexity of maintaining all three pipelines.

---

## Weaknesses

- **Algorithm 1 contains a factual inconsistency.** The paper explicitly states (line 66) that the Divide-and-Conquer strategy generates output "using a **single LLM call**," yet Algorithm 1 shows a sequential decomposition (one decomposition call, one call per sub-question in a loop, one assembly call). This is not a single call. For a core algorithmic contribution, this misstatement needs correction and is not a matter of interpretation.

- **Query Plan CoT test-time mechanism is ambiguous.** The paper describes converting EXPLAIN output into human-readable reasoning steps, but never clearly states whether (a) the LLM is prompted to *generate* a query-plan-style rationale from scratch given the question and schema (purely synthetic reasoning), or (b) an actual SQL query is first generated and run through EXPLAIN to obtain a real execution plan used as chain-of-thought context. If interpretation (b), there is a bootstrapping problem—you need a SQL query to get an EXPLAIN plan before generating the SQL. The appendix prompts and figures are cited but inaccessible to reviewers without the appendix. This ambiguity matters because the mechanism underpins one of the paper's three novel contributions.

- **Selection agent training uses GT hints at training time but not at inference—a train-test distribution mismatch.** Section 2.5 states: "for instances where no correct candidate exists, we include the ground truth SQL query in the prompt as a hint to guide the model in generating correct candidates." At inference, no ground truth is available. The reported 71.01% binary accuracy (Table 5) is measured on pairs generated using this protocol, meaning the model was trained on data partially generated with oracle guidance that is unavailable at test time. The impact of this mismatch on generalization is unaddressed.

- **Correctness of OS ICL synthetic examples is not analyzed.** The synthetic SQL examples injected as few-shot demonstrations are themselves LLM-generated and not validated for correctness. Incorrect examples used as demonstrations could systematically mislead generation. The paper neither measures the error rate of the generated examples nor analyzes whether incorrect examples harm downstream SQL generation quality.

- **Total inference cost is uncharacterized.** Each CHASE-SQL query involves: (a) multi-step OS ICL generation (two synthesis passes), (b) 7 candidates × 3 generators = 21 candidates with multi-step DC CoT, (c) up to β=3 fix iterations per candidate, and (d) Algorithm 3's pairwise comparisons (which doubles all pairs for order-bias mitigation). The total LLM call count per query is large, yet no latency, token count, or API cost estimate is provided. This makes the framework's practical deployability opaque and omits a key trade-off dimension for assessing whether the ~6% gain over self-consistency is cost-efficient.

- **The 9-point gap between oracle (82.79%) and selection agent (73.01%) is undiagnosed.** The paper emphasizes the oracle upper bound as proof of headroom but does not analyze *why* the selection agent fails on ~9% of cases where a correct candidate exists in the pool. Whether failures stem from schema ambiguity, SQL semantic equivalence issues, or specific SQL clause types is unknown. This diagnosis is directly actionable for future improvement.

---

## Nice-to-Haves

- **Compute-normalized comparison with self-consistency.** A comparison between CHASE-SQL and self-consistency given the same total number of LLM calls (rather than the same number of candidates per generator) would clarify whether gains stem from architectural improvements or simply from greater test-time compute budget.

- **Failure mode analysis for 258 unresolved questions.** Fig. 3a shows 258 questions where no generator produces a correct candidate. Characterizing these by difficulty level, SQL features, or domain would help the community understand CHASE-SQL's current limits and guide future work.

- **Query Fixer sensitivity analysis.** β=3 is set without justification; a brief ablation over β∈{1,3,5} would confirm this is not a critical hyperparameter.

- **Selection agent training data statistics.** The number of pairwise training examples, their correct/incorrect distribution, and any class-imbalance handling are unreported, making the selection agent's training difficult to reproduce.

- **Generator-level semantic diversity quantification.** A measurement of how often the 21 candidates (7 per generator) produce semantically distinct execution results (beyond syntactic variation) would strengthen the "diversity" claim and clarify when the pairwise selection adds value over simple deduplication.

- **Releasing model weights and fine-tuning scripts.** Given that independent verification of a 73.0% leaderboard number is not possible from published code alone, releasing selection model weights and prompt templates would substantially increase trust in the result.

---

## Removed Points

*These points are flagged for removal. Treat them with caution — they were raised in sub-reviews but are factually incorrect, apply unfair standards, or constitute style nitpicks.*

- **"CHASE acronym unexpanded"** — Minor formatting nitpick with no bearing on scientific content. Removed per formatting/style rule.

- **"MCS-SQL outperforms CHASE-SQL on Spider (89.6% vs 87.6%)"** — MCS-SQL uses Spider training data; CHASE-SQL does not. The paper explicitly acknowledges this asymmetry ("placing it second among methods that have undergone specific training or prompt optimization for the Spider dataset"). The comparison is intentionally favorable to the baseline to demonstrate stronger generalization, not a weakness.

- **"Weak baseline in Table 4 inflates gains"** — The stated purpose of Table 4's baseline ("original BIRD prompt + zero-shot CoT") is to measure the *isolated* contribution of each new CoT strategy, not to compare against other full systems. Comparing against CHESS's full pipeline would conflate the contribution of other CHESS components. The choice is methodologically appropriate.

- **Statistical significance / confidence intervals** — Single-run evaluation on BIRD and Spider is the established norm for this benchmark community; requesting bootstrap CIs is not a standard expectation and is removed per community standards rule.

- **"Computational complexity of pairwise comparisons is too high"** — While the cost concern is kept as a weakness (uncharacterized cost), the specific criticism that 420 comparisons *by itself* is intractable or disqualifying is not substantiated; the paper should quantify the cost but the design is not inherently unreasonable.

- **"Contributions should be in a bulleted list"** — Style nitpick; contributions are clearly described in the introduction narrative.

- **"Why exactly three generators and not others?"** — The paper provides empirical justification (Venn diagram, complementarity analysis in Fig. 3). Demanding a formal theoretical justification for an empirical systems paper exceeds field norms.

---

## Novel Insights

The most genuinely novel observation synthesized across the reviews is the interaction between **lower-bound performance and selection agent returns**: Fig. 2 reveals that OS ICL has a higher lower bound than the two CoT methods, meaning more of its candidates are uniformly correct, which paradoxically limits the marginal benefit of sophisticated selection for OS ICL compared to CoT methods. This suggests that the optimal strategy for future selection-based systems should explicitly trade off *lower bound diversity* (ensuring not all candidates collapse to the same wrong answer) against *quality* (minimizing invalid candidates). The finding implies that measuring only upper-bound oracle performance is insufficient to predict whether a selection agent will effectively recover gains—the lower bound is an equally important diagnostic. This has implications beyond Text-to-SQL for any system combining diverse candidate generation with learned selection.

---

## Suggestions

1. **Fix the "single LLM call" claim in Section 2.3.** Revise the description of Algorithm 1 to accurately state the number of sequential LLM calls; consider quantifying the average number of sub-questions produced to give readers a practical sense of the call overhead.

2. **Clarify the Query Plan CoT mechanism unambiguously.** Add a sentence explicitly stating whether the query plan is (a) generated synthetically by the LLM as a reasoning format, or (b) obtained from executing EXPLAIN on a preliminary SQL draft. If (b), describe how the initial draft is obtained to resolve the bootstrapping concern.

3. **Assess or mitigate the GT hint train-test mismatch.** Re-evaluate binary selection accuracy on pairs generated *without* GT hints, or report a breakdown showing what fraction of training pairs relied on GT guidance. If a significant fraction relied on GT, retrain without this and report the delta.

4. **Add a cost analysis table.** Provide average LLM call counts and approximate API token consumption per query (perhaps in the appendix) for at least three operating points: single-generator no-selection, three-generator self-consistency, and full CHASE-SQL. This is essential for practitioners.

5. **Add a brief failure case study for the 258 unresolved questions.** Even a coarse breakdown by SQL feature category (e.g., nested aggregation, multi-hop joins) or difficulty level would substantially improve the paper's analytical contribution and guide future work.

6. **Report OS ICL synthetic example error rates.** A small-scale analysis (e.g., on 100 questions) measuring what fraction of synthesized examples contain SQL errors, and how generation quality changes when using validated vs. unvalidated examples, would address a real concern about the method's reliability.

---

### Evaluation on Key Axes

- **Novelty:** Moderate-to-good. The Query Plan CoT and OS ICL are concrete new contributions; pairwise selection via a fine-tuned binary classifier is adapted from preference optimization literature. The primary novelty is in the integration and the specific design of each component for Text-to-SQL.
- **Technical soundness:** Adequate, with notable issues. The Algorithm 1 inconsistency and QP CoT ambiguity are real problems; the GT hint train-test mismatch is a substantive methodological gap that could affect the reported selection accuracy.
- **Empirical support:** Strong for the system-level claims; the ~6% selection advantage over self-consistency is well-supported. Weaker for individual component design choices (e.g., LSH vs. alternatives, β=3 for the fixer, OS ICL example correctness).
- **Significance:** High. BIRD is the most competitive public Text-to-SQL benchmark; a ~6-point improvement over the prior best published method is a meaningful advance with practical implications.
- **Clarity:** Generally good; the framework description and algorithmic presentation are clear, but the QP CoT mechanism and Algorithm 1 contain the inconsistencies noted above.

MY FINAL SCORE: <pineapple>6.4</pineapple>

---

## 9GsgCUJtic

- GT: Accept (Spotlight) (avg 7.3)
- Predicted: Accept (6.8/10)
- Match: YES

### Final Review

## Summary
This paper investigates three interconnected questions about GFlowNets: (1) how balance violations propagate to affect distributional accuracy (TV bounds in Theorem 1, Weighted DB loss), (2) what expressiveness limits GNN-based GFlowNets face when sampling from graph distributions (Theorems 2–4, LA-GFlowNets), and (3) how to tractably and reliably assess GFlowNet correctness (the FCS metric, Theorem 5). Together, the contributions provide a principled theoretical framework for understanding *when* GFlowNets succeed, along with practical methodology for training and evaluation.

---

## Strengths

- **Novel TV sensitivity analysis connecting local flow imbalance to global distributional error.** Theorem 1 establishes tight bounds showing that balance violations near the root of the state graph have disproportionately larger impact than those near leaf states, formalized for arbitrary DAGs and multimodal rewards. This non-obvious heterogeneity is empirically confirmed in Figure 3 across four benchmark tasks, and translates directly into the WDB design principle.

- **Compelling impossibility result (Theorem 3) and targeted remedy (LA-GFlowNets).** The construction in Figure 5 is compact yet illustrative: two 1-WL-indistinguishable actions leading to children with different subtree rewards provably cannot be resolved by any 1-WL GFlowNet. The LA-GFlowNet formulation (Eq. 7) is a minimal and theoretically justified extension — adding child-state embeddings — that provably overcomes this expressiveness barrier (Theorem 4). The insight that a widely-used class of policy networks has a structural blind spot is practically important for graph-domain applications.

- **FCS as a computationally tractable and theoretically grounded evaluation metric.** FCS achieves Spearman correlation of 0.99 (sets) and 0.90 (sequences) with TV distance while being up to three orders of magnitude faster to compute (Figure 7). Theorem 5 provides the right faithfulness property (FCS=0 ↔ TV=0), and the metric's relationship to ratio matching and TV is cleanly characterized through the β interpolation.

- **Case study in Section 5.2 exposing a critical methodological flaw in prior evaluation.** The demonstration that terminally-unrestricted LED- and FL-GFlowNets attain perfect Shen accuracy (100 ± 0.00, Table 2) and outperform standard GFlowNets on exploration metrics while being provably distributionally incorrect is striking. Proposition 1 identifies the precise theoretical cause. This finding is practically important for the GFlowNet evaluation community and goes beyond a generic "standard metrics are imperfect" complaint.

---

## Weaknesses

- **Theorem 1 covers only a single, localized perturbation; the multi-perturbation regime is unaddressed.** In a trained GFlowNet, balance violations occur simultaneously at many edges. No result bounds the cumulative effect of multiple imbalances — even a triangle-inequality-style additive bound would partially address this gap. Without it, the theorem can characterize sensitivity to individual imbalances but cannot directly predict the total distributional error from training, which limits the direct practical impact of the theoretical result.

- **LA-GFlowNets are only validated on narrow synthetic experiments; no real benchmark evaluation.** Figure 6 tests four triples with n=8, k=3, binary rewards, and noiseless settings. No experiment evaluates LA-GFlowNets on any of the paper's own benchmark tasks (sequences, phylogenetics, sets, hypergrid), let alone on molecule generation. For a methodological contribution at ICLR, this is a notable gap: practitioners cannot assess whether the expressiveness gain is practically meaningful or computationally feasible on tasks of realistic scale.

- **Theorem 3's tree-structured SG assumption is not addressed for DAGs.** The paper's impossibility result formally requires the state graph to be a directed tree. Many real GFlowNet applications (e.g., set generation, hypergrid, molecule generation) use DAG-structured state graphs. Whether the impossibility extends to DAGs — or whether additional paths in a DAG provide workarounds for a 1-WL-based policy — is not discussed, leaving an important theoretical gap.

- **FCS coverage concern: the metric may be blind to modes that the learned policy underrepresents.** FCS computes TV on subsets drawn from trajectories sampled from the current policy. If the GFlowNet has assigned near-zero probability to a portion of the support, those states are systematically absent from the subsets, so FCS may appear small even when the GFlowNet has significant mode coverage failure. This is precisely the failure mode most important to detect. The paper does not discuss this limitation, and the PAC bound in Corollary 2 contains the term (#X / 2β) · max|p_T(S) − π(S)|, which could render the bound vacuous for large state spaces — also not addressed explicitly.

- **The implicit critique of Pan et al. (2023a) and Jang et al. (2024) is asserted in the main text but not substantiated there.** The paper writes "we have significant reasons to believe that an unrestricted F(x) was a part of some experiments in the original works of Pan et al. (2023a) and Jang et al. (2024)" and defers to Appendix E.3. If this claim is supported by evidence, it should be presented prominently; if it is speculative, the language in the main text is too assertive and potentially unfair to the cited authors.

- **Computational cost and scalability of LA-GFlowNets are not reported.** Computing the embedding of every child state requires evaluating the GNN on the successor graph for each candidate action, adding cost proportional to the branching factor. For tasks with large action spaces (e.g., molecular graphs), this overhead could be prohibitive. No runtime analysis, memory profile, or discussion of how to scale or approximate child embeddings is provided.

- **WDB's weighting (γ = 1/#D_{s'}) requires counting terminal descendants, which is intractable for large DAGs.** The paper evaluates WDB only on benchmarks where this counting is feasible. For the primary target application (molecule generation), enumerating #D_{s'} is generally intractable. The paper acknowledges this in limitations but provides no approximation strategy, which significantly limits the portability of WDB to the applications that most need faster convergence.

---

## Nice-to-Haves

- Sensitivity analysis of WDB to the choice of γ (e.g., comparing inverse-descendant-count to exponential decay or depth-based functions) would help practitioners understand how to adapt WDB to new domains.
- An ablation study varying β in FCS across different state space sizes would characterize how to tune the metric for large-scale settings.
- Visualization of the distribution of WDB weights γ across trajectory depths for a complex task would visually confirm the theoretical claim that early transitions dominate.
- Evaluation of WDB combined with LA-GFlowNets on at least one shared task, since real failures likely involve both imbalance and expressiveness deficits simultaneously.
- A stochastic sampling strategy for LA-GFlowNet child embeddings (sampling a subset of children rather than enumerating all) would make the approach tractable for large branching factors.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"WDB only clearly helps 2/4 tasks — claim of 'often accelerates' is overstated."** The paper's own Section 3.2 explicitly explains the conditions under which WDB helps ("Note these two environments are exactly the ones for which early-stage transitions dominate the loss, as shown in Figure 3"), and "approximately on par" for the other two tasks is not a failure. The result is coherent and the claim is defensible. The harsh reviewer conflated a nuanced finding with a misleading one.

- **"The introduction's claim of 'up to three orders of magnitude less compute' is only valid for small state spaces."** The paper qualifies this as comparing to exact TV computation, which is the baseline being discussed. The claim is accurate within the stated scope; it is not claimed to apply when exact TV is also intractable.

- **"FCS sensitivity to β is not explored — this is a significant flaw."** This is a reasonable suggestion for future analysis but does not undermine the core validity of FCS as a metric. Theorem 5 holds for any β ≥ 2, and the empirical correlations in Figure 7 are strong. Requesting an exhaustive ablation is more of a nice-to-have.

---

## Novel Insights

The most genuinely novel observation synthesized across the reviews is the coupling between *measurement failure and methodological failure*: the paper shows that commonly used evaluation metrics (Shen's accuracy, top-k reward) can assign perfect scores to models that are provably distributionally incorrect, and does so concretely by linking the theoretical pathology (Proposition 1: unconstrained terminal flows yield marginals proportional to R(x)·F̃(x) rather than R(x)) to the practical metric failure (Table 2). This creates a compounding problem — not only can GFlowNets fail silently due to GNN expressiveness limits or balance violation propagation, but the field's standard diagnostic tools may not detect the failure. The combination of Theorem 3, Proposition 1, and Table 2 constitutes a coherent cautionary argument that goes beyond any individual component.

---

## Suggestions

1. **Extend Section 5.2 or add an appendix section with the full evidence regarding Pan et al. and Jang et al.** Either present the evidence from E.3 in the main text or tone down the main-text claim to "we present evidence in Appendix E.3 suggesting..." This affects both the paper's credibility and fairness to cited authors.
2. **Add at least one real-graph benchmark for LA-GFlowNets** (e.g., the phylogenetics or molecule generation tasks) with runtime reporting to establish practical viability. The current synthetic-only validation significantly limits the contribution's reach.
3. **Explicitly discuss the FCS coverage limitation** (modes underrepresented by the current policy) and, if possible, propose or discuss a remediation (e.g., using an exploratory backward policy or a mixture policy for subset construction).
4. **Provide an approximation or upper-bound strategy for #D_{s'}** in WDB for DAG-structured state graphs, even as a heuristic, to extend the method's applicability beyond enumerable benchmarks.
5. **Discuss the cumulative balance violation case**, even informally — a simple observation that the single-perturbation bound can be applied repeatedly via a union-bound-style argument (with known pessimism) would substantially increase the theorem's practical interpretability.

---

**Novelty:** High — the combination of TV sensitivity analysis, GNN expressiveness impossibility for GFlowNets, and the FCS metric addresses genuinely underexplored territory in the GFlowNet literature, and each piece is technically original.

**Technical soundness:** Good — all main theorems appear correct and the proofs cover the stated claims. The primary caveat is that Theorem 1 and Theorem 3 have scope restrictions (single perturbation; tree-structured SG) that are not always clearly foregrounded.

**Empirical support:** Moderate — strong for WDB and FCS on standard benchmarks, but LA-GFlowNets are only validated synthetically. The coverage of FCS on large-scale tasks is argued theoretically but not demonstrated empirically.

**Significance:** Above average — the FCS case study and expressiveness impossibility results are of direct practical value; the WDB contribution is incremental but useful.

**Clarity:** Good — Table 1 provides an unusually effective roadmap for a multi-contribution paper, and the theorems are precisely stated. The logical thread connecting the three sections is thematic rather than technical, but the paper is transparent about this.

MY FINAL SCORE: <pineapple>6.8</pineapple>

---

## HJp1g4w1Or

- GT: Reject (avg 4.0)
- Predicted: Reject (4.3/10)
- Match: YES

### Final Review

## Summary
This paper applies adversarial harmonization (an ADDA-style framework from Dinsdale et al., 2021) to MEG-based speech decoding, claiming to be the first feature-level deep learning harmonization for MEG neuroimaging data. Two models are evaluated: Brainmagick (Défossez et al., 2023) and MEGalodon (Jayalath et al., 2024), pooling across four MEG datasets. Results are clearly positive for Brainmagick but decidedly mixed for MEGalodon, with harmonization hurting speech detection while marginally improving voicing classification. As a side contribution, the authors release an open-source PyTorch/Lightning reimplementation of Brainmagick.

---

## Strengths

- **Statistically significant, cross-dataset improvement for Brainmagick:** Adversarial harmonization achieves 71.0% ±0.2 top-10 accuracy on the Gwilliams split and 68.6% ±0.2 on MOUS, versus 68.8% and 66.8% for naive pooling, confirmed significant at p<0.05 (one-sided t-test over 3 seeds). Crucially, the harmonized model outperforms the original single-dataset baselines (70.7%, 68.5%), demonstrating genuine benefit from cross-dataset pooling when domain shift is addressed.

- **Empirical evidence for age as a strong confound in MEG decoding:** The controlled comparison of balanced vs. random subsets in Table 3 provides concrete, quantified evidence that participant age distributions across datasets significantly affect model behavior and domain separability — a finding that has practical implications for how the neuroimaging community designs and pools studies.

- **Quantitative domain alignment evidence beyond t-SNE:** The domain classifier accuracy reduction from 99.9% to 79.7%/67.9% (full dataset/subset respectively) for Brainmagick provides a direct numeric measure of harmonization effectiveness, not just qualitative visualization.

- **Open-source Brainmagick reimplementation with verified bug fix:** Replacing internal Facebook Research tooling (Flashy/Dora) with standard PyTorch/Lightning and fixing a sensor-labeling bug meaningfully lowers the barrier to entry for the field. The corrected implementation still reproduces baseline performance within ~1%, establishing its reliability.

---

## Weaknesses

- **Abstract overclaims for MEGalodon:** The abstract states "We successfully improve the performance of both models when training across multiple datasets." However, Table 3 shows that dataset harmonization *reduces* speech detection from 57.29% to 55.04% (best case), and the voicing improvement is only 0.05 percentage points (52.65% vs. 52.60% control) — within any plausible noise margin. The warm-up-only condition (57.76%) actually outperforms all harmonized variants on speech detection. This overclaim is not minor: it misrepresents the central finding for one of the two models.

- **MEGalodon fine-tuning evaluated on only 3 subjects (Armeni dataset):** The paper explicitly notes the Armeni dataset contains three subjects, yet all MEGalodon fine-tuning conclusions rest on it exclusively. Any performance differences in Table 3 (e.g., 0.05% for voicing) are statistically meaningless at this scale. No confidence intervals or significance tests are reported for Table 3, in contrast to Table 2 — making it impossible to distinguish signal from noise. This undermines all MEGalodon-related claims.

- **No comparison to simpler harmonization baselines:** The paper compares adversarial harmonization only against naive pooling and a pre-training scheme, but provides no comparison to ComBat (the standard neuroimaging harmonization tool), z-score normalization per dataset, or other lightweight domain adaptation approaches. Without such baselines, it is impossible to determine whether the adversarial complexity is necessary or whether simpler approaches would suffice.

- **Training instability acknowledged but unresolved:** The paper admits adversarial harmonization is "extremely unstable, with task loss diverging sharply when the harmonization phase begins" and that "equivalent hyperparameter testing" for age harmonization could not be completed. The best speech detection result (57.76%) actually comes from the warm-up-only condition, suggesting the adversarial phase itself is counterproductive for MEGalodon's primary task. This leaves the reader unsure whether positive MEGalodon results are reproducible or lucky survivors of instability.

- **Scope limited to ~15% of available subjects:** Computational constraints restrict experiments to approximately 15% of subjects per dataset, yielding roughly 30–96 subjects per dataset. The "big data" motivation of the paper requires demonstrating that performance scales with pooling large datasets, yet the experiments demonstrate this at only small scale. The paper acknowledges this limitation, but it significantly weakens the generalizability claims.

- **Shallow vs. deep fine-tuning explanation is post-hoc and untested:** Section 5's explanation that harmonization hurts speech detection (shallow fine-tuning) but helps voicing (deep fine-tuning) because of the protocol difference is entirely speculative. The hypothesis is plausible but is presented as an explanation without any experimental verification (e.g., switching the fine-tuning protocol to confirm the causal mechanism).

---

## Nice-to-Haves

- Ablation of σ=10 for the Gaussian spreading in age binning; the choice is not justified and the sensitivity to this parameter is unknown.
- Ablation of α=0.25 scaling factor for aggregated domain classifier losses in MEGalodon.
- Quantitative domain alignment metrics (e.g., MMD, proxy A-distance) to complement t-SNE visualizations.
- Loss convergence curves across seeds to characterize training instability more precisely and help practitioners reproduce stable runs.
- Confusion matrices or per-class accuracy breakdowns for speech/voicing decoding to understand which phoneme categories benefit or are harmed by harmonization.
- A leave-one-dataset-out evaluation to more rigorously assess cross-dataset generalization.
- Explicit fine-tuning protocol ablation (freezing vs. unfreezing encoder for speech detection) to empirically test the shallow/deep hypothesis rather than leaving it speculative.

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **[REMOVED — style/nitpick] Title is misleading:** The harsh critic argues "representations of speech" implies self-supervised representation learning. This is a reading preference issue; the paper's title is interpretable as "MEG-based speech decoding features," not strictly representation learning in the SSL sense.

- **[REMOVED — addressed by paper] Single-GPU vs. multi-GPU performance gap:** The paper explicitly notes and accounts for the GPU-count effect on contrastive loss scaling, and uses single-GPU results as its primary comparison baseline. This concern is adequately addressed.

- **[REMOVED — factual check on Armeni subject count] "3 subjects" as a new criticism:** The harsh critic is correct that Armeni has 3 subjects (confirmed: "three subjects each listening to 10 hours of speech"). This is kept as a legitimate weakness above.

- **[REMOVED — scope creep] Criticizing lack of methodological novelty as a standalone ML contribution:** The paper's target audience includes both the ML and neuroimaging communities. Within the paper's stated scope — demonstrating the first feature-level harmonization for MEG — algorithmic novelty at the level of inventing a new domain adaptation method is not required. The critique that "novelty lies almost entirely in application" is valid context but should not be the sole basis for rejection, given the genuine domain-specific value.

- **[REMOVED — non-standard requirement] Requesting theoretical proofs or formal convergence guarantees for adversarial harmonization:** This is an empirical systems paper; theoretical proofs are not standard expectations in this setting.

- **[REMOVED — minor engineering nitpick] α=0.25 called "ad hoc":** The α=0.25 value is directly motivated by the paper's four-feature-vector design (3 pretext tasks + original input). Treating it as an unmotivated ad-hoc constant is unfair.

- **[WEAKENED → Nice-to-Have] Open-source implementation not being a "scientific" contribution:** While it is unusual to list an implementation as a primary scientific contribution at ICLR, it is a genuine practical contribution with community value (bug fix + accessibility), and is appropriately scoped as such by the authors.

---

## Novel Insights

The most genuinely novel observation in this work — beyond the paper's technical results — is the empirical demonstration that participant age distribution is a dominant source of apparent "dataset bias" in MEG speech decoding: the MEGalodon control performs better when subsets are age-balanced, and this effect is strong enough that removing it partially decouples the domains even before adversarial training. This points to a previously underappreciated confound in cross-dataset MEG studies and has concrete implications for study design (targeted recruitment of older participants). The secondary insight — that deep fine-tuning (unfreezing the encoder) can recover task-specific performance lost during harmonization, while shallow fine-tuning cannot — is plausible and important for practitioners applying harmonization to pretrained models, though it remains unverified experimentally and should be treated as a hypothesis to test rather than an established finding.

---

## Suggestions

1. **Correct the abstract.** "We successfully improve the performance of both models" overstates MEGalodon results. The abstract should honestly characterize the differential outcomes across models and tasks.

2. **Add statistical testing for Table 3.** At minimum, run Table 3 over 3 seeds with confidence intervals, as done for Table 2. Even 3 seeds would allow readers to determine whether the voicing improvement is significant or noise.

3. **Include ComBat or a simpler DA baseline.** A single run of ComBat or z-score-per-dataset normalization applied before standard pooling would establish whether adversarial complexity adds anything over trivial corrections.

4. **Experimentally verify the shallow/deep fine-tuning hypothesis.** Run speech detection with both encoder and task head unfrozen and report the result. If this recovers performance, it strongly validates the paper's main explanatory claim for the MEGalodon results.

5. **Flag the 3-subject evaluation as a named limitation.** The Limitations section should explicitly state that the Armeni evaluation is insufficient for statistical reliability, and propose that future work replicate on a larger held-out dataset (e.g., a held-out MOUS subset).

---

## Evaluation

- **Novelty:** Moderate-low from an ML perspective (direct application of Dinsdale et al., 2021 to a new modality with minor engineering choices); moderate from a neuroimaging perspective (first MEG application of this class of harmonization).
- **Technical soundness:** Adequate for the Brainmagick experiments; weak for the MEGalodon experiments, which suffer from a 3-subject evaluation, missing significance testing, and an acknowledged failure to complete hyperparameter search.
- **Empirical support:** Strong for Brainmagick (statistically significant, multiple seeds, scaling confirmed). Poor for MEGalodon (no CIs, 3 subjects, primary task harmed by harmonization).
- **Significance:** Meaningful for the MEG/BCI/neuroimaging community, particularly the age confound finding. Limited for the core ML community as a methods contribution.
- **Clarity:** Generally clear in motivation and high-level description; the "shallow" vs. "deep" fine-tuning distinction is introduced only in the discussion without being defined in methods, and some computational details (four-pass training cost for MEGalodon) are underspecified in the main text.

Overall, this is a paper with a credible contribution for one model and a weak, overinterpreted contribution for the other. The Brainmagick half meets a reasonable publication bar; the MEGalodon half does not, yet is used to support a central claim in the abstract. The paper would need significant revision — particularly honest characterization of MEGalodon results, statistical testing for Table 3, and at least one simpler baseline — to be considered strong enough for ICLR.

MY FINAL SCORE: <pineapple>4.3</pineapple>

---

