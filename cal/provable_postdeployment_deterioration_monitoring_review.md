=== CALIBRATION EXAMPLE 6 ===

# Final Consolidated Review
Now I have a complete picture of the paper. Let me write the consolidated review.

---

## Summary

This paper formalizes **Post-Deployment Deterioration Monitoring (PDD)** in the unsupervised setting via a new concept, **Disagreement-based PDD (D-PDD)**, which detects deterioration by finding an auxiliary model that agrees with the deployed classifier on training data but increasingly disagrees on deployment data. The proposed algorithm, **D-PDDM**, decouples pre-training (computing a hypothesis set $\mathcal{H}_p$ and an empirical quantile distribution $\Phi$ of max-disagreement rates) from deployment (a one-shot max-disagreement test), so that monitoring requires **no access to raw training data** during deployment. The paper provides FPR and TPR guarantees and validates the framework on synthetic, CIFAR-10.1, and a large-scale real-world healthcare dataset (GEMINI).

---

## Strengths

- **Precise and novel problem formalization.** The distinction between *deteriorating* and *non-deteriorating* shifts is rigorously encoded in Definition 2 (D-PDD), and Lemma 2.1 establishes the conditions under which this surrogate is equivalent to the ground-truth PDD (Definition 1). No prior work in the disagreement-monitoring literature simultaneously provides FPR *and* TPR guarantees under both shift types (confirmed by Table 1, which accurately compares to prior art).

- **Training-data-free deployment via a principled decoupling.** The two-stage protocol—compressing all training-data dependence into $\mathcal{H}_p$ and $\Phi$ during pre-training, then running a purely label-free and training-data-free test at deployment—is a clean architectural insight that directly addresses a real regulatory and scalability constraint not addressed by any existing disagreement-based method.

- **Transparent treatment of failure modes.** Section 4.3 (Regime 2, Theorem 4.5) honestly characterizes conditions under which D-PDDM cannot achieve high TPR, provides geometric intuition (Figure 3), and offers a principled (if imperfect) remedy. This level of honesty in a theory paper is commendable and rare.

- **Real-world healthcare validation.** The GEMINI experiments introduce a large-scale clinical dataset with both a temporal (non-deteriorating) and an age-stratified (deteriorating) split. The fact that D-PDDM achieves low FPR on the temporal shift while outperforming baselines on the age shift provides complementary evidence that is absent from comparable work, which typically relies on benchmark vision datasets only.

- **Proposition 4.1 ($\xi = \mathrm{TV} - 2\eta$).** This equality—relating the degree of D-PDD to TV distance minus an error-gap term—is a non-trivial and interpretable result. If the proof (in the appendix) is correct, it is a genuinely useful characterization absent from prior disagreement-based analyses.

---

## Weaknesses

### Fatal
None.

### Major

- **Theory-to-practice gap with VC dimension (Corollary 4.3 / Theorem 4.4).** The FPR bound in Corollary 4.3 is $\alpha + (1-\alpha)\mathcal{O}(\exp(-n\epsilon_0^2 + d))$, where $d = \mathrm{VC}(\mathcal{H})$. For neural networks used in practice, $d$ is enormous (scales with number of parameters), making this additive term non-negligible unless $n \gg d/\epsilon_0^2$—a condition essentially never satisfied in the deep-learning regime. The paper partially acknowledges this by (a) mentioning linear/forest models in the "Practical insights" and (b) restricting experiments to networks of $\approx 32$ hidden nodes. However, this restriction is buried in Section 5.1 and its implications are not clearly stated: the theoretical guarantees apply only to very small networks, not to any modern ML model deployed in the healthcare or vision settings the paper motivates. The introduction leads with healthcare AI as a primary motivation, but the theory does not cover that regime. This discrepancy needs explicit acknowledgment (e.g., a dedicated Limitations paragraph that separates empirical findings from provably covered model classes).

- **Figure 5(b) axis mislabeling / contradictory claim.** Figure 5 presents the GEMINI temporal shift, which the paper establishes (via Figure 5(a)'s stable AUROC) as **non-deteriorating**. The y-axis of Figure 5(b) is labeled "TPR@5%." In a non-deteriorating setting, there are no true positives; the relevant metric is **FPR**. The figure caption correctly says "D-PDDM is robust with small False Positive Rate (FPR)," but this contradicts the axis label "TPR@5%." If D-PDDM is achieving 0.8–0.9 on an axis labeled "TPR" for a non-deteriorating scenario, that reads as 80–90% FPR—opposite of the paper's claim. Either the axis label is wrong (should be "FPR@5%") or the scale/direction is inverted. This ambiguity directly affects interpretation of a key empirical result and must be corrected.

- **Missing baselines in Table 2 (CIFAR10.1).** The experimental setup (Section 5.1) names six baselines, including BBSD (Lipton et al., 2018) and RMD (Ren et al., 2021), but Table 2 reports results for only four (MMD-D, H-divergence, JS-divergence, KL-divergence). No explanation is given for why BBSD and RMD are absent from this key quantitative result. Their omission, without justification, raises concerns about selective reporting.

- **Theorem 4.4 denominator can become vacuous for mild deterioration.** The sample complexity for high TPR (Eq. 13) has denominator $\xi - 2\epsilon_f$. From Eq. 9, $\xi \geq \epsilon_q - \epsilon_p \geq \xi - 2\epsilon_f$, so for the bound to be defined one needs $\xi > 2\epsilon_f$. For a realistically non-trivial base error (e.g., $\epsilon_f = 0.1$), the method requires deterioration $\xi > 0.2$ just for the TPR bound to be non-trivial—meaning **subtle but important deterioration events are not covered** by the theory. This condition should be stated explicitly and its practical significance discussed.

### Minor

- **Multi-class / binary theory mismatch.** The formal framework (all definitions, lemmas, theorems) is strictly binary ($\mathcal{Y} = \{0,1\}$), but both the CIFAR-10.1 and GEMINI experiments involve multi-class or multi-output tasks. The paper does not discuss how disagreement is aggregated in the multi-class case or whether the provable bounds transfer. At minimum, a clarification of the aggregation strategy and a comment on validity of the guarantees is required.

- **Computational intractability of the core optimization.** Both Algorithm 1 (line 5) and Algorithm 2 (line 2) require $\arg\max_{h \in \mathcal{H}_p} \widehat{\text{err}}(h; \mathcal{D}^m)$, which is non-convex for neural networks. The main text only briefly mentions a Bayesian approximation with a pointer to "Appendix B.1." This is a central algorithmic detail—without it the algorithm is not reproducible. At least a brief description of the posterior sampling scheme should appear in the main paper.

- **No ablation on pre-training rounds.** The paper uses 500 pre-training rounds but provides no analysis of how many are needed for $\Phi$ to reliably estimate the true quantile. Too few rounds inflate FPR during deployment, but no guidance or sensitivity analysis is given. This is particularly important since the theoretical guarantees do not cover the finite-round approximation of $\Phi$.

- **$g = g'$ assumption not prominently advertised.** Lemma 2.1 requires identical labeling functions in training and deployment—i.e., the theoretical equivalence between PDD and D-PDD **excludes concept drift**, arguably the most dangerous deployment failure mode. This restriction appears in the body of Lemma 2.1 but is never stated in the abstract or introduction. Concept drift is distinct from covariate shift and should be flagged as out-of-scope at the outset.

### Tiny

- The notation $Q_f$ (distribution over $Q_x$ with pseudo-labels from $f$) is crucial throughout but only implicitly defined in Section 2. A boxed definition at first use would improve readability.
- The practical implication paragraph in Section 4.3.1 advises "train a better base classifier $f$" to escape Regime 2, but gives no quantitative guidance on how much improvement in $\epsilon_f$ suffices. A brief discussion of what is achievable in practice would strengthen this guidance.

---

## Nice-to-Haves

- **Sequential / online extension.** The current test is batch-based (a fixed-size window of $m$ samples). Extending to a sequential change-point detection framework would be a natural and high-impact next step for continuous deployment monitoring, which is the realistic operational setting.

- **Ablation on $\mathcal{H}_p$ construction diversity.** The method's power depends on $\mathcal{H}_p$ containing diverse functions. If $\mathcal{H}_p$ collapses to be nearly identical to $f$, the disagreement signal vanishes. An ablation varying the posterior sampling diversity (e.g., different seeds, dropout temperatures) would strengthen confidence in the framework's robustness to this practical concern.

- **Empirical characterization of Regime 2 frequency.** It would be informative to quantify, across the real-world datasets, how often real-world shifts fall into Regime 2 (deteriorating but $\epsilon_q \leq \epsilon_p$). This would help practitioners understand when the method's guarantees apply in practice.

- **Ablation on base model error $\epsilon_f$.** Varying the quality of the base classifier $f$ and measuring the empirical TPR/FPR transition between Regime 1 and Regime 2 would concretely validate Figure 3's geometric intuition and the paper's claim that "training a better $f$" is a viable remedy.

- **Disagrement density visualization.** Plotting the full distribution of disagreement scores on $P_f$ versus $Q_f$ (not just summary statistics) would make the separability of these distributions visually apparent and add intuition for why the quantile-based test works.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: Baseline comparison is unfair because baselines weren't designed for deterioration.** REMOVED. The comparison is asymmetric in a direction that *favors* the baselines: they are given oracle access to the generating distributions for permutation testing (as acknowledged in Section 5.1), while D-PDDM is not. This is intentionally baseline-empowering and constitutes a stronger demonstration of D-PDDM's advantage, not a flaw. The fact that D-PDDM still outperforms is a stronger result.

- **Harsh Critic: Privacy benefit is overstated.** REMOVED. The paper's claim is precise: monitoring does not require raw training data post-deployment. That $\mathcal{H}_p$ is derived from training data is no different from any model's weights being derived from training data—it does not constitute access to training data at deployment time. The regulatory argument (Mühlhoff 2023) is about not requiring ongoing access to sensitive records during the monitoring phase, which D-PDDM genuinely satisfies.

- **Harsh Critic: "$\xi = 0$ for non-deteriorating shift stated without justification."** REMOVED. The argument is straightforward: for non-deteriorating shift, all $h \in \mathcal{H}_p$ satisfy $\mathrm{err}(h; Q_f) \leq \mathrm{err}(h; P_f)$, hence $\xi \leq 0$; but $f \in \mathcal{H}_p$ gives $\mathrm{err}(f; Q_f) = \mathrm{err}(f; P_f) = 0$, so $\xi \geq 0$. Thus $\xi = 0$. The paper correctly states this; the derivation is implicit but takes one line.

- **Harsh Critic / Balanced: Regime 2 is "undetectable" and remedy is circular.** WEAKENED (removed as a standalone major criticism). The paper honestly flags this as a known failure mode with no available diagnostic—Theorem 4.5 is itself the formal acknowledgment. The criticism that "retraining requires knowing you are in Regime 2" is fair, but the remedy (lower $\epsilon_f$ through better training before deployment) is a reasonable design-time recommendation, not a circular argument. The absence of a deployment-time Regime 2 detector could be mentioned as a nice-to-have but is not a fatal flaw.

- **Harsh Critic: Storage cost of $\mathcal{H}_p$ "can be larger than the training dataset."** REMOVED as a standalone weakness. While accurate in principle, whether 500 small neural networks (≈32 hidden nodes each) exceed training dataset storage depends entirely on the dataset size. For GEMINI (large-scale hospital data), the 500 small models are almost certainly smaller than the raw training data. The concern is not universally valid.

- **Harsh Critic: Synthetic data baseline given oracle access → results hard to interpret.** REMOVED. The paper explicitly acknowledges and frames oracle access as empowering the baselines (Section 5.1). This is a deliberate choice that strengthens D-PDDM's demonstrated advantage, not a methodological flaw.

---

## Novel Insights

The most genuinely novel theoretical contribution is the decoupling of the monitoring problem into (i) a pre-training stage that compresses all training-data information into a hypothesis set and an empirical disagreement quantile distribution, and (ii) a deployment-time test that requires only unlabeled test data. The resulting FPR guarantee (Corollary 4.3) is independent of the test sample size $m$ (the additive term decays in training samples $n$), which is an unusual and practically important inversion of the typical sample-complexity dependence. Proposition 4.1 ($\xi = \mathrm{TV} - 2\eta$) further offers a precise decomposition of the degree of D-PDD into distribution-level shift and hypothesis-class-level error gap, which could serve as a useful theoretical tool for future work on adaptive monitoring thresholds or model selection for monitoring.

---

## Suggestions

1. **Fix or clarify Figure 5(b).** If the y-axis measures FPR (as the caption claims), relabel it "FPR@5%." If it measures something else, provide a precise definition in the caption. This is essential for reproducing the GEMINI results and for the figure to support the paper's claims.

2. **Add BBSD and RMD to Table 2.** If these methods were excluded for a principled reason (e.g., they require a different setup for CIFAR10.1), explain why in the text. If excluded without reason, include them.

3. **Add a Limitations section.** Explicitly state that: (a) theoretical FPR/TPR guarantees apply to hypothesis classes of manageable VC-dimension (e.g., small neural networks, linear/forest models), not to large-scale deep nets; (b) the D-PDD / PDD equivalence requires $g = g'$ and so does not cover concept drift; (c) the pre-training quantile $\Phi$ relies on a finite number of simulation rounds with no accompanying sample-complexity bound.

4. **Clarify the multi-class extension.** Add a paragraph (or a short appendix section) explaining how the binary disagreement framework is adapted to multi-class prediction tasks in CIFAR-10.1 and GEMINI, and whether the theoretical bounds extend to that setting.

5. **Provide at minimum a sketch of the Bayesian posterior sampling scheme** used to approximate $\arg\max_{h \in \mathcal{H}_p}$ in Algorithm 2 in the main text (even one paragraph), as this is the only feasible implementation for neural networks and cannot be left entirely to the appendix.

6. **Include an ablation on the number of pre-training rounds** (e.g., 50, 100, 200, 500) to give practitioners guidance on when $\Phi$ is reliable enough for deployment.

---

**Axis evaluations:**
- *Novelty:* High — the training-data-free deployment constraint, its formal decoupling, and the simultaneous FPR/TPR guarantees distinguishing deteriorating from non-deteriorating shifts are all novel relative to the disagreement-monitoring literature.
- *Technical soundness:* Moderate-to-high — the theoretical framework is rigorous within its stated scope (manageable VC-dimension, covariate shift only, binary classification), but the theory-to-practice gap is significant and inadequately disclosed for the motivating applications.
- *Empirical support:* Moderate — the GEMINI and CIFAR-10.1 results are meaningful, but the Figure 5(b) labeling issue and missing baselines in Table 2 must be resolved before the empirical case is fully credible.
- *Significance:* High — training-data-free monitoring with provable guarantees addresses a genuine and unmet need in production ML pipelines, especially in regulated domains.
- *Clarity:* Moderate — the two-stage algorithm and theoretical contributions are well-organized, but the multi-class gap, the theory-practice scope, and the figure labeling issue reduce overall clarity in critical places.

# Actual Human Scores
Individual reviewer scores: [6.0, 5.0, 6.0, 3.0]
Average score: 5.0
Binary outcome: Reject
