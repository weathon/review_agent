## Summary

The paper proposes *SubDisMO*, a federated learning algorithm that trains pruned submodels on resource-constrained clients while adding local perturbations (a SAM-style ascent step) to mitigate “arbitrary submodel sharpness.” The authors introduce the *minimum covering number* $C^*$ to characterize convergence in submodel FL, prove a convergence rate, and present a PAC-Bayesian generalization bound. Empirically, they compare against 16 baseline configurations on CIFAR-10 and CIFAR-100.

## Strengths

- **Unified convergence characterization via $C^*$.** Corollary 1 and Remark 1 rigorously show that setting $C^*=N$, $C^*=1$, or $\delta=0$ recovers the known rates of FedAvg, FedSAM, RAM-Fed, and OAP. This organizes prior work under a single quantity and provides practical insight into how parameter coverage affects convergence.
- **Extensive empirical comparison.** Table 1 reports results across multiple heterogeneity levels ($\mu=0.5, 1.0$, IID) and 16 baseline configurations combining IST, OAP, PruneFL, FedRolex, and RAM-Fed with FedAvg/FedProx/SCAFFOLD/FedAdam aggregators, plus full-model FedAvg/FedSAM.
- **Experimental corroboration of $C^*$ dependence.** Figure 4 shows that manually increasing $C^*$ from 1 to 3 yields faster convergence and higher final accuracy, aligning with the theoretical prediction in Corollary 1.

## Weaknesses

### Fatal
None.

### Major

- **Generalization bound does not apply to the actual algorithm.** Theorem 2 (Eq. 17) derives a PAC-Bayesian bound that explicitly requires Gaussian perturbations: “with $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$” (Section 4.2, line 233). However, Algorithm 1 and Eq. (5) implement a *deterministic* signed-gradient perturbation $\tilde{\theta} = \theta + \delta g/\|g\|$. Lemma 4 further requires the perturbation to satisfy a probabilistic stability condition independent of the training data. The deterministic FGSM-style update violates this premise, so the bound in Theorem 2 does not substantiate the paper’s central claim that the perturbation mechanism yields provable generalization.
- **Convergence rate claim obscures non-vanishing bias.** The abstract and introduction advertise an “asymptotically optimal convergence rate of $\mathcal{O}(1/\sqrt{QTC^*})$.” Yet Corollary 1 (line 193) contains persistent additive terms $\mathcal{O}(l^2/C^*)$ and $\mathcal{O}(\sigma_g^2/C^*)$ that do *not* vanish as $Q,T \to \infty$. Because $l$ and $\sigma_g$ are constants independent of $Q$ and $T$, the bound converges to a constant neighborhood rather than to zero. The unqualified “asymptotically optimal” phrasing is misleading.
- **No controlled ablation isolates the perturbation mechanism.** The core claim is that the inner maximization (perturbation) improves generalization. However, the experiments do not include a $\delta=0$ ablation under the *same* mask allocation and aggregation protocol as SubDisMO. RAM-Fed is cited as the $\delta=0$ special case, but it is a different method with its own (unspecified) mask strategy and hyperparameters. Without this ablation, the experiments cannot attribute the observed gains specifically to the perturbation.
- **Mask policy is undefined and experiments contradict “adaptive” claims.** Section 3 introduces a “resource-aware adaptive mask policy $P(\theta_q; R_n)$” (line 65), but $P$ is never formally specified. Algorithm 1 simply writes $m_{q,n} = P(\theta_{q,n})$ without explaining how $R_n$ maps to submodel sizes or whether pruning is structured/unstructured. Section 5.1 reveals that the experiments use a fixed manual split into four non-overlapping submodels, with clients randomly assigned 1/4 or 1/2 of the parameters (line 292). This directly contradicts the abstract’s claim of “adaptive-sized submodels.”

### Minor

- **Low absolute baseline performance and large variance.** Full-model FedAvg reaches only ~56–59% on CIFAR-10 with ViT-Small (Table 1), far below standard performance, suggesting under-training or suboptimal hyperparameters. Standard deviations exceed 8% in several non-IID settings, yet no statistical significance testing is reported.
- **Learning-rate choices in Corollary 1 may violate Theorem 1 constraints.** Theorem 1 requires $\eta_l \leq \sqrt{C^*}/(16TL\sqrt{N})$, while Corollary 1 sets $\eta_l = 1/\sqrt{Q}$. These choices are not guaranteed to satisfy the theorem constraints for the reported experimental settings; the main text only states “when the constant $C>0$ exists” without specifying the required relationship between $Q$, $T$, $N$, and $C^*$.
- **Loss-landscape visualization is anecdotal.** Figure 3 compares SubDisMO to a single baseline (RAM-Fed) on a single dataset and heterogeneity level. It is illustrative but does not quantify sharpness or substitute for rigorous generalization metrics.

### Trivial
None.

## Nice-to-Haves

- A $\delta=0$ ablation and SAM-on-submodel baselines (e.g., OAP combined with SAM) to isolate whether gains come from the perturbation mechanism or from SubDisMO’s specific aggregation.
- Coverage-dynamics plots showing how $C^*$ evolves across layers and communication rounds.
- Statistical significance tests (confidence intervals or paired tests) for the accuracy differences in Table 1.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Framing complaint:** The suggestion that SubDisMO should be framed as “federated SAM with submodels” rather than distributed minimax is a matter of presentation preference. The formulation as a minimax problem over perturbations is standard for SAM and technically valid.
- **Confounded comparisons:** The criticism that baseline combinations using FedProx/SCAFFOLD/FedAdam create confounded comparisons is removed because the asymmetry favors the *baselines*, not the authors’ method. If SubDisMO still outperforms them under its native FedAvg-style aggregation, this strengthens rather than weakens the empirical claim.
- **Formatting/style nitpicks:** Typos, citation formatting, and line-break artifacts from the PDF parser are not author errors.

## Novel Insights

The minimum covering number $C^*$ offers a clean, interpretable quantity for characterizing convergence in submodel federated learning. The paper’s explicit mapping of prior methods (FedAvg, FedSAM, RAM-Fed, OAP) as special cases of a single rate is a useful organizational device that could benefit future work in heterogeneous-resource FL.

## Suggestions

1. **Align theory and algorithm.** Either modify Algorithm 1 to sample the Gaussian perturbations assumed by Theorem 2, or derive a generalization bound (e.g., via deterministic PAC-Bayes or uniform convergence) for the signed-gradient perturbation actually used.
2. **Clarify the persistent bias.** The abstract and introduction should explicitly state that the $\mathcal{O}(1/\sqrt{QTC^*})$ term is the leading vanishing component of a bound that also contains a persistent $\mathcal{O}((l^2+\sigma_g^2)/C^*)$ neighborhood.
3. **Define the mask policy and run the missing ablation.** Formally specify $P(\theta_q; R_n)$ and include a $\delta=0$ ablation under the identical mask allocation, aggregation, and hyperparameter protocol used for SubDisMO.

## Score and Decision

**Calibration anchors used:**
- *High:* `/home/wg25r/review_agent/human_reviews/jj5ZjZsWJe.md` (SCAFFOLD-based compression, avg 8.00) — clean theory, extensive experiments, all claims well supported. SubDisMO is clearly below this: its theory has a fundamental mismatch and its absolute accuracies are much lower.
- *High:* `/home/wg25r/review_agent/human_reviews/EcetCr4trp.md` (FedAvg feature learning, avg 5.75) — accepted poster with sound theory and experiments that verify predictions. SubDisMO is below this because its generalization bound does not apply to its algorithm and its experiments lack a critical ablation.
- *Medium/Low:* `/home/wg25r/review_agent/human_reviews/jhiByZpuIS.md` (MSfusion, avg 4.67, Reject) — model-splitting FL with a gap between claims (large models) and experiments (ResNet-18), plus weak ablations. SubDisMO is comparable: it also has a gap between claimed adaptive masks and actual fixed splits, plus a theory-algorithm mismatch, but slightly more extensive baselines.
- *Low:* `/home/wg25r/review_agent/human_reviews/9Q9KXUTjmd.md` (FedTOGA, avg 4.00, Withdrawn) — SAM in FL with messy methods and presentation issues. SubDisMO is somewhat above this because its algorithm is cleaner and its baseline comparisons are more thorough, but its theory-practice disconnect is at least as severe.
- *Low:* `/home/wg25r/review_agent/human_reviews/fxCSiPPulq.md` (FedMAP, avg 4.00, Withdrawn) — weak experiments (~30% accuracy, non-convergent models). SubDisMO is above this because its models do converge and show consistent relative improvements.

Relative to these anchors, SubDisMO sits between the rejected/withdrawn cluster (4.0–4.7) and the accepted-poster threshold (~5.75). The paper’s real contributions (the $C^*$ analysis and extensive baselines) keep it above the very weakest papers, but the structural theory-algorithm mismatch, the misleading convergence-rate claim, and the missing controlled ablation place it firmly below the acceptance bar.

**Score:** 4.5  
**Decision:** Reject

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>