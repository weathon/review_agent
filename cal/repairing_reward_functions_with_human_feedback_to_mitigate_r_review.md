=== CALIBRATION EXAMPLE 26 ===

# Harsh Critic Review
Now I have a thorough understanding of the paper. Let me write the review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title accurately names the method and problem. The abstract is well-written and makes three claims: (i) PBRR repairs proxy reward functions via a targeted exploration strategy and a new preference-learning objective; (ii) a variant achieves sub-linear cumulative regret matching prior work up to constants; (iii) empirical performance beats baselines on reward-hacking benchmarks. All three claims are substantiated in the paper, though—as discussed below—the alignment between (ii) and (iii) is weaker than the abstract implies.

---

### Introduction & Motivation

The motivation is compelling and clearly positioned at the intersection of proxy-reward misspecification and RLHF cost. The running example of autonomous vehicles blocking on-ramps cleanly illustrates reward hacking. The contributions are clearly enumerated.

One concern: the paper asserts that a proxy reward function can yield "highly suboptimal" behavior yet "corrections on only a few transitions may suffice to recover optimal performance." This claim is partially supported by the illustrative MDP in Appendix I but is stated as universal without proof in the main text. Precisely characterizing when this property holds would strengthen the motivation significantly, especially since the paper's own MDP 2 (Appendix I) provides a counterexample where the algorithm with C₁=0 gets stuck.

---

### Method / Approach

**Additive correction formulation (Eq. 2):** The decomposition r̂_t = r̂_proxy + g_t is clean and the three benefits claimed for it (point estimate rather than Bayesian prior; low-dimensional correction space; enables loss design) are reasonable, though the second benefit is stated without formal justification—the dimensionality of g is assumed smaller than r but no bounds are provided.

**The optimism assumption:** The paper's key motivating assumption is that proxy reward functions are "aligned or overly optimistic" (footnote 1 defines this as r̂(s,a,s') ≥ r(s,a,s') for all transitions). This is a very strong *pointwise* assumption. The paper acknowledges it does not hold in the Glucose Monitoring environment and that PBRR still empirically outperforms baselines there, but the theoretical connection to the loss design in Eq. 3 breaks down for such cases. The decay of λ₁ and λ₂ (Appendix E.6) is described as a remedy, but no analysis is provided for how fast the "wrong" assumption degrades performance before being corrected by the decay.

**Loss function (Eq. 3):** The partition into D⁺ and D⁻ based on whether the proxy reward's ranking matches the human preference label is well-motivated. However, there is an ambiguity regarding ties (µ = ½): the partition condition is `sign(r̂_proxy(τ₂) − r̂_proxy(τ₁)) = sign(µ − 0.5)`, which assigns ties to D⁻ (since sign(0) ≠ sign(0) is ambiguous). This edge case is never addressed. In environments where many trajectory pairs are nearly equally preferred, a substantial fraction of data could be misclassified into D⁻, incorrectly triggering negative corrections.

The regularization λ₁ = λ₂ = 10/|D⁺| decay scheme (Appendix E.6) is ad hoc and the starting value of 10 is reported without any sensitivity analysis. It is described in the appendix rather than the main text, yet it is a key algorithmic hyperparameter that directly affects whether the method respects or ignores the optimism assumption over time.

**Exploration strategy:** The use of the reference policy paired against the current policy for data collection is well-motivated and connected to Xie et al. (2024). The C₁ > 0 fallback to full optimistic exploration is elegant on paper, but as the authors explicitly note in Section 6, this branch is **never activated in any experiment** (C₁ = 0 throughout). This means the algorithm actually deployed empirically is a strict simplification of Algorithm 1, and the more complex exploration mechanism with policy set Π_t is never tested in realistic settings. This is a non-trivial gap.

---

### Regret Analysis

**The key tension — theory vs. practice:** Theorems 5.1 and 5.2 prove √T-regret bounds *only when C₁ > 0* and the reference policy + proxy-reward policy happen to maximize uncertainty up to C₁ of the optimal pair. In every experiment, C₁ = 0. The weaker result for C₁ = 0 is deferred entirely to Appendix K, where it is noted that PBRR only guarantees *asymptotically no worse than π_ref*—a much weaker guarantee. This distinction is not clearly conveyed in the main body of the paper; a casual reader would incorrectly conclude that the √T regret bound applies to the algorithm actually used in experiments.

The paper should either (a) more prominently state that the theoretical results apply to a different configuration than the empirical one, or (b) provide regret guarantees for C₁ = 0 that are stronger than "no worse than π_ref."

**Assumptions:** Assumptions 5.1–5.4 require the ground-truth return to be *linear* in a feature trajectory embedding. The paper immediately notes (Section 6, footnote 4) that the experimental settings have non-linear reward functions in unknown feature spaces. The theory thus applies to a formally separate regime from the experiments, connected only by the shared algorithmic skeleton. This limits the informativeness of the theoretical analysis for the paper's empirical claims.

**Proof sketch:** The proof strategy (inheriting results from Pacchiano et al. (2023) via a coupling argument) is described at the appropriate level of detail for a conference paper, with full proofs in Appendix J.

---

### Experiments & Results

**Environments:** The four benchmark environments from Pan et al. (2022) / Laidlaw et al. (2024) span a reasonable range of task types and misspecification modes. They are all high-dimensional with continuous or large discrete spaces, making the problem hard. The AI Safety Gridworld is a reconfigured toy environment—the choice to make it harder (more tomatoes) is reasonable but means results are not directly comparable to prior uses of this benchmark.

**Baselines:** The baselines are generally well-chosen. Competing against the best variant of the state-constraint approach (using ground-truth reward to select divergence measure and β, as stated in Appendix E.3) actually *disadvantages* the baselines compared to PBRR, which is a principled choice. The authors are transparent about this.

The RRM baseline (implementing Cao et al. (2025)) uses PPO rather than the original SAC optimizer; the authors acknowledge this in Appendix D.2. It is unclear whether this policy optimizer choice materially affects performance in these environments—a brief justification or a small sensitivity experiment would be reassuring.

**Statistical power:** Only **3 random seeds** are used for the main results. Figure 2 shows standard error bands, and for several environments (e.g., Pandemic Mitigation) the shaded regions across methods overlap substantially for extended preference budgets. The additional 10-seed analysis (Appendix G.9) is welcome but is only reported for Pandemic Mitigation and only for the first two proxy reward updates. Given the computational cost of the experiments, this is understandable, but the evidential weight for the "consistently outperforms" claim would benefit from broader statistical support.

**Hyperparameter selection:** PBRR's own hyperparameters were tuned using a held-out test-set of preferences (constructed from the ground-truth reward) as described in Appendix E.1. This procedure uses the ground-truth reward to some degree (constructing initial preference datasets for hyperparameter selection), which arguably provides a form of privileged access analogous to what the authors criticize in the baseline setup. While the procedure is carefully described and appears reasonable, the paper's claim that "our approach never uses privileged information" (Appendix E.3) may be too strong.

**Scalability experiment in Figure 2:** The results are displayed on a scaled/clipped normalized axis (Figure 2) with unscaled results relegated to Appendix G.7. The normalization is helpful for comparison but clips very poor performance to −1, which can hide the degree to which some baselines (e.g., Online-RLHF in Pandemic early on) fail catastrophically. The choice to clip and normalize should be stated in the caption of Figure 2 (it is mentioned in the figure caption, but only after several sentences of result interpretation).

**Reward function robustness (Appendix G.2):** The finding that the Glucose environment's repaired reward function degrades when optimized for substantially more RL steps is an important limitation. This implies PBRR's reward repair may be tied to a specific RL budget, reducing portability of the learned reward function—a concern not discussed in the main paper's limitations section.

---

### Ablation Study

The ablation in Figure 3 effectively separates the contribution of the learning objective from the exploration strategy. The result that using PBRR's Eq. 3 objective with RRM's exploration (or vice versa) does not match full PBRR in most environments is meaningful evidence for the joint necessity of both components. 

However, the ablation of L⁺ versus L⁻ individually (Appendix G.4) reveals environment-dependent patterns (L⁺ matters more in Pandemic, L⁻ in Glucose). The authors note this but offer no explanation for why the components have different relative importance across environments. A deeper analysis here would strengthen understanding of when each component is most critical.

---

### Writing & Clarity

Section 5 is difficult to parse without the appendix. Theorem 5.1 is stated with the bound (Eq. 4) appearing *after* the theorem statement due to a misplacement—readers encounter the assumption conditions before seeing the theorem conclusion. Algorithm 1 references "the non-dominated policy set" before it is formally defined in Section 5. The notation in footnote 2 uses the same subscript/superscript conventions as the main text but adds new usage without explicit callout.

The practical interpretation of the algorithm's fallback branch (lines 6–11 of Algorithm 1) would benefit from a concrete example in the main text, given that this branch is theoretically important but always inactive in experiments.

---

### Limitations & Broader Impact

The authors acknowledge key limitations: trajectory-level rather than segment-level preferences introduce credit assignment noise; the optimism assumption may not hold; C₁=0 only guarantees asymptotic parity with π_ref.

Missing from the limitations discussion:
1. **Computational cost**: PBRR re-trains the correction network g from scratch on the growing dataset at each iteration. The cost of this grows with the preference budget and is never analyzed.
2. **Real human feedback**: All experiments use synthetic Boltzmann preferences. The paper does not test robustness to preference noise models beyond the Bradley-Terry specification, and never validates on real human annotators.
3. **The fragility finding in Appendix G.2**: The observation that the repaired reward function can "overfit to a particular RL budget" is significant for deployment and should be stated as a limitation in the main paper.
4. **When the reference policy is highly suboptimal**: The paper shows randomly-initialized reference policies work (Appendix G.8), but the analysis of when the reference policy provides useful coverage vs. being a harmful comparison is limited to qualitative discussion.

---

### Overall Assessment

PBRR is a well-motivated and practically relevant contribution to the reward misalignment literature. The core idea—learning an additive correction to a proxy reward function using targeted comparisons against a reference policy, guided by a custom regularized loss—is technically sound and yields empirically compelling results across four diverse environments. The ablations are genuinely informative, and the comparison to Cao et al. (2025) is handled fairly with transparency about implementation differences.

The paper's most significant weakness is the gap between the theoretical guarantee and the practical algorithm: the √T-regret bounds of Theorems 5.1 and 5.2 apply to the configuration with C₁ > 0, while all experiments use C₁ = 0 for which only a weak asymptotic guarantee exists (Appendix K). This is not a fatal flaw but should be prominently acknowledged in the main text rather than buried in the appendix. Additionally, the limited number of random seeds (3 for most results) and the absence of any real human preference validation leave the statistical and practical robustness claims somewhat underdetermined. Taken together, these issues are addressable without requiring new experiments and do not invalidate the contribution—PBRR represents a meaningful step forward. The paper is above the ICLR acceptance bar but would benefit from revisions clarifying the theory/practice disconnect and sharpening the limitations section.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Preference-Based Reward Repair (PBRR), an iterative framework that aligns a human-specified proxy reward function by learning an additive, transition-dependent correction term from trajectory preferences. PBRR combines a targeted exploration strategy (comparing the current policy against a reference policy, with an uncertainty-maximizing fallback) with a novel preference-learning objective that regularizes corrections to avoid destabilizing optimistic proxy signals. The authors demonstrate superior data efficiency and stability over standard RLHF and residual reward modeling baselines across multiple reward-hacking benchmarks, and provide cumulative regret bounds matching prior strategic RLHF methods in linear/tabular settings.

### Strengths
1. **Clear, Practical Problem Formulation:** PBRR effectively bridges the gap between brittle manual reward design and prohibitively expensive RLHF. By parameterizing the reward as $\hat{r}_{proxy} + g$, the method reduces the hypothesis space for the learned component, which empirically yields strong sample efficiency gains across four diverse environments (Sec. 6.3, Fig. 2).
2. **Well-Motivated Objective with Validated Components:** The preference loss in Eq. 3 introduces $L_+$ and $L_-$ regularizers grounded in the intuition that proxy rewards are typically optimistic or aligned. The ablation studies (Fig. 3, App. G.4) quantitatively demonstrate that both terms are necessary for stability and performance, particularly in preventing the reward model from over-valuing suboptimal reference trajectories.
3. **Rigorous Empirical Evaluation & Strong Baselines:** The paper benchmarks against state-of-the-art and natural ablation methods (Online-RLHF, RRM, state-constrained PPO) on established reward-hacking environments with high-dimensional state spaces. PBRR consistently outperforms baselines in final ground-truth return, requires fewer preference queries for competitive performance, and exhibits notably less oscillatory training behavior (Sec. H.1 analysis is particularly insightful).
4. **Solid Theoretical Grounding & Reproducibility:** The paper provides cumulative regret bounds (Thm. 5.1, 5.2) that match Pacchiano et al. (2023) up to constants, and formalizes "unhackability" guarantees in the infinite-data limit (Thm. K.1). Implementation details, hyperparameter search procedures, and environment configurations are thoroughly documented in the appendices, and code is provided, meeting high reproducibility standards.

### Weaknesses
1. **Theory-Practice Gap:** The regret bounds assume linear reward embeddings and known/tabular dynamics, whereas all experiments use high-dimensional, non-linear environments with neural network approximators and estimated dynamics. While common in RLHF literature, the manuscript does not discuss how the theoretical mechanisms (e.g., undominated policy sets, covariance norms) translate or degrade under function approximation, leaving the empirical success theoretically unexplained in the deep RL regime.
2. **Reliance on the Optimism Assumption:** The loss design hinges on $\hat{r}_{proxy}$ being optimistic relative to the ground truth. The authors acknowledge this is violated in the Glucose environment and show empirical robustness, but the failure modes and convergence behavior under pessimistic proxies are only briefly explored in a single gridworld experiment (App. G.6). The heuristic decay of $\lambda_1, \lambda_2$ (App. E.6) lacks theoretical justification for this mismatch.
3. **Discarded Exploration Fallback in Practice:** Algorithm 1 includes a principled uncertainty-driven exploration fallback triggered when the divergence between $\pi_{\hat{r}_t}$ and $\pi_{ref}$ falls below a threshold. However, for empirical runs, the authors set $C_1=0$ (Sec. 6), effectively disabling this mechanism. The paper does not analyze how often the fallback would trigger in practice or quantify the potential sample efficiency loss from relying solely on reference-vs-repaired exploration.
4. **Preference Simulation Mismatch:** Preferences are labeled using differences in summed rewards rather than regret, despite citing literature (Knox et al., 2022) showing regret better aligns with human preferences. While computationally pragmatic, this choice introduces a systematic bias between the simulated feedback and real human judgments. The paper does not evaluate how sensitive PBRR's repair process is to this labeling discrepancy.

### Novelty & Significance
**Novelty:** The work makes a clear methodological contribution by combining a targeted exploration strategy with a regularized preference loss explicitly designed to repair, rather than learn from scratch, proxy reward functions. While residual reward modeling and reference-policy constraints are prior concepts, the integration of optimism-aware regularization ($L_+, L_-$) and the formal analysis of unhackability in this repair context are novel.
**Clarity:** The manuscript is well-structured, with clear mathematical definitions, intuitive explanations of the loss terms, and thoughtful qualitative analyses of failure modes. Minor typographical/OCR artifacts in equations do not detract from readability.
**Reproducibility:** High. Detailed hyperparameter grids, architecture specs, PPO settings, and environment configurations are provided. Code availability is confirmed. Expanding random seeds beyond 3 (only done for one env in App G.9) and reporting wall-clock compute times would further strengthen this.
**Significance:** Strongly aligned with ICLR's focus on efficient, scalable alignment and RLHF. The method addresses a practical bottleneck in real-world RL deployment where expert-designed proxies are flawed but fully learning a reward from scratch is infeasible. The empirical gains in stability and sample efficiency are meaningful for downstream safety-critical applications.

### Suggestions for Improvement
1. **Bridge Theory and Empirics:** Add a discussion section analyzing how the covariance-based exploration and linear regret bounds map to the neural network setting. Consider using concepts from Neural Tangent Kernel (NTK) literature or local linear approximations to hypothesize why the tabular insights hold under function approximation, or explicitly outline limitations.
2. **Systematic Pessimistic & $\lambda$ Sensitivity Analysis:** Expand the pessimistic proxy analysis across at least one high-dimensional environment. Additionally, conduct a controlled ablation on the $\lambda_1, \lambda_2$ decay schedule (e.g., fixed vs. step decay vs. adaptive) to demonstrate robustness and move beyond the heuristic normalization by $|D_t^+|$.
3. **Quantify the Exploration Fallback Trade-off:** Run a set of experiments with $C_1 > 0$ to measure how frequently the algorithm triggers the uncertainty-maximizing exploration in practice. Report the preference budget vs. performance curve with and without the fallback to justify the $C_1=0$ choice empirically, or refine the algorithm to dynamically tune $C_1$.
4. **Preference Labeling Robustness:** Include a sensitivity analysis or ablation comparing sum-of-reward preference labels vs. approximated regret labels (or varying noise models) on at least one environment. This would strengthen claims about real-world applicability where human preferences may not strictly follow reward sums.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Real Human Feedback Evaluation:** Replace simulated ground-truth preferences with real human labels on at least one environment. Without this, the claim of mitigating reward hacking via *human* feedback is unverified against actual human noise, inconsistency, and cognitive burden.
2. **Reference Policy Quality Sweep:** Systematically vary $\pi_{ref}$ quality from random to near-optimal across all high-dimensional environments. The claim that a random policy suffices (Appendix G.8) is under-supported for continuous control spaces where random policies visit negligible valid state space.
3. **Non-Linear Convergence Analysis:** Empirically verify if the reward model converges to the ground truth in the deep RL setting, given the theory only holds for linear rewards. Without this, the theoretical contribution does not support the empirical claims using neural networks.
4. **Preference Noise Robustness:** Evaluate performance under varying levels of simulated human noise (e.g., flipping preference labels probabilistically). The method's stability claims rely on consistent preference signals which may not hold in practice, risking overfitting to noisy labels.
5. **Baseline Hyperparameter Fairness:** Ensure RRM and Online-RLHF baselines are not disadvantaged by hyperparameter choices tuned specifically for PBRR. If baselines are under-tuned relative to the proposed method, the superiority claim regarding data efficiency is unreliable.

### Deeper Analysis Needed (top 3-5 only)
1. **Optimism Assumption Validity:** Quantify how often real-world proxy rewards are actually "optimistic" vs. "pessimistic" in standard benchmarks. The loss function (Eq. 3) is designed for optimism; if this assumption rarely holds, the method's design is misaligned with practical reward design failures.
2. **Correction Term Magnitude:** Analyze the ratio $\|g\| / \|\hat{r}_{proxy}\|$ over iterations to determine if the method repairs or rewrites the reward. If $g$ dominates, the method is effectively learning from scratch, undermining the "repair" and data-efficiency narrative.
3. **Exploration Strategy Discrepancy:** Explain why the uncertainty-based exploration ($C_1 > 0$) was disabled ($C_1=0$) in experiments despite being central to the regret bounds. This disconnects the theoretical guarantees from the practical implementation, leaving the empirical success unexplained by the theory.
4. **Credit Assignment Gap:** Explain how transition-level corrections ($g(s,a,s')$) are learned from trajectory-level preferences without segmenting trajectories. This mismatch may obscure which transitions are actually being repaired, potentially leading to sparse or incorrect credit assignment.
5. **Failure Mode Frequency:** Quantify how often the "MDP 2" failure case (Appendix I) occurs in the benchmark environments. Showing success rates without characterizing failure conditions limits trust in the method's reliability across diverse tasks.

### Visualizations & Case Studies
1. **Reward Function Heatmaps:** Visualize $\hat{r}_{proxy}$, $\hat{r}_{final}$, and $r_{groundtruth}$ on a 2D slice of the state space (e.g., Gridworld). This reveals whether the method repairs specific hacked regions or reshapes the entire reward landscape.
2. **Correction Sparsity Map:** Plot the magnitude of $g(s,a,s')$ across states to show where corrections are applied. This validates the core hypothesis that only a few transitions require correction to fix hacking behavior.
3. **Trajectory Rollout Comparison:** Side-by-side visualization of trajectories from $\pi_{proxy}$, $\pi_{PBRR}$, and $\pi_{optimal}$. This exposes whether the policy behavior actually changes to avoid hacking or if the reward values simply shift without behavioral correction.

### Obvious Next Steps
1. **Conduct a Real Human-in-the-Loop Study:** Validate the framework with actual human stakeholders to confirm data efficiency gains hold under realistic cognitive load and noise. This is essential to move beyond simulated preference labels and prove practical utility.
2. **Develop Non-Linear Regret Bounds:** Extend the theoretical analysis to cover neural network reward models to align theory with the empirical deep RL setup. Without this, the theoretical contribution remains disconnected from the primary experimental results.
3. **Automate Reference Policy Generation:** Propose a method to generate a minimally viable $\pi_{ref}$ automatically, removing the dependency on user-provided demonstrations or random initialization. This would reduce the human burden further and strengthen the automation claim.

# Final Consolidated Review
## Summary

This paper introduces Preference-Based Reward Repair (PBRR), an iterative framework that repairs a human-specified proxy reward function by learning an additive, transition-dependent correction term from trajectory preferences. The method combines a targeted exploration strategy—comparing trajectories from the policy induced by the current proxy against a supplied reference policy—with a novel preference-learning objective that regularizes corrections toward transitions where the proxy incorrectly assigns high reward. The authors prove cumulative regret bounds matching prior strategic RLHF methods in linear settings and demonstrate superior data efficiency and stability across four reward-hacking benchmark environments.

## Strengths

- **Principled problem formulation with empirical validation.** The decomposition r̂_t = r̂_proxy + g allows the method to leverage structure in the initial proxy while efficiently learning only the correction. The empirical results across four diverse environments (autonomous vehicle traffic control, pandemic lockdown design, glucose monitoring, and an AI safety gridworld) demonstrate consistent improvements over baselines that learn reward functions from scratch or use alternative repair strategies. The ablation studies (Figure 3, Appendix G.4) cleanly separate the contributions of the learning objective (Eq. 3) and exploration strategy, showing both components are necessary.

- **Theoretical grounding with meaningful guarantees.** Theorem K.1 provides an important guarantee: under noiseless preferences and in the infinite-data limit, the repaired reward function induces a policy that matches or exceeds the reference policy's ground-truth performance. This formalizes a concrete sense in which reward hacking is mitigated. The regret bounds (Theorems 5.1, 5.2) match prior strategic RLHF work up to constants when the exploration fallback is enabled.

- **Insightful analysis of failure modes.** The qualitative analysis in Appendix H explains why Online-RLHF oscillates in the AI Safety Gridworld (learning to over-value visiting tomato states rather than watering them) and why RRM fails to improve (exploiting the proxy reward function does not induce effective exploration). These mechanistic explanations strengthen confidence that PBRR's components address real algorithmic challenges.

- **Transparent experimental methodology.** The baseline comparisons use state-constrained methods with divergence measures selected using ground-truth access (Appendix E.3)—a *stronger* baseline configuration that disadvantages PBRR. The unscaled results (Appendix G.7) and re-training robustness analysis (Appendix G.2) are provided without hiding limitations.

## Weaknesses

- **Gap between theoretical and empirical configurations.** The √T regret bounds in Theorems 5.1 and 5.2 require C₁ > 0, meaning the algorithm must fall back to uncertainty-maximizing exploration when the reference-policy divergence is insufficient. However, all experiments use C₁ = 0 (Section 6), for which only the weaker Appendix K guarantee applies—ensuring asymptotic parity with π_ref but not sub-linear regret. This disconnect should be prominently acknowledged in the main text. The theoretical analysis also assumes linear reward embeddings and known/tabular dynamics, while experiments use neural network function approximation in high-dimensional continuous environments, leaving the empirical success theoretically unexplained in the deep RL regime.

- **Optimism assumption may not broadly hold.** The loss design (Eq. 3) assumes the proxy reward is "aligned or overly optimistic" (footnote 1: r̂(s,a,s') ≥ r(s,a,s')). The Glucose Monitoring environment violates this assumption, and while PBRR still works empirically, the theoretical justification for the L_+ and L_- regularization terms breaks down. The decay scheme for λ₁, λ₂ (Appendix E.6) is heuristic, and the paper offers no analysis of how performance degrades before the decay corrects for assumption violations.

- **Limited statistical validation.** The main results report only 3 random seeds per method per environment. While Appendix G.9 provides a 10-seed analysis for Pandemic Mitigation (showing statistically significant differences after two updates), the broader claim of "consistent outperformance" would be strengthened by more seeds across all environments. Standard error bands in Figure 2 overlap meaningfully in several cases.

- **No validation with real human feedback.** All preferences are simulated using the Boltzmann distribution under the ground-truth reward function. While this is standard for controlled experiments, it leaves open whether PBRR's advantages persist under human noise models, inconsistent judgments, or cognitive biases that deviate from the Bradley-Terry assumption.

- **Credit assignment from trajectory-level preferences.** The method elicits preferences over full trajectories but learns transition-level corrections g(s,a,s'). Appendix A notes this may introduce credit assignment challenges, though the paper argues trajectory-level preferences better match human judgment models. The potential mismatch between feedback granularity and correction granularity remains an open limitation.

## Nice-to-Haves

- **Human preference study.** Validation with real human annotators on at least one environment would strengthen claims about practical applicability and robustness to preference noise beyond the simulated Bradley-Terry model.

- **Analysis of correction term magnitude.** A quantitative analysis of how the ratio ||g||/||r̂_proxy|| evolves over iterations would clarify whether PBRR truly "repairs" the proxy (small corrections) or effectively learns a new reward function from scratch (large corrections), with implications for interpretability and debugging.

- **Extended analysis of C₁ > 0 configurations.** Since the theoretical guarantees require C₁ > 0 but experiments use C₁ = 0, experiments measuring how often the fallback would trigger and its impact on sample efficiency would bridge theory and practice.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Weakness:* "Real Human Feedback Evaluation" — This is not standard for RLHF papers at ICLR; simulated preferences with ground-truth labels are the accepted methodology for controlled algorithmic comparisons.

- *Weakness:* "Baseline hyperparameter fairness concerns" — The paper clearly describes using the same hyperparameter selection procedure (Appendix E.1) for all methods. The ground-truth is used only for offline hyperparameter tuning before experiments, not during method execution.

- *Weakness:* "PPO vs SAC for RRM baseline" — The authors transparently acknowledge this implementation difference (Appendix D.2). Using a consistent PPO baseline across methods is methodologically sound for fair comparison.

- *Weakness:* "Formatting and minor notation issues" — These are identified as parser artifacts in the review instructions and do not constitute substantive criticisms.

- *Weakness:* "Computational cost of re-training" — This is standard practice in online RLHF methods and not a unique limitation of PBRR.

## Novel Insights

The paper's core insight—that repairing an existing proxy reward function is more data-efficient than learning from scratch because the correction term often lies in a lower-dimensional space than the full reward—has significant implications for practical RL deployment. The loss function design explicitly encodes the intuition that well-designed proxy rewards are typically *optimistic* or *aligned*, and that corrections should focus on transitions where the proxy incorrectly assigns high reward rather than arbitrarily adjusting all transitions. The theoretical result that PBRR guarantees recovery of a policy at least as good as the reference policy (Theorem K.1) formalizes a meaningful notion of "safety" in reward repair: even with limited data, the method cannot do worse than the provided baseline. The MDP 2 counterexample (Appendix I) honestly reveals when and why the C₁=0 configuration can fail to reach optimality, grounding expectations for the method's limitations.

## Suggestions

- **Clarify the theory-practice gap in the main text.** Add a sentence in Section 5 or 6 explicitly stating: "All experiments use C₁=0, for which we provide a weaker asymptotic guarantee in Appendix K. The regret bounds in Theorems 5.1-5.2 apply to the C₁>0 configuration, which remains an interesting direction for future work in complex environments."

- **Expand limitations discussion.** Explicitly mention in Section 7 that: (i) the optimism assumption is violated in some domains; (ii) the reward function may overfit to a particular RL optimization horizon (Appendix G.2); (iii) statistical confidence would benefit from more seeds.

- **Report absolute performance scale.** While normalized plots aid comparison, include a table with absolute ground-truth returns alongside scaled values so readers can assess practical significance of gaps between methods.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 8.0, 2.0]
Average score: 5.0
Binary outcome: Reject
