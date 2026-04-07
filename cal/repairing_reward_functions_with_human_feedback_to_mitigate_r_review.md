=== CALIBRATION EXAMPLE 58 ===

# Harsh Critic Review
Now I have enough material to write a thorough review. Let me produce it.

---

## Section-by-Section Critical Review

### Title & Abstract

The title accurately reflects the paper's content. The abstract makes three substantive claims: (1) PBRR repairs a proxy reward function via an additive correction learned from preferences, (2) a variant achieves √T cumulative regret matching prior work up to constants, and (3) PBRR outperforms baselines requiring substantially fewer preferences. All three are supported in the body. However, the abstract asserts that "corrections on only a few transitions may suffice to recover optimal performance," which is only demonstrated in stylized tabular examples (Appendix I) and not quantified in the real benchmarks. This claim should be caveated or specified.

---

### Introduction & Motivation

The problem is well-motivated and the case for automating the informal reward-repair loop is compelling. The framing against RLHF (costly data) and divergence-regularized methods (requires performant reference policy) is tight. Contributions are clearly enumerated.

**Concern**: The third contribution—"PBRR effectively repairs a proxy reward function even when the proxy induces a substantially suboptimal policy"—is presented alongside the theoretical contribution but only the latter is a *technical* contribution. This conflates a design goal with a contribution. The experiments support the claim, but it oversells it slightly.

---

### Background & Setting (Section 2)

Notation is standard and clearly introduced. The preference over full trajectories (rather than segments) is an important design choice explained in Appendix A. The justification (citing Knox et al., 2022) is reasonable but has a practical consequence: in the Glucose environment (H = 5760), the authors split trajectories into three equal segments for preference elicitation anyway, which contradicts the trajectory-level framing. The inconsistency between the paper's stated principle and its actual implementation in the most complex environment deserves a more prominent disclaimer.

---

### Related Work (Section 3)

Coverage is broad and intellectually honest. The discussion of concurrent work (Cao et al., 2025) is appropriately differentiated: PBRR differs in exploration strategy, loss function, and provides theoretical guarantees. The distinction is meaningful. The claim that Cao et al.'s empirical success arises because their proxy reward induces meaningful progress toward the true objective (Appendix H.2) is plausible and supported by the qualitative analysis of RRM's failure mode.

---

### Methodology (Section 4)

**The additive correction parameterization** (Eq. 2) is natural and well-motivated. The three benefits cited are reasonable.

**The optimism assumption** is the load-bearing pillar of the new loss (Eq. 3). The paper defines a proxy reward as "optimistic" if it upper-bounds the ground-truth reward at every transition (footnote 1). This is a strong, pointwise assumption. Most realistic reward hacking scenarios involve selective over-weighting of certain dimensions—not uniform pointwise dominance—so the assumption is narrower than it might appear. The paper mitigates this by noting (i) the Glucose environment violates it yet PBRR still performs, and (ii) λ₁ and λ₂ are decayed over iterations. However:

- No ablation of the decay schedule (λ_i = 10/|D⁺|) is provided. This is a critical hyperparameter for robustness when the optimism assumption fails.
- The L⁻ regularization term (penalizing upward corrections to preferred transitions in misclassified pairs) is correct *given* optimism, but can actively misdirect learning when optimism doesn't hold. The λ decay is the only safeguard, making its schedule crucial.

**The partition of D_t into D_t⁺ and D_t⁻** (page 4) is based on `sign(r̂_proxy(τ₂) − r̂_proxy(τ₁)) = sign(µ − 0.5)`. The case of ties (µ = 0.5) is not clearly handled: a tie in µ makes `sign(0) = 0`, and `sign(µ − 0.5) = sign(−0.5) < 0`, so a "tie preference" would always be placed in D_t⁻. It is unclear if this is intentional and whether it occurs in the experiments (since synthetic Boltzmann preferences could yield ties rarely, but they can occur).

**Exploration strategy** (Algorithm 1, Lines 5-12): The fallback to maximum-uncertainty-divergence policy pairs (C₁ > 0) is only used theoretically; all experiments set C₁ = 0. This creates a substantial theory-practice gap (see below).

---

### Regret Analysis (Section 5)

**Nature of the theoretical contribution**: Theorems 5.1 and 5.2 show that PBRR inherits the √T regret bound of Pacchiano et al. (2023) up to a factor of C₁ ≥ 1. The key observation (stated explicitly) is: if the reference policy and proxy-optimizing policy are within a constant factor of the maximally-uncertainty-diverging pair, the regret is bounded by C₁ × (Pacchiano's bound). This is an inheritance result, not a fundamentally new analysis. The novel loss function (Eq. 3) plays no role in the theoretical guarantees—the theory only covers the exploration strategy. This weakens the theoretical contribution somewhat.

**Critical gap between theory and practice**: In all empirical evaluations, C₁ = 0, which bypasses the non-dominated policy set construction and uses only π_ref and the current proxy policy for data collection. Appendix K acknowledges that with C₁ = 0, PBRR only asymptotically guarantees performance no worse than π_ref. The paper never proves (or even argues) sublinear regret for the empirically-deployed algorithm. The theoretical analysis is for a variant of PBRR that is never tested, while the tested variant has much weaker formal guarantees. This disconnect should be more prominently flagged in the main text.

**Assumption 5.1 (linearity in trajectory embedding)**: This is incompatible with the empirical settings, which the paper explicitly states involve nonlinear ground-truth rewards in high-dimensional spaces (footnote 4). The theory thus provides no formal insight into the empirical results.

---

### Experiments & Results (Section 6)

**Environments**: The four environments from Pan et al. (2022) are well-chosen to exhibit clearly-defined reward hacking. State and action spaces are substantially larger than standard RL benchmarks, which is a strength. However, all four environments come from the same prior benchmark suite, and the method is not tested on any held-out domain. This limits claims about generalizability, especially since hyperparameters (architecture, learning rates) were tuned per environment using the ground-truth reward function (Appendix E.1).

**Simulated preferences**: All preference labels are generated from the Boltzmann distribution over the ground-truth reward—there are no real humans. This is standard in the RLHF literature, but the paper's motivating text emphasizes real human feedback. Using trajectory-level Boltzmann labels implicitly assumes the preference model is well-calibrated, which may not reflect the messier signals real annotators provide. The paper would benefit from at least a sensitivity analysis to noise in preferences.

**Only 3 random seeds** for most main results: Given the variability visible in Figure 2 (some baselines show large standard errors), 3 seeds is insufficient to draw statistically robust conclusions. The extended analysis in Appendix G.9 (10 seeds for the Pandemic environment, only first two updates) partly addresses this and shows statistically significant results there, but this is only one environment and only early in training.

**Baseline fairness concern**: The State-Constrained-PPO and related baselines use the ground-truth reward function to tune their divergence measure and β coefficient (Appendix E.3), giving them privileged access. Meanwhile, PBRR never uses the ground-truth reward at deployment. This advantages PBRR in comparisons against these baselines (since the baselines are given an oracle hyperparameter for best-case performance but still cannot overcome PBRR). However, PBRR's hyperparameters were also tuned using the ground-truth reward (Appendix E.1). The paper should clarify whether PBRR's λ₁, λ₂ = 10 and the architecture choices were selected via the same oracle process.

**The k² preference structure**: PBRR elicits k² preferences per iteration by crossing k trajectories from π_ref with k from π̂*_t. For the Pandemic environment, k = 79, yielding 6241 preference queries per iteration. While this is compared fairly against baselines using the same k² budget, the absolute cost is non-trivial and the "substantially fewer preferences" claim in the abstract is relative, not absolute.

**Traffic Control is the weakest result**: PBRR matches (but does not clearly outperform) Online-RLHF in Traffic Control, which is the only environment with a multi-dimensional continuous action space. The paper acknowledges this but does not analyze why PBRR's advantage is diminished there.

**Ablations (Section 6.4 and Appendices G.4, G.5)**: The ablations demonstrating the necessity of both the loss and the exploration strategy are well-executed. Figure 3 is informative. The per-term ablation (Appendix G.4) shows environment-dependent sensitivity, which is honest.

**Random reference policy (Appendix G.8)**: The finding that a randomly-initialized reference policy suffices is practically important and somewhat surprising. It essentially decouples PBRR from the quality requirement on π_ref. However, this may partially undercut the narrative that the reference policy provides a "useful contrast"—if random exploration suffices, the role of the reference policy is primarily as a diversity mechanism, not a behavioral prior.

**Reward fragility under extended training (Appendix G.2)**: The finding that in Glucose Monitoring the repaired proxy deteriorates under substantially extended training is a meaningful limitation. It suggests the learned correction term is not a general-purpose reward repair but is co-optimized with the policy training budget. This needs to be discussed in the main paper's limitations section, not just the appendix.

---

### Limitations & Broader Impact

**Not discussed in the main text**. The Conclusion (Section 7) briefly mentions that credit assignment may be easier with segment-level preferences and that additional RLHF techniques remain untested. More substantive limitations are scattered in appendices:
- The reward fragility under extended training (G.2)
- The failure of PBRR with C₁ = 0 to improve beyond π_ref in certain MDPs (MDP 2, Appendix I)
- The dependence on the number of RL steps during training

The absence of a dedicated limitations section is a weakness for an ICLR submission. The paper's stated societal contexts (pandemic policy, clinical treatment) are particularly high-stakes, and the lack of robustness guarantees for the practical (C₁ = 0) algorithm should be clearly disclosed.

---

### Writing & Clarity

The paper is generally well-written. The motivation for each component of the loss (Eq. 3) is explained intuitively and formalized. The illustrative MDP examples (Appendix I) are genuinely helpful for building intuition. Algorithm 1 is clear.

**One substantive clarity issue**: The regret bound in Eq. (4) appears displaced from its theorem statement due to PDF parsing, making it difficult to read the full theorem as stated. The factor C₁ is important but its precise definition (and whether it can be bounded a priori) is not made clear in the main text—it is defined only as a constant factor by which PBRR's trajectory selection deviates from the max-divergence pair, but no bound on C₁ is provided.

---

### Overall Assessment

PBRR addresses a well-motivated problem—automating iterative reward repair—with a principled combination of a novel exploration strategy and a targeted loss function. Empirically, it consistently outperforms baselines on a challenging suite of reward-hacking benchmarks, with particularly strong "jump-start" performance. The ablations are thorough. However, the paper has two major structural weaknesses that reduce confidence in the strength of the contribution. First, there is a fundamental disconnect between theory and practice: the regret bounds apply to a version of PBRR (C₁ > 0) never used in experiments, while the empirically-deployed algorithm (C₁ = 0) has no sublinear regret guarantee. Second, the core optimism assumption underpinning the novel loss function is acknowledged to fail in at least one of the four evaluation environments, yet the robustness mechanism (λ decay schedule) is neither ablated nor theoretically characterized. Additionally, statistical support for the main results relies on only 3 seeds in most cases. These issues do not invalidate the work—the empirical results are convincing and the overall framework is coherent—but they need to be addressed more directly. At ICLR's bar, this paper is borderline: the empirical contribution is solid, but the gap between the theoretical framing and the deployed algorithm, combined with limited statistical rigor, makes it difficult to recommend acceptance in the current form without revision.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Preference-Based Reward Repair (PBRR), a framework that automates the correction of misspecified reward functions using a minimal number of human trajectory preferences. PBRR learns an additive correction term to a human-supplied proxy reward function, utilizing a novel loss function and a targeted exploration strategy that contrasts the proxy-optimized policy with a reference policy. The authors provide theoretical regret bounds matching prior optimal preference-based RL methods in tabular/linear settings and demonstrate superior data efficiency and stability across multiple reward-hacking benchmarks compared to learning rewards from scratch.

### Strengths
1.  **Effective Mitigation of Reward Hacking:** The method successfully addresses the core issue of reward hacking in complex environments where manually designed proxies lead to suboptimal behavior. Experimental evidence in Section 6 shows PBRR achieves significantly higher ground-truth returns with fewer preferences than baselines like Online-RLHF and Residual Reward Modeling (RRM).
2.  **Strong Theoretical Foundation:** Section 5 establishes cumulative regret bounds (Theorems 5.1 and 5.2) that match the order of constants of prior work (Pacchiano et al., 2023) for both known and unknown dynamics settings. This provides a rigorous justification for the preference acquisition strategy used.
3.  **Thoughtful Loss Function Design:** The preference-learning objective in Equation 3 is a key contribution. By regularizing the correction term $g$ to only update where the proxy reward misaligns with preferences (specifically prioritizing negative corrections), the method addresses the "over-optimism" assumption and prevents reward function instability, as shown in the ablation study in Figure 3.
4.  **Rigorous Empirical Analysis:** The paper includes extensive ablation studies (Section 6.4, Appendix G) isolating the contributions of the exploration strategy and the loss function. It also tests robustness to random reference policies (Appendix G.8) and non-optimistic proxies (Appendix G.6), demonstrating the method's applicability beyond ideal assumptions.

### Weaknesses
1.  **Assumption Dependency on Proxy Optimism:** The proposed loss function (Eq. 3) explicitly relies on the assumption that the proxy reward function is "aligned or overly optimistic" (Section 4). While Appendix G.6 shows PBRR works even if this assumption is violated, performance degrades (requiring more preferences), and the theoretical derivation in Section 4 leans on this assumption. ICLR papers with strong claims often require broader robustness guarantees or clearer delineation of failure modes.
2.  **Discrepancy Between Theory and Practice:** The regret analysis (Section 5) assumes a linear feature space for trajectory returns ($r(\tau) = \langle \phi(\tau), w_* \rangle$), whereas the empirical experiments (Section 6) use high-dimensional continuous state spaces where rewards are learned via neural networks (non-linear). While common in RLHF literature, this gap means the theoretical guarantees do not strictly hold for the reported experimental settings.
3.  **Trade-offs in Preference Elicitation:** The authors state in Appendix A that they elicit preferences over full trajectories rather than segments to align with regret-based human preference models. However, they acknowledge this introduces credit assignment challenges and noise compared to segment-level preferences common in existing literature (e.g., Christiano et al., 2017). This limitation is significant for tasks with long horizons like Glucose Monitoring ($H=5760$).
4.  **Comparison with Concurrent Work:** The paper cites Cao et al. (2025) as concurrent work doing similar residual reward modeling. The distinction focuses on the nature of the proxy (highly suboptimal vs. robust) and the learning objective. However, the experimental comparison does not explicitly include the Cao et al. method (only RRM, which is described as a modification of the Cao baseline). Clarifying direct comparisons or implementation parity is necessary to fully validate the claimed superiority.

### Novelty & Significance
**Novelty:** The combination of targeted exploration using a reference policy paired with a specific regularization constraint on the additive correction term constitutes a meaningful methodological contribution. While residual reward modeling exists, integrating it with optimistic exploration bounds and regret analysis for alignment specifically addresses a high-impact gap in RL safety.
**Significance:** ICLR places high value on alignment and sample efficiency. This work offers a practical pathway to reducing human labeling costs in RLHF while mitigating specification gaming (reward hacking), making it highly relevant to the conference's scope.
**Clarity:** The paper is generally well-organized. The distinction between the proxy repair method and standard RLHF is clear. However, Section 4's derivation of Equation 3 is somewhat dense and might benefit from a simplified high-level explanation of the intuition behind $L_+$ and $L_-$ terms in the main text rather than relying solely on the equation.

### Suggestions for Improvement
1.  **Clarify the Reference Policy Requirement:** While Appendix G.8 suggests random initialization works, discuss the theoretical bounds more explicitly in the context of coverage. If the reference policy does not cover the optimal region, can PBRR still recover it? Providing a bound or discussion on the "information gap" between the proxy policy and reference policy would strengthen the theoretical contribution.
2.  **Address the Trajectory Segment Preference Trade-off:** Consider including a small ablation or discussion on how preference elicitation over trajectory segments (as opposed to full trajectories) impacts performance in the Glucose Monitoring environment. This would address the credit assignment limitation mentioned in Appendix A.
3.  **Refine Theoretical Assumptions:** In the Discussion or Conclusion, explicitly state that the regret bounds apply when the reward difference can be embedded linearly, and the experiments validate the heuristic utility in non-linear regimes. Avoiding the implication that the linear theory fully explains the deep RL results prevents overclaiming.
4.  **Direct Baseline Comparison:** If possible, include at least one direct comparison or more detailed discussion against the specific implementation of Cao et al. (2025) used in their robotics benchmarks, rather than just the generic RRM baseline, to concretely establish the advantage in the "highly suboptimal proxy" regime.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Real Human Evaluation:** Replace simulated preferences with a pilot study involving real humans, or simulate realistic human noise models (e.g., inconsistency, bias) instead of Boltzmann on ground-truth, because without this, the "Human Feedback" claim is unsubstantiated and likely overestimates performance.
2. **Reference Policy Sensitivity:** Systematically vary the quality of $\pi_{ref}$ from random to near-optimal across all environments, because the method's practical utility depends on whether a useful reference policy is always available in real-world deployment.
3. **Proxy Misspecification Severity:** Sweep the degree of proxy reward error (from minor tuning to completely wrong), because the claim that "few transitions may suffice" depends on the error being localized rather than systemic.
4. **Wall-Clock Efficiency:** Report total training time including policy re-training at every iteration, because fewer preferences do not guarantee efficiency if the computational overhead per preference is significantly higher than baselines.

### Deeper Analysis Needed (top 3-5 only)
1. **Optimism Assumption Limits:** Quantify performance degradation when the proxy is pessimistic or uncorrelated, because the loss function (Eq. 3) explicitly regularizes based on assumed optimism and Appendix G.6 suggests sensitivity.
2. **Correction Term Magnitude:** Analyze the ratio of $\|g\|$ to $\|\hat{r}_{proxy}\|$, because if the correction term dominates, the method is effectively learning from scratch rather than repairing, undermining the core contribution.
3. **Preference Noise Robustness:** Evaluate performance under high label noise rates, because real human feedback is inconsistent and the current noiseless/Boltzmann simulation masks potential instability in the repair process.
4. **Failure Mode Frequency:** Estimate how often real-world reward hacks resemble the theoretical failure cases in Appendix I (e.g., MDP 2), because theoretical failure modes must be shown to be rare in practice to trust the method.

### Visualizations & Case Studies
1. **Correction Heatmaps:** Visualize the learned correction term $g(s,a)$ overlaid on the Gridworld states, because this directly verifies whether the method targets hacked states vs. globally altering the reward.
2. **Failure Case Trajectories:** Show specific trajectories where PBRR fails to correct the proxy in complex environments (e.g., Traffic or Glucose), because understanding failure modes is critical for trust in safety-critical domains.
3. **Reward Loss Curves:** Plot the reward modeling loss over iterations for PBRR vs. baselines, because this explains the claimed stability improvements quantitatively rather than just showing final returns.
4. **State Coverage Divergence:** Visualize state occupancy of $\pi_{ref}$ vs $\pi_{proxy}$, because the exploration strategy relies on their divergence to identify misaligned transitions.

### Obvious Next Steps
1. **Human-in-the-Loop Pilot:** Conduct a small-scale study with real domain experts (e.g., for Pandemic or Glucose tasks) to validate the preference elicitation burden and interface.
2. **Scaling to High-Dimension:** Test on vision-based or language-based tasks to verify if the repair mechanism scales beyond tabular/low-dimensional control environments.
3. **DPO Integration:** Adapt the correction framework to Direct Preference Optimization (DPO) to remove the inner-loop policy optimization bottleneck and improve stability.

# Final Consolidated Review
## Summary
This paper introduces Preference-Based Reward Repair (PBRR), an iterative framework that repairs a human-specified proxy reward function by learning an additive correction term from trajectory preferences. PBRR combines a targeted exploration strategy (contrasting a proxy-optimized policy with a reference policy) with a novel three-term preference-learning objective that regularizes corrections toward transitions where the proxy incorrectly assigns high reward. The authors provide theoretical regret bounds in tabular/linear settings and demonstrate empirical improvements over baselines on four reward-hacking benchmark environments.

## Strengths
- **Well-motivated problem formulation:** The paper addresses the practical reality that humans iteratively repair reward functions manually, and automates this process with preference queries. The framing against costly RLHF (needs many labels) and divergence-regularized methods (requires performant reference policy) is clear and accurate.
- **Novel loss function design (Eq. 3):** The three-term objective—$L_{pref}$ plus regularization terms $L_+$ and $L_-$—encodes the insight that proxy rewards are typically optimistic and corrections should be focused on incorrectly over-rewarded transitions. The ablation in Figure 3 and Appendix G.4 provides clear evidence that both regularization terms contribute to stability and final performance.
- **Empirical effectiveness across diverse domains:** PBRR consistently outperforms baselines on four environments from Pan et al. (2022) spanning pandemic policy, glucose monitoring, traffic control, and gridworlds. The "jump-start" phenomenon—achieving strong performance after just 1-2 updates—is practically valuable and demonstrated across environments.
- **Random reference policy sufficiency (Appendix G.8):** The finding that randomly initialized policies work nearly as well as trained reference policies is practically significant and somewhat surprising. This substantially reduces deployment requirements compared to methods that assume access to a high-quality behavioral prior.
- **Thorough ablations:** The paper isolates the contribution of each component (loss function vs. exploration strategy) and tests robustness to violated assumptions (non-optimistic proxies in Appendix G.6), providing confidence in the method's reliability beyond ideal conditions.

## Weaknesses
- **Theory-practice disconnect:** The regret analysis in Section 5 applies when $C_1 > 0$ (Algorithm 1, Lines 6-11), which requires computing non-dominated policy sets. However, all experiments use $C_1 = 0$, which bypasses this mechanism. Appendix K correctly notes that with $C_1 = 0$, PBRR only guarantees asymptotic performance no worse than $\pi_{ref}$—not the sublinear cumulative regret bounds advertised. This gap should be explicitly discussed in the main text, not relegated to the appendix.

- **Optimism assumption without full robustness characterization:** The loss function (Eq. 3) fundamentally assumes the proxy reward is optimistic (pointwise upper-bounds ground-truth). The paper notes this can fail (Glucose environment) and relies on $\lambda$ decay as a safeguard, but no ablation of the decay schedule ($\lambda_i = 10/|D^+|$) is provided. If optimism fails significantly, $L_-$ can misdirect corrections by penalizing updates to preferred transitions. The robustness analysis in Appendix G.6 is limited to one non-optimistic configuration of one environment.

- **Limited statistical support for main claims:** Most results report only 3 random seeds with substantial variance in some baselines (visible in Figure 2). Appendix G.9 provides 10-seed results for Pandemic showing statistical significance, but this covers only one environment and only the first two updates. Broader statistical validation would strengthen the claims.

- **Synthetic preferences only:** All preference labels are generated from Boltzmann distributions over ground-truth rewards. While this follows prior RLHF work, the paper's framing emphasizes "human feedback" without testing realistic human noise models (inconsistency, systematic bias). The trajectory-level preference model chosen to match Knox et al. (2022)'s theoretical justification introduces credit assignment challenges acknowledged in Appendix A, but the practical impact is not empirically characterized.

- **Fragility under extended training (Appendix G.2 only):** The Glucose environment shows that the repaired proxy can degrade when PPO training continues significantly beyond the iterations used during reward learning. This is an important limitation—suggesting the correction is co-optimized with training budget—that appears only in the appendix. For safety-critical applications (clinical decision making, pandemic policy), this robustness issue deserves main-text discussion.

## Nice-to-Haves
- **Human-in-the-loop evaluation:** A pilot study with real annotators would substantiate the "Human Feedback" framing and reveal practical challenges in preference elicitation over full trajectories.
- **Sensitivity analysis on reference policy quality:** While random policies suffice in these benchmarks, systematic analysis of when reference policy quality matters would clarify the boundary conditions.
- **Sparsity of learned corrections:** The abstract claims "corrections on only a few transitions may suffice"—quantifying this in empirical results (not just the stylized Appendix I examples) would strengthen the efficiency narrative.

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Claim of hyperparameter tuning unfairness:** The harsh critic suggested PBRR has an advantage because baselines use ground-truth reward to tune divergence measures. However, Appendix E.1 clarifies that PBRR's hyperparameters were also tuned using held-out preference data labeled by ground-truth reward. The comparison is fair; the limitation is that *all* methods assume access to ground-truth for hyperparameter selection, which is acknowledged.

- **Demand for comparison with Cao et al. (2025) robotics tasks:** The spark finder suggests comparing directly against Cao et al.'s robotics benchmarks. However, the paper explicitly distinguishes its setting (highly suboptimal proxies) from Cao et al.'s (proxies that induce meaningful progress), and RRM is implemented as a direct baseline representing Cao et al.'s core method. This is a reasonable experimental design choice for the specific problem setting addressed.

- **Request for wall-clock efficiency:** While computational cost matters, this is not a core contribution claim. The paper focuses on preference efficiency, not compute efficiency.

## Novel Insights
The finding that randomly initialized reference policies perform nearly as well as domain-appropriate ones (behavior cloning from demonstrations) is the most practically significant observation. This suggests PBRR's exploration strategy primarily benefits from *diversity* in trajectory comparisons rather than from the reference policy providing behavioral guidance toward the true objective. This reframes the deployment problem: practitioners need not craft or train a "safe" reference policy before using PBRR—random exploration suffices for repair. This substantially broadens the method's applicability beyond settings where a reasonable behavioral prior is available.

## Suggestions
- **Add a Limitations paragraph to the main text** that explicitly addresses: (1) the $C_1 = 0$ theory-practice gap and what guarantees hold for the implemented algorithm, (2) the dependence on $\lambda$ decay schedule for non-optimistic proxies, (3) the extended-training fragility observed in Glucose, and (4) the synthetic preference setting.
- **Clarify the regret bound interpretation:** State clearly in Section 5 that Theorems 5.1-5.2 provide guarantees for a variant of PBRR with $C_1 > 0$, while the empirical implementation uses $C_1 = 0$ for computational tractability, with only the weaker guarantee from Appendix K applying in practice.
- **Consider additional ablation of $\lambda$ decay:** Even a simple experiment showing robustness to different initial $\lambda$ values or decay schedules would address concerns about the optimism assumption's importance.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 8.0, 2.0]
Average score: 5.0
Binary outcome: Reject
