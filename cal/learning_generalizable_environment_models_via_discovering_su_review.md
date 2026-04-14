=== CALIBRATION EXAMPLE 13 ===

# Final Consolidated Review
## Summary

SAM (Superposed cAusal Model) addresses a genuine gap in causal world model learning for RL: prior work assumes a single global causal graph, but real environments may exhibit *superposed* (mixed) causal mechanisms that differ across episodes. SAM proposes an end-to-end differentiable framework using a Transformer to infer a per-trajectory causal mask on the fly, which is then used to constrain a factored dynamics model. The approach is validated on two custom synthetic environments (Mixed-Chemical and Confusing-Minigrid), where it substantially outperforms clustering-based and single-graph causal baselines in causal graph recovery and robustness to spurious correlations.

---

## Strengths

- **Clearly identified and well-scoped problem:** The paper identifies a concrete, underexplored failure mode of existing causal world models — their degradation when the dataset mixes trajectories from multiple causal regimes. The episodic framing (one causal graph per episode) is a natural and tractable instantiation of this problem.

- **Strong causal discovery performance over competitive baselines:** SAM achieves an average SHD of 4.05 vs. 33.33 (KMeans+CDL) and 27.06 (FCDL) in Mixed-Chemical (Table 1), and perfectly recovers one graph configuration (SHD = 0.00 in M-C012). The margin is not marginal — it is large enough to suggest a genuine qualitative advantage in disentangling trajectory-specific causal structure.

- **Demonstrated on-the-fly adaptation:** Figure 6 and Figure 7 jointly show that SAM's inferred causal graph rapidly converges toward ground truth within ~10–15 steps of a new episode, and the qualitative heatmap in Figure 7 is a convincing visualization of this convergence. This is the key practical property of the method and is demonstrated with appropriate detail.

- **End-to-end differentiable design:** By using Gumbel-Softmax for discrete causal mask sampling, SAM avoids the non-differentiable search-based causal discovery used by CDL and GRADER, enabling scalable joint optimization of the mask predictor and dynamics model.

- **Spurious-correlation robustness is meaningfully tested:** The spurious setting — where specific nodes receive random noise at test time — is a genuine test of whether the model has internalized the correct causal support rather than memorizing correlations. SAM's advantage grows with noise level (Figure 3), which is the correct behavior for a causally grounded model.

---

## Weaknesses

### Fatal
*None identified. The core contribution is genuine and the empirical advantage over baselines is real. However, the cumulative weight of the Major weaknesses below places the paper below ICLR's acceptance bar in its current form.*

---

### Major

- **No acyclicity constraint despite DAG formalization.** The problem is formally defined using Directed Acyclic Graphs (Section 3.2), but the optimization objective (Eq. 2) only enforces sparsity via an L1 norm. There is no mechanism preventing the learned causal masks from being cyclic. This is not a mere theoretical nit: cyclic "causal" graphs violate the SCM factorization in Eq. (1), potentially undermining the causal interpretation of the model entirely. The paper should either enforce acyclicity (e.g., via a differentiable DAG constraint such as NOTEARS), report what fraction of learned graphs are acyclic, or explicitly justify why acyclicity is empirically satisfied in these environments.

- **Missing critical ablation: Transformer without causal masking.** The ablation in Section 5.5 compares SAM vs. an RNN baseline, which tests whether *any* trajectory-conditioned model outperforms a simple recurrent baseline. However, the key question for SAM's causal masking module is: does the *causal* inductive bias (sparse factored masking) provide benefit beyond what a standard Transformer dynamics model would achieve? A Transformer that infers a dense per-episode context vector without explicit causal structure is the appropriate control. Without it, the performance gain could be attributed entirely to Transformer capacity rather than causal discovery.

- **No statistical uncertainty reported.** Tables 1 and 2 and Figures 3–5 report single-point estimates with no standard deviations or confidence intervals across random seeds. For custom synthetic benchmarks at ICLR, it is not possible to assess whether the reported differences are statistically reliable. This affects every quantitative claim in the paper.

- **Ambiguous online inference mechanism.** Section 4.2 defines $q_\phi(\mathcal{G}|\tau)$ as conditioning on the full trajectory $\tau$. Section 5.6 shows SHD decreasing over individual episode time steps (Figure 6), implying the mask is updated online as new steps are observed. The paper never explicitly resolves this: is $\tau$ a growing prefix of the current episode? Is the mask re-inferred at every step, or at intervals? This is a core reproducibility issue for the method's test-time behavior and must be formally described.

- **No comparison against meta-RL / contextual dynamics baselines.** The problem — adapting dynamics prediction to a per-episode latent context — is the central task of context-based meta-RL. A latent-variable dynamics model (e.g., conditioning on a per-episode context vector inferred from trajectory prefixes) is a natural and well-studied baseline. The paper does not compare against any such approach, leaving open whether the explicit causal inductive bias provides any benefit over a learned latent context without structural constraints. This significantly weakens the novelty argument.

- **Experiments exclusively on custom synthetic environments.** Both benchmarks were constructed by the authors. Neither is publicly available or widely used. All claims of "generalizable world models" rest on these two environments with small state spaces (10 nodes, 5×5 grid). The absence of evaluation on any standard offline RL benchmark (e.g., D4RL with modified dynamics, or any publicly available causal discovery benchmark) makes it impossible to assess whether SAM's advantages extend beyond the specific experimental setup designed to showcase them.

---

### Minor

- **Typo in factored transition definition (Section 4.1).** The stated expression $p(s_{t+1}|s_t, a_t) = \prod_{i=1}^d p(s_{t+1}^i|s_{t+1}, a_t)$ conditions the right-hand side on $s_{t+1}$ rather than $s_t$, making the definition circular. The intended expression almost certainly conditions on $s_t$.

- **Confusing summary notation in Section 4.2.** The sentence "In summary, we have $q_{\phi}(\mathcal{G} \mid \tau) := f_{\theta_2}(s_{t+1} \mid f_{\theta_1}(s_t, a_t) \circ \mathcal{G}, \text{sg}(\mathcal{G}))$" equates the causal mask predictor with the dynamics model, which are two distinct modules. This appears to be an attempt to summarize the full prediction pipeline but uses incorrect notation. The design of the two modules should be stated with separate, clearly distinguished equations.

- **Imprecise ELBO claim.** The introduction states the objective is "based on the evidence lower bound of the trajectory generation likelihood." A proper ELBO requires a KL divergence between the variational posterior and the prior: $\mathbb{E}_q[\log p(\tau|\mathcal{G})] - \text{KL}(q_\phi(\mathcal{G}|\tau)\|p(\mathcal{G}))$. Instead, Eq. (2) uses an L1 penalty. While an L1 norm can be motivated as a KL to a sparse Laplace prior, the paper does not make this derivation. The objective should either be derived from a proper Bayesian prior to justify the ELBO claim, or described straightforwardly as a sparsity-regularized reconstruction objective without invoking ELBO.

- **Stop-gradient design choice unexplained.** The term $\text{sg}(\mathcal{G})$ appends the causal mask as a gradient-stopped input to the dynamics model. The rationale for this choice — whether it acts as a context signal, prevents gradient interference, or serves another purpose — is never explained. This is a non-trivial design decision that affects optimization and should be motivated.

- **Duplicated paragraph in Section 2.2.** The paragraph beginning "However, these methods often require manual design of scoring functions and can be computationally intensive..." appears twice in the related work section. This is a concrete editing error.

- **Exclusion of Minigrid from prediction-error analysis.** Section 5.3 omits prediction-error results for Confusing-Minigrid because "the scale of each dimension in state space are different." However, per-dimension accuracy or normalized metrics could address this without loss of rigor. Excluding one of only two environments from a key evaluation is a meaningful gap.

- **MPC details absent.** Section 5.4 evaluates policies via Model Predictive Control, but the specific planning horizon, rollout count, and uncertainty handling strategy are not described. These choices substantially affect MPC performance.

---

### Tiny

- The "Super-post" label in Figure 1 is a typo for "Superposed."
- The limitations section discusses only computational cost. It does not mention the full-observability assumption, the within-episode stationarity assumption (causal graph fixed within but varying across episodes), or the discrete/factored state space requirement — all strong assumptions that constrain applicability.

---

## Nice-to-Haves

- **Sensitivity analysis for the sparsity coefficient λ.** Causal discovery is well-known to be sensitive to regularization strength. Even a brief ablation over a few values of λ would provide important practical guidance.
- **Visualization of learned graph embeddings (e.g., t-SNE) colored by ground-truth causal regime.** This would confirm that SAM genuinely clusters trajectories by their latent causal mechanism rather than overfitting a unique graph to each trajectory individually.
- **Failure mode case studies.** Showing specific trajectories where SAM infers the wrong causal graph and analyzing how the error propagates into prediction would strengthen the paper's scientific depth.
- **Evaluation on continuous control.** The current factorized state assumptions are well-suited to discrete/grid environments; extending to MuJoCo-style tasks with continuous dynamics would broaden the claim of generalizability.
- **Quantification of warm-up cost.** Figure 6 shows SHD is high during the first ~10 steps of an episode. An analysis of how this warm-up period affects downstream policy reward (e.g., comparing cumulative reward conditional on step number) would characterize a real operational cost of the method.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Superposed" is unconventional terminology (Harsh Critic).** While "superposed" is not standard, it is clearly defined in the paper and is consistently used throughout. This is a style preference, not a scientific flaw. Removed.

- **Missing related works (all reviewers, various).** Per review instructions, specific missing citation claims are not included as the reviewers may lack current knowledge of the literature. Removed.

- **The "contextual MDP" framing as a missing citation (Harsh Critic).** Keeping the *baseline comparison* aspect as a Major weakness (meta-RL/contextual dynamics baselines), but removing the specific claim about missing citations to HiP-MDP, Doshi-Velez, etc.

- **Claim that "generalizable across different tasks and environments" is only supported by two environments (framed as the abstract being deceptive) (Harsh Critic).** The scope mismatch is real, but the abstract is not deceptive — it is standard to write abstracts broadly. The limitation of two synthetic environments is captured adequately under Major weaknesses. The framing of this as a rhetorical problem is removed.

- **Absence of theoretical proofs (Spark Finder, implied).** Demanding theoretical generalization bounds for what is primarily an empirical systems paper is not standard for this setting. Removed (was moved to Nice-to-Haves implicitly, but not included even there since it is non-standard).

- **Unfair comparison with baselines due to asymmetric setup (not raised, but evaluated).** The baselines include Oracle (upper bound) and weaker methods; no unfair asymmetry benefiting SAM is identified.

---

## Novel Insights

The most substantive novel observation across all three reviews, not explicitly stated by the paper itself, is the following tension: SAM conditions its causal mask predictor on the *full* trajectory $q_\phi(\mathcal{G}|\tau)$, yet the central practical claim is on-the-fly adaptation as shown in Figure 6. This implies that during deployment, the mask must be re-inferred from an expanding trajectory prefix — but the paper never formalizes this inference protocol, leaving open whether the model is truly performing *online* causal identification or whether it relies on batch access to the full episode. This distinction matters significantly: if SAM needs the entire episode before it can reliably infer the causal graph, then its "on-the-fly" claim is overstated, and the warm-up cost during the first ~10 steps of each episode represents a real performance penalty that is currently unquantified. Clarifying and formally characterizing this online vs. batch inference distinction would substantially sharpen the paper's contribution.

---

## Suggestions

1. **Enforce or report DAG validity:** Either incorporate a differentiable acyclicity penalty into Eq. (2), or empirically verify and report the proportion of inferred graphs that are cycle-free across all experiments.

2. **Add a Transformer-without-masking baseline:** Compare SAM against a Transformer dynamics model that uses the same architecture but predicts transitions densely without a learned causal mask. This is the essential ablation for separating Transformer capacity from causal inductive bias.

3. **Report results with multiple seeds and standard deviations:** Even with 3–5 seeds, reporting variance would substantially increase confidence in the quantitative comparisons.

4. **Formally define the test-time inference protocol:** In Section 4.2, specify explicitly how $\tau$ is constructed during deployment (i.e., growing prefix of the current episode), and in Section 5.6 clarify whether this matches training-time behavior.

5. **Add a meta-RL/contextual dynamics baseline:** Implement a simple trajectory-conditioned dynamics model (Transformer encoding trajectory prefix → latent context vector → dense dynamics prediction) and compare it to SAM. This is the most important missing comparison for the novelty argument.

6. **Fix the notation in Section 4.2:** Replace the confusing $q_{\phi}(\mathcal{G}|\tau) := f_{\theta_2}(\ldots)$ summary with separate, clearly distinguished equations for the mask predictor and dynamics model, and motivate the stop-gradient design.

7. **Address the ELBO claim:** Either derive Eq. (2) formally from a Bernoulli sparse prior (showing the KL term reduces to an L1 penalty), or relabel it as a sparsity-regularized reconstruction objective.

---

**Axis evaluations:**

- **Novelty:** Moderate. The episodic causal mixture setting is a fresh and well-motivated problem for RL. The Transformer-based differentiable masking is technically sound but not deeply novel in isolation.
- **Technical soundness:** Below acceptable. The DAG constraint is missing, the ELBO derivation is imprecise, and key notation is incorrect. These are not superficial issues.
- **Empirical support:** Weak for an ICLR submission. Two custom synthetic environments, no error bars, one key ablation missing, and no comparison to the natural meta-RL/contextual baseline class.
- **Significance:** Moderate potential. The problem is real and the empirical results are encouraging, but the evaluation scope is too narrow to substantiate broad impact claims.
- **Clarity:** Below average. The duplicated paragraph, notation errors, and underspecified inference protocol collectively impede understanding of the core method.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 3.0]
Average score: 3.0
Binary outcome: Reject
