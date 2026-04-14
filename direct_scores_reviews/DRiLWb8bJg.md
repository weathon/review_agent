## Summary
This paper introduces Soft Analytic Policy Optimization (SAPO), a maximum entropy first-order model-based RL algorithm that incorporates entropy regularization into analytic gradient–based policy optimization for differentiable simulation. Alongside SAPO, the authors present Rewarped, a GPU-parallel differentiable multiphysics platform built on NVIDIA Warp that supports rigid bodies, articulations, and multiple deformable materials (elastic, plasticine, fluid) simultaneously. Together, these contributions enable, for the first time, scaling RL to tasks involving deformable objects, with SAPO outperforming all evaluated baselines across six locomotion and manipulation tasks.

---

## Strengths

- **Filling a genuine system gap:** Rewarped is demonstrably the first parallel *differentiable* simulator to simultaneously support rigid, articulated, and multiple deformable materials. Table 1 makes this clear — every prior platform lacks at least one of differentiability, parallelism, or deformable material coverage — and the paper's footnotes about competitor limitations are well-sourced and defensible.

- **Consistent and substantial empirical gains across all six tasks:** SAPO outperforms all baselines on every task evaluated (Table 2). The improvements are not marginal — e.g., 1820.5 vs. 853.3 on SoftJumper over SHAC, and 90.0 vs. 32.7 on HandFlip — with tight confidence intervals across 10 seeds. This breadth and consistency is meaningful evidence, not cherry-picking.

- **Ablation that cleanly isolates the entropy contribution:** Table 3 and Figure 3 on HandFlip show a clear ladder: SHAC (33) → SHAC + design choices III–V (56) → SAPO without entropy in actor but with soft value (77) → full SAPO (90). The entropy component in the actor objective contributes roughly 40% of the total gain beyond SHAC, a non-trivial fraction. The paper is transparent that design choices III–V also matter, which is scientifically honest.

- **Technically grounded integration of MaxEnt and FOBG:** The derivation of SAPO's first-order estimate (Eq. 19) is mathematically correct and cleanly connects the maximum entropy objective to the FOBG framework through entropy-augmented *k*-step returns (Eq. 20). The paper also explicitly places SAPO relative to SAC (off-policy Q-learning) and SVG(H) (learned world model), clarifying what is new.

- **CUDA graph–based gradient checkpointing for scalable backward passes:** The use of CUDA graph capture for both forward and backward passes of MPM simulation (Section 5.1) is a non-trivial engineering contribution that enables memory-efficient computation of batched analytic gradients over many substeps. This is a concrete and reusable technique for the community.

---

## Weaknesses

- **Ablation study limited to a single task (HandFlip):** All ablations in Section 6.2 are conducted only on HandFlip. Given that the paper claims entropy regularization specifically stabilizes deformable optimization, it is critical to verify this holds across material types. Notably, ablation (c) (design choices III–V, no entropy) was shown to have "minimal impact" on DFlex rigid-body tasks (Appendix F.3), which already suggests the entropy contribution may be task-dependent. Without analogous ablations on, e.g., SoftJumper (elastic) or FluidMove, the claim that entropy is generally beneficial for deformable tasks rests on one data point.

- **One-way coupling is a significant physical limitation that is under-disclosed:** Section 5 mentions "one-way coupling from kinematic articulated rigid bodies to MPM particles" only in passing, and Table 1's feature comparison does not flag this. In tasks like HandFlip and RollingFlat, the rigid bodies receive no reaction forces from deformables. This is a fundamental constraint on physical realism — the robot hand or rolling pin is unaffected by the dough it contacts — and should be explicitly listed as a platform limitation in the main text and Table 1, not only buried in a single clause of Section 5.

- **Wall-clock training time is absent:** The paper argues for sample efficiency gains, but differentiable simulation with MPM and automatic differentiation incurs higher per-step compute than non-differentiable rigid-body simulators. Without wall-clock time comparisons (or at minimum FPS measurements per method), it is impossible to assess whether SAPO's sample efficiency advantage translates into real training speedups relative to PPO/SAC on Isaac Gym equivalents. For a paper framed around "scaling RL," this is a critical missing metric.

- **Gradient computation of entropy through the state sequence is not explicitly addressed:** In Eq. (19), $\nabla_\theta (R_{0:H}^\alpha + \gamma^H V_{\text{soft}}(s_H))$ differentiates through both the explicit $\theta$-dependence of the policy distribution and the implicit dependence via states $s_t(\theta)$. For the entropy term $\mathcal{H}_\pi[a_t|s_t]$, the chain rule includes $\partial\mathcal{H}/\partial s_t \cdot \partial s_t/\partial\theta$, which passes through the simulation dynamics. For a squashed Gaussian with state-dependent variance (design choice III), this matters for the correctness of the FOBG estimate. The paper does not discuss whether this path is included or treated as a stop-gradient, which leaves a technical gap in the derivation.

- **Observation space terminology is misleading:** Section 6 describes deformable tasks as having "high-dimensional (particle-based) visual observations," but the Limitations section acknowledges that policies use "non-occluded subsampled particle states from simulation." These particle state observations are not visual in the pixel sense, and they are physically unattainable in the real world. The paper should consistently use "particle state observations" to avoid implying visual/pixel-based proficiency, and clarify whether the main results in Table 2 use state-based or particle-based inputs for each task.

- **SAC performs catastrophically on SoftJumper (−161.8) without explanation:** SAC achieves a return worse than a random policy on SoftJumper. The paper does not explain whether this is due to the parallel on-policy data collection scheme being incompatible with SAC's replay buffer, poor tuning, or a fundamental mismatch. Since SAC is a primary model-free baseline, understanding why it completely fails on this task (while performing competitively on AntRun) is important for interpreting the comparative results.

- **Temperature target heuristic imported from model-free setting without justification:** Design choice I uses $\bar{\mathcal{H}} = -\dim(\mathcal{A})/2$ following Ball et al. (2023), which was developed for model-free off-policy learning. No analysis is provided of whether this target entropy is appropriate for the FO-MBRL setting where gradient signals are qualitatively different. Sensitivity analysis of this hyperparameter across tasks is absent.

- **HandReorient exhibits a qualitative-quantitative disconnect:** SAPO achieves the highest return on HandReorient (221.7 vs. all baselines near 0 or negative), yet the paper explicitly states the policy is "only capable of catching the cube and preventing it from falling." This raises questions about reward calibration for HandReorient — either the modified differentiable reward function does not well-align with the true task objective, or the success metric is too loose. This disconnect should be analyzed more carefully.

---

## Nice-to-Haves

- **Ablation on rigid-body tasks to isolate entropy benefit:** Running the full SAPO ablation ladder on AntRun or HandReorient would strengthen the claim that entropy regularization is broadly beneficial across material types, not just deformable manipulation.

- **Entropy evolution curves over training:** Plotting policy entropy vs. training steps for SAPO vs. SHAC would directly support the mechanistic claim that entropy prevents convergence to local minima, rather than citing Ahmed et al. (2019) as the sole justification.

- **Gradient norm statistics during contact events:** Measuring and reporting mean/variance of gradient norms for SAPO vs. SHAC during contact-rich phases would empirically substantiate the "landscape smoothing" hypothesis.

- **Horizon length sensitivity analysis:** FO-MBRL performance is known to be sensitive to truncation horizon $H$; reporting performance across a range of $H$ values for deformable tasks (where gradients may vanish faster than in rigid-body tasks) would strengthen the experimental characterization.

- **Broader discussion of AHAC's relationship to SAPO:** AHAC (Georgiev et al., 2024) also stabilizes SHAC by adapting horizon based on contact stiffness, and shares some design choices with SAPO (no target networks, critic ensemble). A clearer delineation of the complementary vs. competing stabilization strategies would be informative.

- **Platform scaling benchmark:** Reporting parallel efficiency (environments per GPU) for Rewarped relative to Brax or Isaac Gym would validate the scalability claim quantitatively.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[REMOVED] Table 1 omits DaXBench:** The harsh critic claimed DaXBench was absent from Table 1, making comparisons appear more favorable to Rewarped. This is factually incorrect — DaXBench appears as "DaXBenCh" in Table 1 with appropriate feature markings. The critic misread the table.

- **[REMOVED] TrajOpt comparison is unfair because it lacks caveats about open-loop nature:** The paper explicitly describes TrajOpt as "gradient-based trajectory optimization… to optimize for an open-loop action sequence" in the Baselines paragraph of Section 6. The open-loop nature is clearly disclosed to readers.

- **[REMOVED] PPO/SAC reward function mismatch concern phrased as favoring author's method:** The concern about smoothed differentiable reward functions for HandReorient is a valid reproducibility question but not an unfair comparison; PPO/SAC also receive the same smoothed reward (necessary for the shared environment), and any advantage from smoothing accrues equally to all methods. The comparison is not asymmetric in a direction that benefits SAPO over baselines.

- **[WEAKENED → nice-to-have] Requesting confidence intervals or per-seed statistics at larger scale / more seeds:** The paper already uses 10 seeds with 95% CIs, which is above the norm for this community. Requesting more seeds would be excessive.

- **[WEAKENED → nice-to-have] Sim-to-real transfer as a required evaluation:** This is outside the paper's stated scope (simulation platform + RL algorithm). Requesting a real-world deployment is scope creep for an algorithmic/systems paper.

- **[WEAKENED] Lack of theoretical proof for landscape smoothing:** The paper frames the entropy-smoothing hypothesis empirically and cites relevant theoretical support (Ahmed et al., 2019). Requiring formal proofs for an empirical systems paper is not standard in this community.

---

## Novel Insights

The most genuinely novel observation to emerge from the synthesis is the **asymmetric benefit of entropy regularization across material types**: ablation (c) in Section 6.2 shows that design choices III–V (architectural/optimization improvements) have "minimal impact" on rigid-body DFlex tasks but account for ~half of the gain on HandFlip. This suggests that entropy regularization and associated stochastic parameterization may be specifically beneficial for deformable contact dynamics — where the reward landscape is more non-smooth and gradient signals are noisier — rather than representing a general improvement over SHAC across all differentiable simulation settings. This interaction between material type and the utility of entropy regularization is not fully exploited in the paper's analysis but could guide future FO-MBRL algorithm design.

---

## Suggestions

- Replicate the ablation table (Table 3) on at least two additional tasks with different material types (e.g., SoftJumper for elastic, FluidMove for fluid) to confirm that entropy in the actor objective is a consistent driver of improvement, not an artifact of the HandFlip reward structure.
- Add a main-text table of wall-clock training time per method (hours to convergence) and Rewarped FPS under varying parallelism levels; position this alongside Table 2 to give practitioners the full cost-benefit picture.
- Explicitly add one-way coupling to Table 1's feature set (or a footnote row for "two-way coupling") and to the Limitations section, since this shapes what physical phenomena Rewarped can faithfully model.
- Clarify the entropy gradient computation: state explicitly whether $\partial\mathcal{H}_\pi[a_t|s_t]/\partial\theta$ is computed with or without the chain rule through $s_t(\theta)$, and whether this choice affects empirical results.
- Replace "high-dimensional (particle-based) visual observations" with "high-dimensional particle state observations" throughout to accurately describe what the policies receive, and confirm in a single sentence which observation type each row of Table 2 uses.
- Provide a brief diagnostic for the HandReorient result: compute a task success rate (cube reoriented to target) as a secondary metric alongside return, to clarify whether return of 221.7 corresponds to any meaningful degree of task completion.

---

## Evaluation

**Novelty:** Moderate. SAPO is conceptually a natural extension of SHAC with maximum entropy augmentation — the combination is well-motivated but not surprising. The primary novel contribution is Rewarped as a parallelized differentiable multiphysics platform; no prior work offers this combination of capabilities. The integration of entropy regularization specifically for deformable simulation instability is a meaningful but incremental algorithmic insight.

**Technical soundness:** Mostly solid. The mathematical derivation is correct and well-presented. The unresolved question about entropy gradient computation through the state sequence is a technical gap that should be addressed. The one-way coupling limitation is physically significant but acknowledged.

**Empirical support:** Good. Consistent results across six tasks with 10 seeds and 95% CIs set a reasonable standard. The main weakness is ablations limited to a single task, leaving the entropy attribution claim partially under-validated. Missing wall-clock comparisons are a notable gap for a paper about scalable simulation.

**Significance:** High for the robotics and deformable manipulation communities. Enabling RL-scale training on deformable tasks — which previously required motion planning or trajectory optimization — is a meaningful step. Rewarped as a shared platform has practical value for the field.

**Clarity:** Good overall. The derivation is clean and the paper is well-organized. The observation space terminology is a genuine source of confusion that should be resolved.

MY FINAL SCORE: <pineapple>6.2</pineapple>