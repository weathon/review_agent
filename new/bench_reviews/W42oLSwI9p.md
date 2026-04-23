Now I have enough information to write the final review. Let me synthesize everything carefully.

## Summary

This paper proposes three one-step diffusion-based solvers (CMILP, SCMILP, MFILP) for integer linear programming that extend beyond binary to non-binary integer variables. The key contributions are: (1) adapting consistency, shortcut, and meanflow training to ILP for fast one-step inference; (2) an Iterative Integer Projection (IIP) layer that enables differentiable approximation of non-binary integer values without costly binarization; and (3) a momentum-based objective-guided sampling scheme to improve solution quality.

## Strengths

- **The IIP layer is a clean, practical contribution for non-binary ILP.** The projection function $f_{\text{proj}}(\mathbf{x}) = \mathbf{x} - \frac{\sin(2\pi\mathbf{x})}{2\pi}$ (Eq. 3) is differentiable, converges to integer values in few iterations, and avoids the exponential blowup of binarization. Table 4 provides compelling evidence: CMILP on IM-(50,5,2) drops from 88% to 3% dataset feasibility when binarized, and on IM-(50,5,5) from 90% to 8%. The IIP layer is the paper's strongest contribution.

- **First end-to-end neural ILP solver for non-binary problems.** Extending beyond binary ILP is a meaningful and underexplored direction. The paper evaluates on both structured inventory management problems (Tables 2–3) and synthetic random ILP (Table 6), demonstrating broad applicability.

- **Near-perfect feasibility on binary ILP without post-processing.** All three methods achieve 100% sample feasibility across SC, CF, and CA datasets (Table 1), compared to IP Guided DDPM's 44.0–95.7% and IP Guided DDIM's 89.7–99.8%.

- **Strong non-binary results on synthetic datasets.** On Table 6, MFILP achieves 0.0% gap with 85% dataset feasibility on Random-(2000,20,2) in 19.4s, while IP Guided DDIM takes 46 minutes for 0.3% gap with 70% dataset feasibility. These are meaningful speed-quality improvements.

- **Momentum-guided sampling provides consistent improvements.** Table 5 shows MGD reduces gap from 104.5%→101.8% (Ti=10) and 99.8%→95.8% (Ti=20) while improving dataset feasibility on the hardest IM-(50,5,10) setting.

- **SCMILP's adjustable inference steps provide a practical speed-quality tradeoff.** The shortcut model allows varying the number of inference steps at test time, which is useful for deployment.

## Weaknesses

### Fatal
None.

### Major

- **The CMILP loss (Eq. 6) replaces the core consistency training objective with direct supervised regression, making the "consistency model" framing misleading.** The original consistency model trains via self-consistency: $f_\theta(\mathbf{x}_t, t) = f_\theta(\mathbf{x}_{t'}, t')$ for points on the same trajectory. Eq. 6 instead minimizes $d(f_\theta(\mathbf{x}_{t_n}', t_n, \mathcal{P}), \delta(\mathbf{x} - \mathbf{x}^*))$, directly regressing to the optimal solution $\mathbf{x}^*$. The paper explicitly justifies this: "Since the solution $\mathbf{x}^*$ is explicit given the problem instance, we can integrate $\mathbf{x}^*$ into the loss" — but then claims "Its minimization is achieved only if consistency holds across all possible trajectories" (line 90), which is vacuously true when both outputs are supervised to the same target. The one-step inference property comes from the architectural design (mapping any timestep to an output), not from enforcing self-consistency. While the paper says it is "inspired by" consistency models, naming the method "CMILP" and deriving the consistency function properties before silently replacing the loss creates a misleading impression of methodological continuity. This matters because it obscures what actually makes the method work — direct supervision with ground-truth solutions — which is a much simpler mechanism than consistency training. The SCMILP and MFILP variants are more standard adaptations and don't share this issue.

- **The abstract and introduction overclaim relative to the experimental evidence, particularly on binary ILP.** The abstract states the method "outperforms existing learning-based methods on both binary and non-binary instances," but on binary benchmarks (Table 1), CMILP's gap is 90.2% on SC vs. DDIM's 68.5%, 79.2% on CF vs. 54.6%, and 80.2% on CA vs. 25.4% — a 22–55 percentage point degradation. The introduction claims "comparable performance" with traditional solvers and diffusion baselines, which is not supported on binary benchmarks where gaps are 80–90%. On CF and CA, CMILP does beat IP Guided DDPM in gap (79.2% vs. 80.5% and 80.2% vs. 98.6%), and the paper's body text (Section 4.2) is more measured than the abstract. However, the abstract's unqualified "outperforms" claim is misleading. The non-binary results are substantially stronger and better support the paper's claims — the abstract should have been more precise about where the advantages lie.

- **Speed comparisons with traditional solvers lack equal-time baselines, making the "strong scalability" claim unsupported.** The paper emphasizes orders-of-magnitude speed advantages over Gurobi/SCIP, but the neural solvers produce solutions with 80–119% optimality gaps on most benchmarks while traditional solvers produce optimal or near-optimal solutions. A meaningful comparison would run Gurobi/SCIP for the same wall-clock time as the neural solver and compare solution quality at equal time budgets. Without this, the speed advantage is trivially true — any method that outputs a poor solution quickly is "faster" than one that finds the optimum. The abstract's claim of "strong scalability compared to traditional solvers" requires demonstrating that the neural solver provides better solutions per unit of compute time, not just that it terminates faster with much worse solutions.

### Minor

- **The Gap metric is computed only on instances where a feasible solution is found (Section 4.1), introducing selection bias.** A solver with low dataset feasibility (e.g., 15%) could appear to have a small gap by succeeding only on easy instances. The paper should either report gap computed over all instances (assigning infinite gap to infeasible cases) or explicitly discuss how the per-instance success rate affects gap comparability.

- **IIP train-test iteration mismatch is not analyzed.** The paper uses K=1 projection iterations during training and K>5 during testing (Section 3.2). This distribution mismatch could degrade performance, but no ablation studies are provided. An analysis of how performance varies with the training iteration count would strengthen the paper.

- **The guidance derivation in Section 3.3 is poorly explained.** The connection between the variational posterior formulation (Eqs. 7–8) and the practical gradient descent on latent variables is not made clear. The claim that "previous guidance methods can be viewed as a special case of gradient descent (with only a single optimization step)" oversimplifies — classifier-free guidance and classifier guidance in diffusion models operate on score functions, not on latent variable optimization.

### Trivial

- The Dirac delta notation $d(\cdot, \delta(\mathbf{x} - \mathbf{x}^*))$ in Eq. 6 is unconventional; if $d$ is simply an L2 or cross-entropy loss between the prediction and $\mathbf{x}^*$, this should be stated directly.

## Nice-to-Haves

- An equal-time comparison running Gurobi/SCIP for the neural solver's wall-clock time would make the speed claims much more convincing and is the single most impactful missing experiment.
- A hybrid approach combining the neural solver's fast initial solution with brief traditional solver refinement would address the poor gaps while retaining speed advantages.
- Analysis of failure cases: 10–38% of instances receive no feasible solution across 30 samples — characterizing these hard instances would be valuable.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"DiffILO's reported results seem anomalously bad"** — This is speculation about a baseline's performance, not a criticism of the paper's methodology. The paper reports DiffILO's results as-is; questioning whether they were misconfigured is not the authors' burden.
- **"Architecture description is vague"** — The paper references Nair et al. (2021) for the architecture and describes the CLIP-style contrastive encoder. Reproducibility concerns about architecture hyperparameters are minor for a systems-style paper.
- **"IIP function is well-known"** — The critic acknowledges novelty is in the application context, which is sufficient. The contribution is applying it to non-binary ILP, not inventing the function.
- **"Nearly 100% feasibility conflates dataset feasibility with reliable feasibility"** — The paper clearly defines and reports both sample feasibility and dataset feasibility as separate metrics (Section 4.1), so there is no conflation.
- **"Gaps of 80-120% mean solutions are worse than trivial heuristics"** — This is an overstatement. On binary benchmarks, the heuristics rins and feaspump also have 69–253% gaps. The gaps are bad but not categorically worse than trivial methods.

## Novel Insights

The paper reveals an important insight about the interplay between supervised signal availability and consistency training: when ground-truth solutions are accessible (as in ILP, where they can be computed offline by traditional solvers), directly supervising the consistency function output to $\mathbf{x}^*$ is more effective than enforcing self-consistency between timesteps. This suggests that the consistency model framework may be unnecessarily indirect for problems where target distributions are point masses — a finding that, while undermining the paper's own "consistency model" framing, points toward a simpler and potentially more effective paradigm for neural optimization solvers.

## Suggestions

- Rename CMILP to reflect what it actually does (e.g., "Direct-Supervision ILP" or "Regression-ILP") and clearly distinguish the architectural contribution (consistency function parameterization enabling multi-step inference) from the training objective modification (direct supervision replacing self-consistency). This would make the paper more honest without losing any substance.
- Add an equal-time comparison: run Gurobi with the neural solver's wall-clock time budget and report the resulting gap. This is the most impactful missing experiment and would either validate or temper the speed claims.
- Qualify the abstract's "outperforms" claim to specify the dimensions of improvement (speed, feasibility on binary; gap and speed on non-binary) rather than making an unqualified superiority claim.

## Score and Decision

**Calibration anchors:**

- **High anchors:** CMT (avg 7.0, consistency/meanflow training), Computational Bottlenecks for Denoising Diffusions (avg 7.33) — these are methodologically cleaner papers with stronger theoretical grounding. The current paper is clearly below these.
- **Medium anchors:** FMIP (avg 5.2, Accept Poster — flow-based MILP solver with joint continuous-integer modeling), RL-SPH (avg 5.0, Reject — RL-based ILP heuristic with limited baselines), Guide to Training CMs (avg 4.67, Reject — baseline fairness issues). The current paper has a cleaner core contribution (IIP layer) than RL-SPH but shares its overclaiming tendency. It has a similar problem setting to FMIP but weaker methodological transparency.
- **Low anchors:** VRG/Lagrangian Diffusion for MILP (avg 4.0, Reject — representation issues, weak ablations), Consistent DLM (avg 3.5, Reject — misrepresentation of theoretical formulation). The current paper is stronger than these — it has genuine, well-supported contributions, especially the IIP layer and non-binary results.

The paper sits between the medium and low anchors. Its IIP layer and non-binary ILP extension are genuine contributions that advance the field. However, the consistency model misrepresentation and abstract overclaiming are significant problems that, while not invalidating the results, undermine trust in the paper's framing. Compared to FMIP (5.2, accepted), this paper has a comparable contribution level but more serious honesty/transparency issues. Compared to RL-SPH (5.0, rejected), it has a cleaner contribution but similar evaluation concerns. The non-binary results in Table 6 are genuinely strong, which partially compensates for the binary benchmark weaknesses.

**Originality:** Moderate. The IIP layer is a clean engineering contribution; extending one-step diffusion to ILP is a natural application rather than a conceptual breakthrough. The CMILP loss modification reduces to supervised regression.

**Importance of research question:** High. Non-binary ILP is genuinely underexplored in the neural solver literature, and practical one-step inference is important for deployment.

**Claims support:** Partially. Non-binary claims are well-supported; binary claims are overclaimed. The "consistency model" framing is misleading.

**Soundness of experiments:** Moderate. Comprehensive evaluation across multiple benchmarks and metrics, but missing equal-time comparisons and the gap metric has selection bias.

**Clarity:** Moderate. The paper is generally readable but the CMILP loss deviation and guidance derivation are poorly explained.

**Value to community:** Moderate to high. The IIP layer and non-binary ILP framework are useful contributions that other researchers can build on.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>