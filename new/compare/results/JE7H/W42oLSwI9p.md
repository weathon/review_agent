---
job_id: c76be041-27f4-4dc2-988c-29f2a088fe90
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: W42oLSwI9p.pdf
paper: One-step Diffusion Solver for Non-binary Integer Linear Programming
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes diffusion-based neural solvers for ILP/MILP, touching generative models, optimization, and graph learning, all squarely within ICLR’s scope.

## Minimum Quality
Pass ✅.  
All required sections are present (Abstract, Introduction, Related Work, Methodology, Experiments/Results, Conclusion). The contributions are nontrivial, the methods are technically meaningful, and the empirical study is substantial, despite several weaknesses and clarity issues.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts or attempts to manipulate automated reviewing systems in the provided content.

---

# Expected Review Outcome:

## Summary

The paper introduces three one-step diffusion-based neural solvers for integer linear programming (ILP): CMILP (consistency-style), SCMILP (shortcut/flow-matching-style), and MFILP (meanflow-style). The models operate on a bipartite-graph representation of ILPs, use a CLIP-like encoder to align problem and solution features, and incorporate an Iterative Integer Projection (IIP) layer to handle general bounded non-binary integer variables without binarization. A gradient-based objective-guided sampling procedure with momentum is further proposed to refine diffusion samples. Experiments on classic binary ILP benchmarks, synthetic non-binary ILPs, and inventory management problems show large speedups over vanilla diffusion solvers and competitive or better feasibility and gaps compared to several learning-based baselines, while traditional solvers remain stronger in optimality.

## Strengths

1. **Clear conceptual step toward fast, learned ILP solvers.**  
   The paper tackles a very relevant pain point in diffusion-based ILP solvers: long inference trajectories. By instantiating three one-step (or few-step) variants (consistency, shortcut, meanflow), the work provides a coherent framework for fast sample generation. The architecture in **Figure 1** gives a reasonably clear overview of the pipeline from ILP bipartite graph encoding to diffusion-based feature generation and decoding back to integer solutions.

2. **Extension from binary to general bounded integer variables with IIP.**  
   The proposed Iterative Integer Projection (IIP) layer in **Equation (3)** is arguably the most distinctive technical ingredient. It is a differentiable mapping defined on the entire real line, iteratively sharpening real-valued outputs to lie near integers using \(f_{\mathrm{proj}}(x) = x - \frac{\sin(2\pi x)}{2\pi}\). **Figure 2** illustrates that as the iteration count \(K\) grows, \(f^{(K)}_{\mathrm{proj}}(x)\) approaches a staircase-like rounding function. Avoiding binary expansion is practically important, and the experimental comparisons in **Table 4** (non-binarized vs binarized IM datasets) nicely support the claim that binarization inflates dimensionality and harms both runtime and feasibility for neural solvers.

3. **Objective-guided sampling framed as non-convex optimization and extended with momentum.**  
   The paper reinterprets prior “guidance” for diffusion-based ILP as essentially one-step gradient descent on a constraint-penalized objective \(l(\mathbf{x};\mathcal P)\) in **Equation (8)**, then extends this to multi-step gradient descent with momentum (**Equation (9)**). **Figure 3** visually clarifies the difference between plain guidance and momentum-based guidance (MGD) by showing how MGD tracks a smoother path in latent space. **Table 5** provides evidence that momentum consistently improves dataset feasibility and reduces gaps on the most difficult IM-(50,5,10) setting for a fixed number of inference steps.

4. **Substantial experimental coverage, especially for non-binary ILP.**  
   Beyond the standard binary Ecole benchmarks (SC, CF, CA; **Table 1**), the paper includes multiple families of non-binary ILP instances:
   - Inventory management problems with varying dimensions and bounds (**Tables 2–4, 8**),  
   - Synthetic Random-(n,m,b) ILPs (**Tables 6–7**).  
   On many of these, the proposed methods match or beat diffusion baselines (IP Guided DDPM / DDIM) in runtime by large margins while maintaining competitive or better gaps and feasibility. For instance, in **Table 6**, on Random-(500,20,2) and Random-(300,30,2) (**Table 7**), CMILP/MFILP reach near-zero gaps in a few seconds versus tens of minutes to hours for the diffusion baselines.

5. **Feasibility penalty is empirically justified.**  
   The inclusion of the feasibility penalty \(\mathcal L_{\text{penalty}}\) in **Equation (2)** is not just an ad-hoc extra term; **Table 8** is a useful ablation. Both the proposed models and IP Guided DDPM/DDIM completely fail to find feasible solutions when the penalty coefficient is zero (NaN gaps and 0% dataset feasibility), while turning on the penalty restores nontrivial feasibility. This is an important design detail for training ILP solvers that is often hand-waved in similar work.

6. **Reasonably solid performance on binary ILP with huge speedups over full diffusion.**  
   On the binary SC/CF/CA tasks (**Table 1**), the one-step models achieve 100% dataset feasibility and competitive gaps compared to IP Guided DDPM/DDIM, but with 1–2 orders of magnitude less inference time (seconds vs minutes/hours). For example, on SC, MFILP finishes in 21.3s vs IP Guided DDPM’s 11h and DDIM’s 65m while maintaining >88% sample feasibility. Even though Gurobi still gives zero gap, the learned solvers clearly dominate prior neural and diffusion baselines on wall-clock efficiency.

## Weaknesses

1. **Optimality gaps remain large compared to state-of-the-art (both classical and some neural) on key tasks.**  
   While the paper emphasizes speed, the gaps relative to the “ground truth” (Gurobi under 100s) are frequently very large, and this is underplayed in the narrative.  
   - In **Table 1**, CMILP/SCMILP/MFILP show gaps around 76–92% on SC and CF, and 79–85% on CA, which means solutions are far from optimal even though feasibility is high. Some neural baselines like PS and DiffILO also have large gaps, but IP Guided DDIM often achieves smaller gaps (e.g., 25.4% on CA vs 79–85% for the proposed solvers) albeit more slowly.  
   - On non-binary inventory problems with larger bounds, gaps become extreme: in **Table 2** for IM-(50,5,10) the methods have ~107–119% gap, which is hard to justify as “comparable performance” to diffusion baselines.  
   - Even the momentum-augmented SCMILP in **Table 5** still shows ~95–105% gaps.  
   From an optimization viewpoint, these gaps are so large that many users would simply not accept these solutions, which makes the practical impact somewhat limited in its current form. The paper should be more upfront about the tradeoff: it is primarily a fast heuristic for generating *feasible but often far-from-optimal* solutions.

2. **IIP layer lacks theoretical analysis and has nontrivial dynamical issues.**  
   The integer projection function \(f_{\text{proj}}(x) = x - \frac{\sin(2\pi x)}{2\pi}\) in **Equation (3)** is interesting but is only justified visually via **Figure 2**. There is no analysis of its convergence properties, fixed points beyond integers, or stability. In fact:  
   - \(f_{\text{proj}}(k) = k\) for integer \(k\), but the derivative at \(x\) is \(f'_{\text{proj}}(x) = 1 - \cos(2\pi x)\). At integers, the derivative is zero (good), but around half-integers it equals 2, which suggests *local expansion*. So starting near midpoints between integers could actually push points away from those midpoints in a somewhat unstable manner.  
   - The paper does not state whether projection is applied elementwise and whether there are bounds (e.g. clipping) to prevent divergence if magnitudes grow.  
   - There is no formal connection to nearest-integer rounding, no guarantee that iterating \(f\) K times converges to the nearest integer, or even converges at all for arbitrary real inputs.  
   This is not just theoretical pedantry; the IIP is central to how non-binary integrality is enforced. Some mild analytic results (e.g., identifying basins of attraction around each integer and bounding the error after K steps) would substantially strengthen the claim that this is a reliable alternative to explicit rounding or relaxed softmax / Gumbel tricks.

3. **Ambiguity in where and how IIP interacts with training and decoding.**  
   Section 3.1 states that “The projection is applied once during training for training efficiency and applied multiple times during testing for approximation accuracy.” However, there are several missing details that impact reproducibility and clarity:  
   - Is IIP applied in the decoder, in the feature space, or directly on final solution logits/values? Does it sit *before* computing \(\mathcal L_{\text{penalty}}\) and the objective-guided loss \(l(\mathbf{x};\mathcal P)\), or after?  
   - For non-binary problems the reconstruction loss is MSE; is that computed on pre- or post-projection values?  
   - During gradient-based sampling (Section 3.3), are gradients backpropagated through IIP, or is the projection treated as a non-differentiable post-processing step during test-time optimization of \(\eta\)?  
   The experimental section only says “integrality is handled separately through the Iterative Integer Projection described below,” but this separation is not concretely specified in formulae. Gray-box integrality handling makes it difficult to reason about the method and to replicate the results precisely.

4. **Objective-guided diffusion formulation is mathematically loose and notation is sometimes inaccurate.**  
   The derivation in Section 3.3 mixing plug-and-play priors, Dirac measures, and KL-like objectives is quite compressed and in places imprecise:  
   - **Equation (6)** uses a distance \(d(f_\theta(\cdot), \delta(\mathbf{x}-\mathbf{x}^*))\). In practice the model outputs a parametric distribution or a mean vector, not a distribution equal to a Dirac delta. Treating the ground-truth solution as a delta is fine conceptually, but the distance function between a neural output and a delta distribution is never instantiated; presumably this is just some \(\ell_2\) or cross-entropy loss on predicted solutions. The notation here is misleading.  
   - In **Equation (7)**, \(F\) is said to be minimized over \(\eta\), but some constants such as \(-\log Z\) and \(-\mathbf{y}^*\) are included inside the expectation in a way that does not strictly depend on \(\eta\). There is no explicit gradient expression, and the step from the general plug-and-play framework to the specific \(l(\mathbf{x};\mathcal P)\) of **Equation (8)** is essentially asserted rather than derived.  
   - The relation between \(q(\mathbf{h}|\eta,\mathcal P)\), \(p_\theta(\eta,\mathbf{h}|\mathcal P)\), and the learned reverse process \(\mathbf{x}_t\) used in CMILP/SCMILP/MFILP is also left vague; e.g., in the consistency model setting, which “latent path” is used for guidance?  
   Overall, the guidance mechanism is interesting, but the math presentation is not at the level one would expect for ICLR and makes it quite hard to check correctness.

5. **Evaluation vs traditional solvers is somewhat superficial and tuned in favor of the learned methods.**  
   Although Gurobi/SCIP/COPT are included throughout, the comparisons rarely address the central question practitioners care about: under a fixed wall-clock budget, what gap-feasibility tradeoff do you get from neural vs classical methods? Some concerns:  
   - On binary ILP (**Table 1**), Gurobi is artificially limited to 100s and SCIP to 16.7 minutes, but the learned solvers are allowed to train offline (which is fine) and then run once. There is no discussion of how changing Gurobi’s time budget (or MIP gap tolerances) shifts the comparison. As is, Gurobi always outputs optimal solutions (0% gap) with 100% feasibility, and is often *faster* than diffusion baselines on small to medium-scale non-binary problems (e.g., Random-(500,20,2) in **Table 6**), so the advantage of one-step diffusion is somewhat scenario-dependent.  
   - On harder inventory problems (**Table 3**, IM-(100,10,2)), Gurobi and SCIP still get 0% gap and 100% feasibility in under 10 minutes, while the proposed methods have 16–18% gap and 62–69% dataset feasibility in ~20 seconds. Presenting this as clearly preferable depends strongly on application needs and is never really analyzed.  
   - Several classical heuristics (rins, feaspump) are run with what appear to be generic default settings; there is no evidence that their time limits or parameterizations were tuned comparably to those of the neural methods.  
   A more balanced comparison such as “under X seconds total, across Y instances, what is the Pareto front of (gap, dataset feasibility)?” would make the empirical contribution much more convincing.

6. **Some experimental results are noisy, inconsistent, or insufficiently explained.**  
   There are multiple peculiarities in the tables that deserve discussion:  
   - In **Table 2**, traditional methods (rins/feaspump) sometimes have 0% gap but very low dataset feasibility, whereas the proposed methods have >10% gap but higher feasibility. Since the gap is computed only on instances where a feasible solution is found, this can be misleading: tiny feasibility sets can still yield near-zero average gap. The paper should explicitly analyze this effect.  
   - In **Table 6**, CMILP achieves 0% gap on Random-(500,20,2) yet only 46.8% sample feasibility and 85% dataset feasibility; conversely, IP Guided DDIM gets slightly worse gap (0.7%) but 85.1% sample feasibility and 100% dataset feasibility. Claims in the text like “accurately solve most instances in significantly less time than Gurobi and SCIP” overlook these feasibility gaps.  
   - **Table 5** appears to list the first two rows identically labeled “SCMILP (T_i = 10, Opt+MGD)” but with slightly different numbers; one of these must be “Opt+GD” or similar. There are also typos like “rms”, “nus”, “leaspump”, and wrong author spellings across tables.  
   - Hyperparameters such as the number of IIP iterations at train vs test, the diffusion/flow time schedulers \(N_t, N_{t,d}, N_{r,t}\), and the exact gradient step sizes / momentum coefficients are not fully specified, which limits reproducibility.  

7. **Positioning relative to closely related MILP-diffusion and differentiable ILP work is incomplete.**  
   The Related Work section covers classic and some neural MILP solvers, plus the guided diffusion method of Zeng et al. (2024), but misses some directly relevant recent work that is very close in spirit:  
   - A feasibility-aware diffusion model for MILP that combines Lagrangian relaxation with generative modeling could give comparable or better feasibility guarantees with different tradeoffs in speed/quality.  
   - Differentiable ILP solvers integrated into neural pipelines for tasks like NLI show alternative ways to build end-to-end differentiable integer reasoning modules.  
   These lines of work are important both conceptually (alternative ways of enforcing constraints and integrality) and empirically (alternative baselines on small and medium ILPs).

Overall, while none of these issues is a showstopper individually, they collectively indicate that the paper, in its current form, is better read as a promising step toward fast feasibility-oriented ILP samplers rather than as a mature, fully competitive alternative to classical solvers or carefully tuned diffusion baselines.

## Potentially Missing Related Work

1. **Wang, R., Li, X., Wang, M. (2025). “Lagrangian Meets Diffusion: Feasibility-aware Generative Modeling for Mixed Integer Linear Programming.”**  
   This work appears to study diffusion models for MILP with an explicit emphasis on feasibility via a Lagrangian or constraint-augmented objective. It is conceptually very close to the objective-guided diffusion and feasibility penalty in this paper. It should be discussed in Section 2 (Neural Solvers for IP) and Section 3.3 (Objective Guided Sampling), and ideally compared empirically on at least one shared MILP benchmark (or similar settings).

2. **Thayaparan, M., Valentino, M., Freitas, A. (2024). “A Differentiable Integer Linear Programming Solver for Explanation-Based Natural Language Inference.”**  
   This paper introduces a differentiable ILP solver integrated with neural representations in an end-to-end fashion. It is directly relevant to the IIP layer and the idea of differentiable integrality constraints. It should be cited in Section 2 (Neural Solver for IP) and briefly contrasted in Section 3.1 or 3 (Methodology) as an alternative approach to differentiable integrality and constraint satisfaction.

3. **Xiang, M., Rossi, R., Martin-Barragan, B. (2017). “Computing Non-stationary (s, S) Policies Using Mixed Integer Linear Programming.”**  
   While primarily an operations-research application, this work deals specifically with inventory management via MILP, which is used extensively as a testbed in this paper. It would be useful background in Section 4.3.1 to connect the proposed inventory formulations to established MILP formulations in the literature and clarify how realistic or stylized the generated datasets are.

4. **Gondzio, J., Yildirim, E. A. (2018). “Global Solutions of Nonconvex Standard Quadratic Programs via Mixed Integer Linear Programming Reformulations.”**  
   This paper shows how nonconvex programs can be reformulated as MILPs, suggesting potential future extensions of the proposed solvers beyond ILP to certain nonconvex problems. It could be usefully cited in the Introduction or Conclusion when discussing the broader scope and applications of MILP-formulated problems.

5. **Czégel, A., G.-Tóth, B. (2025). “A Quantum Constraint Generation Framework for Binary Linear Programs.”**  
   While more distant, this provides another alternative paradigm (quantum algorithms) for accelerating binary linear programs. A brief mention in Section 2 could help contextualize this work as one of several “new paradigms” (learning-based, quantum, etc.) for tackling ILP beyond classical branch-and-bound.

## Questions

1. **IIP convergence and hyperparameters.**  
   - Can you provide a short analysis or at least empirical evidence on the convergence behavior of the IIP mapping \(f_{\text{proj}}\)? For example, for random initial values in \([-B,B]\), how many iterations are needed to get within 0.1 of the nearest integer with high probability?  
   - What values of \(K\) are used during training vs testing across datasets, and how sensitive are results (especially feasibility and gaps) to \(K\)?

2. **Precise interaction between IIP, feasibility penalty, and objective-guided sampling.**  
   - In Equation (2), is \(\mathcal L_{\text{penalty}}\) computed on IIP-projected solutions or on raw decoder outputs?  
   - In Equation (8) during guidance, are gradients taken w.r.t. pre-projection \(\mathbf{x}\) and then IIP applied after each step, or do you differentiate through IIP? Clarifying this pipeline, perhaps with a diagram parallel to **Figure 1**, would be very helpful.

3. **Clarifying the distance \(d(\cdot,\cdot)\) and the actual loss used for CMILP.**  
   - In Equation (6), what is the concrete form of the distance function \(d\) between \(f_\theta(\mathbf{x}_t,t,\mathcal P)\) and \(\delta(\mathbf{x}-\mathbf{x}^*)\)? Is it an \(\ell_2\) loss on mean solutions, cross-entropy over discrete variables, or something else?  
   - How does this relate to the standard consistency loss in Song et al. (2023)? A short explicit formula would make CMILP much easier to reproduce.

4. **Tradeoff versus classical solvers under equal time budgets.**  
   - Could you provide plots or tables showing, for a fixed wall-clock budget (e.g., 10s, 30s, 60s), the Pareto tradeoff between (gap, dataset feasibility) for your methods vs Gurobi/SCIP and IP Guided DDIM? This would help clarify under which time regimes your methods are clearly preferable.

5. **Robustness to scale and instance distribution shifts.**  
   - How does performance change when test instances have substantially more variables or constraints than those in training (e.g., train on Random-(500,20,2), test on Random-(2000,20,2)) without retraining? Some initial scaling experiments are present, but an explicit “out-of-distribution” scaling study would strengthen the claim of scalability.

6. **Ablations on momentum and gradient-descent iterations beyond one dataset.**  
   - Currently **Table 5** only shows MGD vs GD on IM-(50,5,10). Could you provide at least one more dataset (binary or synthetic Random-(n,m,b)) where you compare GD vs MGD over varying number of steps \(T_i\)? It would clarify whether the observed 2–4% gain in dataset feasibility is consistent or dataset-specific.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The methods are conceptually solid and empirically validated across multiple datasets, but there are mathematical gaps (especially around IIP convergence and objective-guided derivations) and some evaluation choices and notational looseness that reduce confidence in the full rigor of the claims.

## Presentation Rating

3: good.  
The core ideas and architecture are understandable, and figures like **Figure 1–3** help. However, there are nontrivial clarity issues in the math, missing implementation details, and several typos/inconsistencies in tables and references that make the paper feel less polished.

## Contribution Rating

3: good.  
The combination of one-step diffusion, a differentiable integer projection for non-binary ILP, and momentum-based guided sampling is a meaningful step forward for learned ILP solvers, especially in terms of speed and handling of general bounded integers. The large optimality gaps and incomplete theoretical treatment, however, limit the overall impact.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper offers a coherent and practically relevant contribution to fast learned ILP solving, with interesting ideas (IIP, one-step diffusion variants, momentum guidance) and an extensive experimental study, particularly for non-binary ILPs where literature is thin. At the same time, optimality gaps are often large, several mathematical and algorithmic details are under-specified, and comparisons to classical solvers and closely related diffusion-based MILP work could be more balanced and complete. With additional theoretical clarification and stronger empirical analysis, this line of work could be quite influential.

## Reviewer Confidence

4: confident.  
I am reasonably familiar with diffusion models, combinatorial optimization, and neural MILP solvers, and I carefully checked the main equations and experimental tables, though I did not fully reconstruct every derivation.