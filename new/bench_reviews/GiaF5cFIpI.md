Now I have a good understanding of the paper and the calibration papers. Let me construct the final review.

Let me verify a few specific claims from the harsh critic by checking the paper again.

Key things to verify:
1. Whether results on "Designed" in Fig 4a use s_obs or s(u) (the predicted response)
2. The "10-20 stimulations" claim
3. Whether there's any circularity in evaluation
4. The optimization formulation issue with ∥u∥_max_0

For item 1, let me re-read the relevant section:

The text around Fig 4 says: "We can assess the degree to which a stimulation had the desired effect by checking the angle between v, the effect of stimulation we desired, and s_obs, the deviation from previously predicted dynamics." This strongly suggests that Fig 4a uses s_obs, not predicted s.

For Fig 4b: "we quantified how well we can target perturbations in arbitrary directions in the latent space by comparing the s(u) from equation (8) to v" - this explicitly states Fig 4b uses predicted s(u).

Fig 4c explicitly compares predicted vs observed.

So the "circularity" claim is partially valid for 4b but not for 4a or 4c. The 4a comparison (the main comparison showing Designed > Random etc.) uses observed responses, addressing the key claim.

For the "10-20 stimulations" claim - this appears in the abstract as "roughly 10-20 total stimulations delivered." But looking at the paper body, where is this actually demonstrated? The toy model has binary stimulations happening frequently. On real data, there's no explicit convergence plot by number of stimulations. The claim in the abstract says "quickly learn a stimulation-response mapping within roughly 10-20 total stimulations delivered" but the evidence for this seems limited.

For the optimization formulation - ∥u∥_max^0 is defined in context as "max number of nonzeros" (i.e., the target sparsity), so λ₁(∥u∥_max^0 − ∥u∥₁) penalizes deviation from the L1 approximation of the target number of nonzeros. This is actually a reasonable formulation though the notation could be clearer.

Now let me synthesize.</think>Now I have a thorough understanding of the paper and the relevant calibration papers. Let me write the final review.

## Summary

This paper presents a streaming framework for adaptively modeling and optimizing neural stimulation patterns to drive latent neural dynamics in desired directions. It integrates three components: (1) streaming latent space construction (including a novel streaming jPCA variant called sjPCA), (2) a nonparametric kernel regression model for stimulus-response mapping that adapts to non-stationarity, and (3) a constrained optimization procedure that designs high-dimensional stimulation patterns under feasibility constraints (non-negativity, sparsity, magnitude limits). The method is evaluated on simulated data and two real neural datasets (calcium imaging and electrophysiology) with simulated (not actual) stimulations, demonstrating sub-10ms average runtimes.

## Strengths

- **Addresses an important and underexplored problem.** Designing stimuli to causally perturb specific latent directions in real time is a recognized need in neuroscience, yet methods that combine streaming latent tracking, stimulus-response modeling, and constrained optimization are scarce. The paper directly tackles this gap with a coherent end-to-end framework.

- **Well-motivated experimental constraints.** The optimization explicitly handles non-negativity (excitation-only optogenetics), sparsity (limited targets), and magnitude limits. The analysis of infeasible directions (inhibition vs. excitation vs. sparse feasible targets) is physically grounded and informative (e.g., the "Negative" and "Dense" conditions in Fig. 4b).

- **Adaptive handling of non-stationarity.** The temporal kernel K₃ in the stimulus-response model (Eq. 7) allows discounting old observations, and Fig. 2e demonstrates recovery from both abrupt discontinuities and continuous drift in the stimulus-response mapping. This is a practically important feature for real experiments.

- **Novel sjPCA with Procrustes stabilization.** The streaming jPCA variant with Orthogonal Procrustes alignment (Eq. 2) is a clean contribution, and Fig. 1a demonstrates convergence to offline fits.

- **Fast runtimes.** End-to-end computation averaging <10ms per timepoint and always <100ms makes the method genuinely compatible with real-time closed-loop experiments, which is critical for the intended application.

- **Transparent evaluation framework.** The paper distinguishes between predicted error (angle between ŝ and v) and observed error (angle between s_obs and v), and Fig. 4c directly compares the two, acknowledging that predicted error serves as a "loose lower bound" on observed error.

## Weaknesses

### Major:

- **All real-data stimulation results are simulated, not from actual perturbations.** For both calcium imaging and electrophysiology datasets, stimulations are injected via a simple autoregressive model (y_t = r_t + a_t, a_t = 0.8·a_{t−1} + u_t). The core claim is about designing stimulations that "drive latent dynamics in arbitrary directions" in real neural systems, but the stimulus responses are never observed from actual neural tissue—they are simulated with a linear model. The discussion acknowledges this ("our real data experiments were performed offline, though in a realistic streaming setting"), but frames the results as validating the method's capability, when in reality this tests only that the algorithm can operate on data streams with hand-crafted stimulation effects. This is the most significant gap between the paper's claims and its evidence.

- **Baselines for stimulation optimization are too weak.** The primary comparison baselines are random single-neuron stimulation, random multi-neuron stimulation, and shuffled versions of designed stimuli (Fig. 4a). These are trivially weak alternatives. The cited existing methods—Bayesian optimization (Minai et al., 2024), input-output dynamical modeling (Yang et al., 2021), optimal experimental design (Wagenmaker et al., 2024)—are not compared against. Even simple heuristics (e.g., stimulating neurons with largest loadings on the target direction) would provide a more informative baseline. Without such comparisons, it remains unclear whether the proposed optimization offers meaningful advantages over simpler principled approaches.

- **Sample efficiency claim of "10–20 stimulations" is not well-supported.** The abstract states the method "quickly learns a stimulation-response mapping within roughly 10–20 total stimulations delivered," but no figure directly plots alignment or prediction error as a function of the number of observed stimulations. The toy model uses continuous binary stimulation over hundreds of timepoints. The claim is central to the practical value of the method and needs rigorous substantiation.

- **Scalability of the kernel regression estimator is not analyzed.** The estimator Ŝ(x,u,t) in Eq. (7) computes over all N previously observed stimulus-response pairs at each step. As the number of observations grows, this computation increases linearly, yet there is no discussion of pruning, budgeting, or computational complexity analysis—particularly concerning for a method that must run in real time.

### Minor:

- **The optimization objective has minor notational ambiguity.** The term ∥u∥₀^max in Eq. (8)/Algorithm 1 is described informally as "max number of nonzeros" but the notation is unconventional and could cause confusion about whether it is a fixed target or an adaptive parameter. The L1 relaxation of the L0 constraint and its properties are not discussed (convergence, sensitivity to λ₁).

- **The single-pending-stimulus assumption is restrictive.** The method assumes at most one stimulus is pending at a time (Section 2.3), which precludes overlapping stimulation effects common in high-throughput optogenetic experiments. This limitation is acknowledged only briefly and not explored.

- **The adaptive latent space selection mechanism is proposed but not evaluated.** The parallel evaluation of multiple latent representations (Fig. 1c) and the Bubblewrap-based prediction error mechanism for selecting among them are described but not shown to improve downstream stimulation design or dynamical prediction.

- **No evaluation of multi-step control behavior.** The optimization targets single-step alignment, but practical applications (e.g., BMIs) require driving the system toward a target region over multiple time steps. Whether repeated application of optimized stimuli can consistently steer trajectories is not tested.

### Trivial:

- The "simple vs. non-simple" S mapping distinction in Fig. 5 is not rigorously defined (dimensions, nature of nonlinearity), making it hard to assess the generality claim.

## Nice-to-Haves

- A comparison to at least one principled alternative (Bayesian optimization, input-output LDS, or greedy heuristic) for stimulation design would substantially strengthen the contribution.
- A quantitative convergence analysis plotting alignment vs. number of observed stimulations would directly validate the sample-efficiency claim.
- A pilot experiment with even a simple real stimulation paradigm (e.g., in a simple preparation) would be transformative for the paper's impact.
- Reporting actual sparsity levels of designed stimuli (||u||₀ distribution) would help assess whether the L1 relaxation achieves intended sparsity targets.

## Removed Points

- **"Circular evaluation" claim that all results are measured against predicted responses.** This is overstated. Fig. 4a explicitly uses s_obs ("the angle between v...and s_obs, the deviation from previously predicted dynamics"), and Fig. 4c compares predicted and observed errors. The predicted-vs-desired analysis in Fig. 4b is explicitly flagged as measuring optimization quality, not real-world efficacy. The paper is transparent about what each panel shows.

- **Criticism that the stimulation optimization is evaluated principally against the model's own predicted mapping rather than observed responses.** As above, the main comparison (Fig. 4a) uses observed responses s_obs, not predicted s. The 4b panel explicitly measures optimization quality against predicted s, which is appropriate for that analysis.

- **Demands for comparison with specific unreleased or unavailable methods.** The harsh critic demands comparisons to methods like Minai et al. 2024 and Wagenmaker et al. 2024. While these would strengthen the paper, the paper does cite them and differs in scope (e.g., Minai et al. uses Bayesian optimization over a discrete set of stimuli, while this method optimizes over continuous stimulus space). The lack of such comparisons is a legitimate weakness but the specific demand for these exact methods is not necessary.

- **Criticism that the paper overclaims by using "real-time" and "in vivo compatible" framing.** The paper clearly states runtimes are demonstrated on commodity hardware and that real-data experiments were "performed offline, though in a realistic streaming setting." The "in vivo compatible" claim is specifically about the runtimes and constraints, not that in vivo experiments were conducted. This framing is reasonable for a methods paper proposing a framework for future experiments.

- **Demand for behavioral outcome evaluation.** The paper explicitly scopes out behavioral outcomes ("we did not include any explicit consideration of the effects of stimulations on behavior") and this is outside the stated scope of stimulating latent neural dynamics.

- **Nitpick about ∥u∥₀^max being undefined.** The notation is described in the text ("the number of neurons or channels to target") and in context is the target sparsity. While unconventional, it is interpretable.

## Novel Insights

The paper's most interesting contribution is the feasibility-based decomposition of the stimulation optimization problem: explicitly showing that inhibition (Negative) and blanket excitation (Dense) are infeasible directions under non-negativity and sparsity constraints, while sparse excitation along principal component directions (Q₀, Feasible) is achievable. This provides an elegant geometric characterization of what stimulation directions are actually reachable given realistic optogenetic constraints—a point that transcends the specific algorithm and speaks to the fundamental structure of the problem. Separately, the finding that the nonparametric estimator learns "simple" and "non-simple" stimulus-response mappings at approximately the same rate (Fig. 5a) is notable, though it warrants more rigorous characterization.

## Suggestions

- Add a direct comparison to at least one principled stimulation-design baseline (even a simple top-k neurons heuristic) to contextualize the optimization's advantages.
- Plot alignment metrics (angle between s_obs and v) as a function of number of observed stimulations to rigorously support the "10–20 stimulations" claim.
- Discuss or demonstrate computational scaling of the kernel regression as observations accumulate, and consider adding a budgeting or pruning mechanism.
- Acknowledge the simulated-stimulation limitation more prominently in the abstract or introduction, not just in the discussion, to avoid misleading readers.
- Consider evaluating multi-step steering (repeated stimulation toward a target region) rather than only single-step alignment.

## Score and Decision

**Calibration comparison:**  
- Papers with similar weakness profiles (simulated-only perturbations, weak baselines) like iSSM (FwW3jqchtY, scores 3-6, rejected) and realSEUDO (b7DsNJYmeo, scores 3-5-6, withdrawn/rejected) received low scores.  
- The closed-loop EEG framework paper (4ltiMYgJo9, scores 3-6-8, rejected) had a similar gap between claimed generality and actual validation.  
- FCCA (4AlNpszv66, scores 3-3-5-8, rejected/withdrawn) had theoretical novelty but weak experimental validation.  
- BRAID (3usdM1AuI3, scores 6-8-8-3, accepted) had real data with actual perturbations and stronger baselines.  
- The neural transfer function paper (PqGPjts7Ug, scores 3-8-3, withdrawn) had similar concerns about in silico-only validation.

This paper sits above the clearly weak papers (iSSM, realSEUDO, closed-loop EEG) in that it has a more coherent and complete algorithmic framework, faster runtimes, and a clearer problem formulation. However, it shares their critical weakness: no validation with real perturbations. The simulation-only evaluation on real data, combined with the weak baselines, substantially undermines the core claims about being able to "drive neural dynamics." The framework itself is valuable and well-constructed, but the evidence is at the simulation/blueprint level. The paper is comparable to but somewhat stronger than papers in the 4-5 range (realSEUDO, closed-loop EEG) due to its more comprehensive framework and runtime demonstration, but clearly below papers with real perturbation data like BRAID (6-8).

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>