=== CALIBRATION EXAMPLE 24 ===

# Harsh Critic Review
Now I have a thorough understanding of the paper. Let me write my review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title accurately describes the work, though "Online Fitting" may initially suggest something unrelated to gradient-based optimization; "Online Gradient Optimization" might be clearer. The abstract's claims are largely substantiated, but one implicit framing is misleading: the abstract presents emergent criticality and weight-distribution alignment as direct consequences of the optimization, when in fact the background input design (discussed below) introduces a significant confound. The claim that memory scales "only with the number of parameters rather than the entire temporal sequence" is accurate for D-RTRL but omits that the number of parameters is itself enormous (millions of synaptic weights), so the absolute memory use still needs to be benchmarked explicitly against BPTT on the same model.

---

### Introduction & Motivation

The motivation is compelling and the paper situates itself well in the literature. The core computational bottleneck of BPTT for whole-brain-scale models is real and important. However, the introduction overstates the novelty slightly: "resolving" the bottleneck implies the solution is general, whereas the method applies an existing algorithm (D-RTRL from BrainScale, Wang et al., 2024, a bioRxiv preprint) to a new biological system. The introduction should be clearer about what constitutes this paper's methodological contribution versus reapplication of BrainScale to the Drosophila setting. This matters for ICLR, where the machine-learning contribution needs to be clearly delineated.

---

### Method / Approach

**Background input design (Section 3.5) is circular and severely compromises evaluation.** The background input to each neuron at time *t* is defined as a ReLU nonlinear function of the neuropil firing rates at time *t*−1 (Eq. 3), which are themselves derived from the *training data targets*. This means during training, the model is given the target signal (one step delayed) as an explicit input. During testing, the model presumably feeds back its own predictions into this term, but this is never clearly stated. If so: (1) the fact that training loss reaches 0.03 is largely explained by this shortcut, and (2) the test-set generalization result (FC correlation 0.556) collapses partially because the model now must predict the encoding input from its own potentially diverging predictions. The paper needs to ablate the encoding input term entirely to establish what fraction of the signal is captured purely by the recurrent connectome dynamics.

**Massively underdetermined fitting problem.** The model has over 130,000 neurons with per-neuron time constants, millions of synaptic weights, and a large encoding weight matrix W_enc ∈ ℝ^{73 × 138639} — yet the entire fitting target is 73 scalar time series (neuropil firing rates). The parameter-to-observation ratio is astronomically high. While Appendix G demonstrates reproducibility across initializations, convergence to a unique solution in this context may simply reflect the strong pull of the encoding shortcut (see above) and the optimization landscape rather than genuine structure-function learning. A rigorous identifiability analysis is warranted, or at minimum the paper should acknowledge this limitation explicitly.

**Neurotransmitter polarity assignment (Section 3.4) requires justification.** Glutamate is classified as inhibitory (−1) throughout. In vertebrates this is incorrect; in Drosophila it can act on inhibitory GluCl receptors at some synapses but is also involved in excitatory signaling. Serotonin is similarly classified as uniformly inhibitory, which is a substantial simplification of neuromodulatory reality. A reference specifically supporting this choice in the fly brain context is needed, and the sensitivity of results to polarity misassignments should be discussed.

**Algorithm novelty vs. BrainScale.** The D-RTRL algorithm and the BrainScale platform are attributed to Wang et al., 2024 (bioRxiv). The paper does not clearly state which aspects of the online optimization constitute novel contributions beyond applying an existing algorithm to a new domain. ICLR readers need to know whether the paper's ML contribution is (a) the algorithm itself, (b) the connectome-constrained formulation, (c) the biological findings, or some combination. Appendix E's comparison between GRU+D-RTRL and GRU+BPTT adds methodological content, but the role of this paper versus the BrainScale paper must be clarified.

**Spectral radius as criticality proxy is problematic.** Section 4.4 and Fig. 5C track the spectral radius of the *weight matrix* converging to 1 as evidence of criticality. This is a valid heuristic for linear dynamical systems, but the network employs ReLU nonlinearities. For nonlinear networks, the relevant quantity is the spectral radius of the *Jacobian* (effective linearization), not the weight matrix. A weight matrix with spectral radius 1 does not guarantee dynamical criticality in a network with saturating or rectifying nonlinearities. This analysis should either be replaced with the appropriate Jacobian-based measure, or its limitations carefully acknowledged.

---

### Experiments & Results

**BPTT comparison is not fair (Section 4.1, Fig. 2B).** The comparison between online D-RTRL and BPTT is conducted on a *low-rank (k=10) approximation* of the weight matrix, not the full connectome connectivity. The justification given is that BPTT cannot handle the full connectome. While this is true, it means Figure 2B compares the online method on the full biological model versus BPTT on a qualitatively different (low-rank) model. The authors should at minimum show that D-RTRL and BPTT achieve comparable accuracy on the *same* low-rank model across multiple sequence lengths, and separately demonstrate why the full-rank model is intractable for BPTT. As presented, the memory comparison (Fig. 2A) is the strongest result; the convergence comparison is confounded by model architecture differences.

**Generalization performance is marginal and lacks statistical testing.** The main generalization result (Section 4.2) shows test FC correlation of 0.556 versus a baseline of 0.474 (direct similarity between training and test experimental data). The improvement of only 0.082 correlation points is presented without confidence intervals, standard deviations across the 18 available brains, or a statistical significance test. Is this difference significant given variability across brains? Only one brain's results appear to be shown in the main text, even though 18 brains were recorded. The paper should report mean ± standard deviation across subjects and include a significance test.

**Weight distribution comparison (Section 4.3) conflates distinct quantities.** The learned model weights |w_ij| are compared to the synapse count distribution in the FlyWire connectome (Fig. 4B). These are fundamentally different quantities: model weights are learned scalars reflecting functional coupling strength, while connectome synapse counts reflect anatomical connectivity. The fact that both develop heavy tails is interesting but not surprising — the optimization starts from a constrained, sparse connectivity matrix and gradient descent with weight decay or implicit regularization is known to produce sparse/heavy-tailed weight distributions in deep networks generally. The paper does not rule out that this convergence is a generic property of gradient-based optimization rather than a biologically meaningful structure-function correspondence. A control with random network topology (same degree sequence, random neurotransmitter assignment) would clarify whether the connectome structure specifically drives this convergence.

**Power-law fitting methodology is statistically inadequate (Appendix J.1).** The scaling exponent α is estimated via "linear regression in log-log space," which is known to produce biased estimates with poor statistical properties. The field standard (since Clauset et al., 2009, *SIAM Review*) is maximum likelihood estimation with xmin optimization and goodness-of-fit testing via the Kolmogorov-Smirnov statistic and likelihood ratio tests against alternative distributions (lognormal, exponential). The reported R² values (0.916 and 0.912) for power-law fits are not meaningful because R² is not a valid goodness-of-fit measure for log-transformed count data. The criticality claims could be substantially weakened or strengthened by rigorous power-law analysis, and this is a serious methodological gap.

**The early-training exponent α ≈ 3.86 is dismissed too quickly (Appendix J.2, Fig. S7B).** The paper acknowledges this is "notably higher than typical critical exponents, potentially indicating supercritical regime" but uses this to argue the model improves as training progresses. A supercritical exponent this far from 2 (nearly double) actually suggests the system is nowhere near criticality at initialization. The narrative of "spontaneous self-organization toward criticality" would be more convincing if the exponent trajectory from ~3.86 to ~1.90 were tracked explicitly across training epochs alongside the spectral radius in Fig. 5C.

**Missing cross-brain generalization.** The framework is trained on one brain's data and tested on a held-out segment of the *same brain*. With 18 brains available, cross-brain generalization (train on N-1 brains, test on the held-out brain) would be a much stronger validation of the model's generalizability.

**Sugar stimulation experiment (Appendix K) is illustrative but anecdotal.** The lateral asymmetry in MN9 response to unilateral gustatory stimulation (Table S1) is presented as a success, but there is no experimental comparison: is this asymmetry present in real Drosophila behavior, and does the magnitude and direction match? Without this grounding, the result is speculative.

---

### Writing & Clarity

Section 3.5 (Background Input) is the paper's most consequential methodological choice, yet it is described in only one short paragraph without discussion of the circularity concern. This needs substantially expanded treatment. Similarly, the relationship between D-RTRL and RTRL more broadly (and what "D" stands for — "Diagonal"?) is not explained for readers unfamiliar with BrainScale.

---

### Limitations & Broader Impact

The paper's limitations section is essentially absent — there is no dedicated discussion of the method's failure modes, the underdetermination problem, the simplified neuron model (rate-model, not spiking), or the neuropil-vs-neuron resolution gap. The simplification of all glutamatergic synapses as inhibitory is not acknowledged as a limitation. The assumption that calcium imaging data faithfully captures single-neuron firing rates via sparse deconvolution (Appendix A) is taken for granted, but this deconvolution from population-level neuropil signals (not individual cells) introduces substantial uncertainty. The paper would benefit from a frank limitations section.

---

### Overall Assessment

This paper addresses a genuine and important computational challenge — fitting whole-brain, connectome-constrained neural models over biological timescales — and demonstrates a technically impressive implementation using D-RTRL on the full 138,639-neuron FlyWire connectome. The memory-efficiency result is real and the scale of the model is a milestone. However, the scientific conclusions are substantially undermined by several critical issues: the background encoding input creates a circularity that makes the goodness-of-fit result difficult to interpret; the comparison with BPTT is conducted on different model architectures; the power-law/criticality analysis uses invalid statistical methodology; the weight-distribution alignment claim conflates learned weights with anatomical synapse counts without ruling out generic optimization artifacts; and the generalization performance is modest and presented without statistical rigor across the available 18 brains. The spectral-radius criticality argument is technically incorrect for nonlinear networks. For ICLR, where methodological rigor is paramount and machine-learning contributions must be clearly distinguished from domain applications, the paper in its current form does not meet the acceptance bar. A major revision addressing the encoding input circularity, improving the statistical methodology for criticality analysis, presenting cross-brain generalization results, and properly scoping the novelty relative to BrainScale could make this a strong submission.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces an online learning framework using dynamic recurrent topological learning rules (D-RTRL) to fit a 130,000-neuron *Drosophila* whole-brain model to resting-state calcium imaging data. By updating parameters in forward time, the method reduces memory requirements to scale with the number of parameters rather than sequence length, overcoming the computational bottlenecks of Backpropagation Through Time (BPTT). The optimized model successfully reproduces experimental functional connectivity and spontaneously develops synaptic weight distributions and dynamics consistent with criticality, demonstrating a scalable bridge between connectome anatomy and neural function.

### Strengths
1.  **Computational Scalability:** The paper effectively addresses the primary bottleneck in connectome-constrained modeling: memory usage. Figure 2A and Figure S1B provide clear evidence that the proposed online method scales independently of sequence length, enabling training on a single GPU where standard BPTT fails due to OOM errors. This is a significant practical contribution for large-scale neural modeling.
2.  **Biological Plausibility and Emergence:** The model does not just fit data but recovers structural and dynamical statistics absent from the optimization objective. Figure 4 demonstrates that learned synaptic weights evolve a heavy-tailed distribution matching the empirical FlyWire connectome. Furthermore, Figure 5 shows the network self-organizing toward criticality (spectral radius approaching 1, power-law avalanche exponents), providing mechanistic insight into how structure supports function.
3.  **Architectural Integration:** The integration of anatomical constraints (connectome topology, synaptic polarity) with data-driven optimization is well-executed. The comparison in Figure 2C and Section 4.1 quantifies the necessity of these biologically motivated inductive biases, showing that connectome-constrained readouts outperform unconstrained linear baselines in both training loss and generalization.

### Weaknesses
1.  **Baseline Comparison Asymmetry:** While the online method is compared to BPTT, the architectural baseline comparison (GRU vs. Connectome Model) is potentially confounded. The GRU baseline in Figure S2 lacks the anatomical constraints entirely (fixed weight matrix vs. learnable connectivity mapped to structure). While the connectome model is claimed to be superior in generalization, it is unclear if this is due to the learning algorithm or the architectural priors. A fairer comparison might involve online learning with random weights vs. the connectome.
2.  **Reproducibility and Code Availability:** For an ICLR submission, the explicit availability of code and the specific versioned data pipeline is crucial. The text mentions "BrainScale" and FlyWire data, but a direct link to the code repository (e.g., GitHub) or a detailed algorithmic appendix for the online update steps (beyond the provided equations) is necessary to ensure the community can reproduce these specific 130k-neuron results.
3.  **Generalization Horizon:** The training and testing window is 500 time steps each (approximately 400+ seconds at 1.2Hz). While sufficient for short-term dynamics, resting-state behavior in neuroscience often spans minutes to hours. The short horizon used to claim "resting-state dynamics" and "generalization" should be contextualized, as short-term correlations might not capture the full complexity of the resting-state attractor landscape.

### Novelty & Significance
**Novelty:** The core novelty lies in the application of online gradient-based optimization to a *full-brain, synaptic-resolution* connectome model at scale. While D-RTRL is known in the literature (cited Wang et al., 2024), its application to a 130k-neuron, 15M-synapse system with strict anatomical constraints to recover critical dynamics is a distinct contribution at the intersection of deep learning and systems neuroscience.
**Significance:** The work significantly impacts the feasibility of whole-brain modeling. By solving the memory bottleneck, it moves the field from small-scale or abstracted models to cellular-resolution models fitted to real data. The finding that optimization drives systems toward criticality reinforces theories of brain efficiency and provides a new tool for generating mechanistic hypotheses about brain states.

### Suggestions for Improvement
1.  **Enhance Baseline Analysis:** To isolate the specific contribution of the *online algorithm* versus the *connectome architecture*, consider adding an ablation where the online fitting is applied to a random connectivity network initialized similarly to the real connectome.
2.  **Clarify Code and Data Access:** Explicitly state where the code and pre-processed data are hosted. Ensure the supplementary material includes a detailed recipe for initializing the connectivity and tuning the online learning rates to achieve convergence (e.g., specific values for learning rate and batch size).
3.  **Longer-Term Dynamics Testing:** Extend the testing phase to evaluate if the critical dynamics and functional connectivity patterns persist over significantly longer simulation times (e.g., 10x the training duration) to rigorously validate claims about stable resting-state behavior.
4.  **Sensitivity Analysis:** Provide a brief analysis on how robust the emergence of criticality is to hyperparameter choices (e.g., learning rate, noise levels). If the system is too easily pushed to or away from criticality by small hyperparameter changes, the claim of spontaneous emergence might be weakened.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Shuffled Connectome Control:** Train a model with the same degree distribution but randomized connectivity to verify that the specific FlyWire topology drives the results. Without this, the "connectome-constrained" claim is unsupported, as global feedback loops alone could produce the dynamics.
2. **Background Input Ablation:** Remove or fix the learned background input term ($I^{enc}$) to determine if criticality emerges from the connectome or the engineered global feedback loop. If criticality persists without this term, the mechanistic claim is stronger; if not, the connectome's role is overstated.
3. **Parameter-Matched Random Baseline:** Compare against a sparse random network with the same number of neurons and parameters instead of a 256-unit GRU. The current baseline has orders of magnitude fewer parameters, making the performance comparison unfair and uninformative regarding architectural benefits.
4. **Extended Autonomous Generation:** Simulate the model for at least 10x longer than the 500-step test window to ensure stability and robust avalanche statistics. Criticality claims based on short traces are fragile and may reflect transient dynamics rather than a stable operating regime.

### Deeper Analysis Needed (top 3-5 only)
1. **Rigorous Power-Law Testing:** Replace $R^2$ fit metrics with Kolmogorov-Smirnov tests and likelihood ratio comparisons against exponential/log-normal distributions. $R^2$ is insufficient to confirm criticality and is a common statistical pitfall that undermines the paper's primary dynamical claim.
2. **Edge-Wise Weight Correlation:** Correlate learned synaptic weight magnitudes with actual FlyWire synapse counts per edge, not just global distributions. Matching the histogram is weak evidence; showing that strong learned weights align with high-count structural edges validates the biological interpretability.
3. **Generalization Scope Clarification:** Explicitly analyze whether test data is held-out time from the same average template or held-out individual flies. If the model only generalizes across time within a template, the claim of capturing biological dynamics rather than overfitting the average is weakened.

### Visualizations & Case Studies
1. **Avalanche Size-Duration Scaling:** Plot avalanche size versus duration to verify critical scaling exponents, not just duration distributions. True criticality requires specific scaling relationships between these variables that are currently missing.
2. **Functional Connectivity Error Map:** Visualize the difference matrix (Model FC minus Data FC) to identify specific neuropil pairs where the model fails. This reveals whether errors are random or structured around specific circuit motifs, informing where the model breaks.
3. **Spectral Radius vs. Loss Trajectory:** Plot the spectral radius evolution against training loss across multiple random seeds. This confirms whether the drive toward criticality (spectral radius ≈ 1) is a consistent optimization outcome or a stochastic artifact.

### Obvious Next Steps
1. **Stimulus-Evoked Fitting:** Demonstrate the framework fits task-based or stimulus-evoked data, not just resting state, to prove utility for mechanistic modeling. Resting state is unsupervised; showing evoked fitting validates the method for broader neuroscience applications.
2. **Wall-Clock Time Reporting:** Report total training time on the single GPU to validate practical scalability claims. Memory efficiency is meaningless if the online method requires prohibitive computation time compared to offline batches.
3. **Cross-Subject Validation:** Train on one fly's data and test on another to prove the model captures species-level dynamics rather than individual idiosyncrasies. This is the standard for validating whole-brain models and is currently absent.

# Final Consolidated Review
## Summary
This paper introduces an online gradient-based optimization framework for fitting connectome-constrained whole-brain neural models to experimental data. Using a D-RTRL algorithm that updates parameters in strictly forward time, the method reduces memory consumption from scaling with sequence length to scaling only with the number of parameters, enabling—on a single GPU—training of a 138,639-neuron *Drosophila* model constrained by the FlyWire connectome to match resting-state calcium imaging activity. The optimized model reproduces held-out neural dynamics, functional connectivity patterns, and spontaneously develops heavy-tailed synaptic weight distributions and critical dynamics characteristic of biological neural systems.

## Strengths
- **Computational scalability enabling a first-of-kind model scale.** The paper convincingly demonstrates that online learning eliminates the memory bottleneck that makes BPTT infeasible for whole-brain models. Figure 2A shows memory consumption independent of sequence length, and the achievement of fitting 138k neurons with millions of synapses on a single GPU is a genuine computational milestone. The comparison to BPTT memory requirements (which fail due to OOM even on 32GB GPUs for the full connectome) establishes practical feasibility where none existed before.
- **Emergent biological properties beyond the optimization objective.** The finding that training drives weight distributions toward heavy tails matching empirical connectome statistics (Fig. 4), and that spectral radius spontaneously approaches unity with training loss (Fig. 5C), suggests the optimization process recovers organizing principles rather than merely overfitting. The avalanche duration power-law exponent (α ≈ 1.90) matching experimental values (α ≈ 1.78) provides quantitative validation that the model captures a fundamental dynamical regime.
- **Clear demonstration that anatomical constraints provide meaningful inductive bias.** Figure 2C shows the connectome-constrained readout achieving lower training loss than an unconstrained linear readout, providing quantitative evidence that the anatomical prior aids fitting rather than merely constraining the hypothesis space arbitrarily.

## Weaknesses
- **Background input formulation creates a partial circularity during training.** During the warmup phase (Fig. S8), the encoding input $I_i^{[enc]}$ is computed from *ground-truth* neuropil firing rates at time $t-1$, meaning the model receives the target signal (one step delayed) as an explicit input. While teacher forcing is standard practice in sequence modeling, this design choice makes it difficult to assess how much of the model's performance derives from the recurrent connectome dynamics versus the background input shortcut. The paper should report performance with the encoding input ablated (fixed to zero or replaced with uninformative noise) to isolate the contribution of the connectome.
- **Massively underdetermined fitting problem limits interpretability of parameter recovery.** The model contains ~140,000 per-neuron parameters (time constants), millions of synaptic weights, and encoding weights $W^{[enc]} \in \mathbb{R}^{73 \times 138639}$—yet the fitting target comprises only 73 neuropil time series. While Appendix G demonstrates reproducibility across initializations, convergence to a consistent solution in such an overparameterized regime may reflect optimization dynamics rather than unique structure-function correspondence. The paper should explicitly acknowledge this identifiability limitation.
- **Power-law fitting methodology does not meet current statistical standards.** The avalanche duration exponents are estimated via linear regression in log-log space (Appendix J.1), which produces biased estimates and lacks proper goodness-of-fit testing. The field-standard approach (Clauset et al., 2009, *SIAM Review*) requires maximum likelihood estimation with $x_{min}$ optimization and Kolmogorov-Smirnov testing against the power-law null hypothesis, plus likelihood ratio tests comparing against exponential and log-normal alternatives. The reported $R^2$ values are not valid goodness-of-fit statistics for this purpose.
- **Weight distribution comparison conflates distinct biological quantities.** The paper compares learned synaptic weight magnitudes $|w_{ij}|$ to empirical synapse counts in the FlyWire connectome. While both exhibit heavy tails, these are fundamentally different quantities—one is a learned functional coupling strength, the other is an anatomical count. The convergence could be a generic property of gradient descent with sparsity-inducing structure rather than evidence of biological structure-function alignment. A random-network control (same degree sequence, shuffled connectivity) would clarify whether the connectome's specific topology drives this result.
- **Lack of cross-subject validation.** The dataset comprises 18 brains, yet the paper reports results from a single brain's training/test split (500 steps each). Cross-brain generalization (train on N-1 brains, test on held-out brain) would be substantially stronger validation that the model captures species-level dynamics rather than individual idiosyncrasies.
- **Spectral radius argument is technically incomplete for nonlinear networks.** Section 4.4 presents the weight matrix spectral radius approaching 1 as evidence of criticality. For networks with ReLU nonlinearities, the relevant quantity is the spectral radius of the *Jacobian* (the linearized dynamics around the operating point), not the weight matrix. While the avalanche analysis provides stronger evidence for criticality, the spectral radius claim should either use the Jacobian or acknowledge this limitation.

## Nice-to-Haves
- **Extended autonomous simulation for criticality validation.** Running the model for >10× the current test duration (500 steps) would strengthen claims about stable critical dynamics versus transient effects.
- **Edge-wise weight-to-synapse-count correlation.** Correlating learned weights with synapse counts per edge would validate that strong learned connections correspond to structurally strong connections, going beyond histogram matching.
- **Code and reproducibility materials.** The paper references the BrainScale platform and FlyWire connectome but should explicitly state code availability and preprocessing pipelines for reproducibility.

## Removed Points
- **BPTT comparison fairness concern.** The harsh critic claims the BPTT comparison is "unfair" because it uses a low-rank approximation rather than full connectome. However, the paper is transparent that both D-RTRL and BPTT are compared on the *same* low-rank model (Section 4.1), and separately demonstrates that BPTT fails on the full connectome due to memory. This is a reasonable experimental setup for isolating algorithm performance.
- **Neurotransmitter polarity classification as a "serious error."** The harsh critic questions glutamate being classified as inhibitory. In *Drosophila*, glutamate acts primarily through inhibitory GluCl receptors at many synapses. While a simplification, calling this a substantive error without specific counterevidence in the fly context is overstated.
- **GRU baseline parameter count mismatch.** The spark finder notes the 256-unit GRU has far fewer parameters than the connectome model. While true, this reflects the fundamental difference between biologically-constrained and unconstrained architectures; matching parameter counts would require abandoning the connectome constraint entirely, defeating the purpose of the comparison.
- **Demand for stimulus-evoked fitting.** The spark finder suggests demonstrating the framework on task-based or stimulus-evoked data. This is beyond the paper's stated scope (resting-state dynamics) and represents scope creep.

## Novel Insights
The emergence of critical dynamics as a *consequence* of fitting resting-state data under anatomical constraints—rather than explicit tuning to criticality—is the paper's most intriguing finding. If robust, it suggests that biological criticality may arise from the intersection of structure (connectome topology) and function (activity fitting) rather than requiring explicit homeostatic mechanisms. The trajectory of spectral radius toward unity with decreasing loss (Fig. 5C) provides preliminary mechanistic support for this hypothesis, though the statistical validation of criticality requires strengthening.

## Suggestions
- **Ablate the background input term during training** and report the performance gap. Even a reduced performance would clarify the connectome's contribution.
- **Apply rigorous power-law testing** (Clauset et al., 2009 methodology) to the avalanche distributions and report KS statistics and likelihood ratios against alternative distributions.
- **Add a shuffled-connectome control** where edge connections are randomly permuted while preserving degree sequence. If criticality and heavy-tailed weights still emerge, the claim of connectome-specific structure-function alignment is weakened.
- **Report cross-brain generalization** using the 18 available subjects. Even a modest correlation drop on held-out brains would provide important validation.
- **Acknowledge the underdetermination explicitly** in the discussion, noting that while reproducibility across initializations suggests optimization converges to a consistent region of parameter space, unique parameter identifiability cannot be guaranteed.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
