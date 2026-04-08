=== CALIBRATION EXAMPLE 11 ===

# Final Consolidated Review
## Summary

This paper introduces an online gradient-based learning framework (D-RTRL) for fitting connectome-constrained whole-brain models, replacing backpropagation through time (BPTT) to achieve memory scaling with the number of parameters rather than sequence length. Using the FlyWire *Drosophila* connectome (138,639 neurons, ~15M synapses) as a fixed anatomical scaffold, the authors optimize synaptic weights, time constants, and background input parameters to match resting-state calcium imaging data. The trained model reproduces neuropil-level dynamics, and the authors report that optimization spontaneously produces heavy-tailed synaptic weight distributions matching connectome statistics and drives the network toward critical dynamics (power-law neuronal avalanches, spectral radius → 1).

## Strengths

- **Scalability enabling previously infeasible experiments:** The core contribution—reducing memory from O(T·P) to O(P) via forward-time eligibility traces—is demonstrated concretely: a 138k-neuron whole-brain model trains on a single GPU where BPTT fails entirely (Fig. 2A). This is not a generic scaling claim; it specifically unlocks a class of connectome-constrained whole-brain fitting experiments that were computationally prohibitive, as corroborated by the prior mouse visual cortex work requiring 160 GPUs for 50k neurons (Chen et al., 2022).

- **Connectome-constrained readout provides meaningful inductive bias:** The comparison in Fig. 2C shows that the anatomically constrained readout (Eq. 4, synapse-weighted neuropil aggregation) converges to lower training loss than an unconstrained linear readout. This is a specific, non-obvious result demonstrating that the connectome structure itself carries functional information, beyond simply providing a sparse wiring diagram.

- **Two emergent properties from a single optimization objective:** The simultaneous emergence of both heavy-tailed weight statistics (Fig. 4) and critical dynamics (Fig. 5) from fitting resting-state activity—without explicit regularization for either—is a striking finding. If validated with proper controls, it would suggest deep structure–function links recoverable from data-driven optimization.

## Weaknesses

- **Critical ambiguity in the training data flow for the background input term (Eq. 3):** Equation 3 defines $I_i^{enc}(t)$ as a function of $FR^{neuropil}_k(t-1)$, but the paper does not specify in the main text whether this refers to ground-truth experimental firing rates or the model's own predicted firing rates during training. Supplementary Fig. S8 clarifies a two-stage pipeline (warm-up with ground-truth feedback, then prediction with autonomous feedback), but this critical detail is absent from Sections 3.5–3.7. If the warm-up phase trains with ground-truth feedback, the background input term acts as a strong bypass from target data directly into neuron dynamics—potentially allowing the model to minimize loss by propagating this signal through to the readout rather than learning meaningful recurrent dynamics ($I_i^{conn}$). No ablation tests the model's behavior when the background input source is switched or removed. This ambiguity undermines confidence in the generalization claims.

- **No controls for claimed emergent properties:** The two most striking results—heavy-tailed weight distributions and critical dynamics—are presented as emergent properties of fitting biological data under anatomical constraints. However, no control experiments establish that these properties are specifically tied to (a) the biological training target, (b) the connectome constraint, or (c) both. It is not shown whether training on shuffled or synthetic activity data, or training on a randomized graph with the same sparsity, would produce qualitatively similar weight distributions and spectral radii approaching 1. Without such controls, these properties could be generic effects of gradient-based optimization in sparse recurrent networks rather than meaningful signatures of biological organization.

- **BPTT comparison performed only on a simplified surrogate, not the actual connectome model:** The central algorithmic comparison (Fig. 2B) uses low-rank weight factorization rather than the true 138k-neuron connectome-constrained network. The paper acknowledges this is necessary because BPTT cannot run on the full model, which itself validates the motivation—but it means the convergence quality of D-RTRL on the actual connectome problem relative to BPTT remains undemonstrated. The comparison shows D-RTRL matches BPTT on a proxy; whether it matches on the real problem is unknown.

- **Massively underdetermined inverse problem with limited analysis of solution meaning:** The model optimizes millions of parameters (synaptic weights, time constants, encoding weights for 138k neurons) against only 73 neuropil-level signals sampled at 1.2 Hz. Appendix G shows that different initializations converge to similar neuropil-level weight profiles, which addresses reproducibility at the population level—but this does not establish that individual neuron-level parameters are meaningfully constrained. The underdetermined nature means many parameter configurations could reproduce the observed population-averaged signals, raising questions about whether recovered single-neuron properties (e.g., avalanche dynamics at 138k-neuron resolution) reflect biology or overfitting to a degenerate solution space.

- **Criticality analysis requires more rigorous validation:** (i) Avalanche detection (Appendix J) uses a threshold of 3σ, but criticality claims are notoriously sensitive to threshold choice. The paper does not test robustness across a range of thresholds (e.g., 2σ–4σ). (ii) Power-law fitting via linear regression in log-log space with R² as the sole metric is insufficient; no goodness-of-fit test (e.g., Kolmogorov-Smirnov) or comparison to alternative distributions (exponential, log-normal) is provided. (iii) The temporal resolution of 1.2 Hz (~833 ms per sample) is coarse for defining neuronal avalanches, which typically require millisecond-scale resolution. (iv) The spectral radius approaching 1 is presented as a mechanistic explanation, but for a nonlinear ReLU network, the spectral radius of the weight matrix is less informative than the spectral radius of the effective Jacobian; the paper does not clarify which is measured.

- **GRU baseline suffers from severe capacity mismatch:** The GRU comparison (256 hidden units vs. 138k neurons, Appendix F) cannot isolate the benefit of connectome constraints from the benefit of simply having vastly more parameters. The GRU's failure to exhibit criticality (Appendix Fig. S7A) may reflect finite-size effects rather than architectural differences. A fairer baseline would be a comparably sized RNN with randomized or shuffled connectivity.

- **Glutamate assigned inhibitory polarity without adequate justification:** Section 3.4 assigns glutamate as inhibitory (−1), but glutamate signaling in *Drosophila* is complex—it is the primary excitatory neurotransmitter at the neuromuscular junction and can be excitatory or inhibitory in the CNS depending on receptor composition. Treating all glutamatergic connections as inhibitory is a significant simplification that could systematically bias the recurrent dynamics. The paper does not discuss or investigate the effect of this assignment.

## Nice-to-Haves

- Cross-individual generalization test (train on N-1 brains, test on held-out individual) to strengthen the claim that learned parameters capture generalizable dynamics rather than individual-specific patterns.

- Comparison to alternative scalable recurrent training methods beyond BPTT (e.g., other RTRL approximations, reservoir computing approaches) to establish that D-RTRL specifically provides advantages.

- Wall-clock training time reporting: while memory efficiency is the primary claim, practical utility also depends on computational cost per step, which is not reported.

- Perturbation experiments (simulated lesions or silencing) to test whether the model makes testable predictions about interventions, strengthening the "mechanistic model" claim.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Weakness: Missing related work on e-prop and local learning rules.** The critic suggested the paper should discuss alternative online/local learning algorithms to position D-RTRL. Per rules, I do not flag missing related works as I cannot confirm their existence or relevance independently.

- **Weakness: No discussion of truncated BPTT or gradient checkpointing as alternatives.** The critic suggested the paper should explain why these standard memory-saving techniques are insufficient. This is a reasonable methodological question, but the paper does implicitly address it: Fig. S1B shows BPTT memory scaling even with low-rank connections exceeds GPU capacity at moderate sequence lengths, and the full connectome makes BPTT infeasible regardless (Section 4.1). The critique that truncated BPTT wasn't discussed is partially addressed by the empirical demonstration.

- **Weakness: Reproducibility concerns about undisclosed hyperparameters.** Per rules, nitpicks about reproducibility such as undisclosed hyperparameters are removed.

- **Weakness: Equation notation ambiguity (FR used for both data and model output).** This is a formatting/style nitpick. While the notation could be clearer, it doesn't constitute a substantive weakness.

- **Weakness: The approach is not biologically plausible because it uses gradient descent.** The paper makes no claim of biological plausibility for the learning algorithm—it explicitly frames this as a computational framework. Criticizing it for not being biologically plausible in its credit assignment is scope creep.

## Novel Insights

The most provocative finding is that fitting to resting-state activity under anatomical constraints spontaneously drives the network toward critical dynamics (spectral radius → 1, power-law avalanches) *and* produces weight distributions matching connectome statistics—both without explicit regularization. If validated with proper controls, this would suggest that critical dynamics and heavy-tailed connectivity are not independent biological features but rather joint consequences of a brain operating under functional constraints imposed by its anatomy. This aligns with the "criticality as optimal computation" hypothesis but provides a novel mechanistic path: the connectome topology, combined with the requirement to match real population dynamics, may be sufficient to push the system toward criticality as an implicit optimization outcome. However, this insight remains provisional without the controls noted above—it could alternatively reflect generic properties of gradient-based optimization in structured sparse networks.

## Suggestions

- **Resolve the training data flow ambiguity explicitly in the main text:** Add a clear statement in Section 3.5 or 3.7 specifying whether $FR^{neuropil}$ in Eq. 3 uses ground-truth or model-predicted values during each training phase. If both are used (as Supp. Fig. S8 suggests), describe the transition protocol and include an ablation showing the model's behavior without the background input term.

- **Add at least one control for the emergent properties:** Train the same model architecture on temporally shuffled activity data (preserving marginal statistics but destroying temporal structure) and report whether heavy-tailed weights and critical dynamics still emerge. This single experiment would dramatically strengthen or weaken the central claims.

- **Strengthen the criticality analysis:** Report avalanche distributions for at least two additional thresholds (e.g., 2σ and 4σ), and perform a formal goodness-of-fit test (e.g., Clauset-Shalizi-Newman method) comparing power-law vs. exponential fits. This is now standard practice in criticality analysis.

- **Provide a capacity-matched baseline:** Compare against a sparse RNN with similar dimensionality (e.g., ~138k units with randomized sparse connectivity of similar sparsity) to isolate the contribution of the connectome topology from the contribution of model scale.

## Axis Evaluations

- **Novelty:** Moderate-to-good. The combination of online learning with connectome-constrained whole-brain modeling is novel, and the scale achieved is a genuine first. However, D-RTRL itself is prior work (Wang et al., 2024), and the individual components (rate models, connectome constraints, criticality analysis) are established. The novelty lies in the integration and the emergent findings.

- **Technical soundness:** Moderate concerns. The training procedure ambiguity (Eq. 3 data flow), lack of controls for emergent properties, and underdetermined optimization undermine confidence in the strongest claims. The core scalability contribution is sound.

- **Empirical support:** Mixed. The scalability demonstration is strong and well-supported. The dynamics reproduction (Fig. 3) is adequate but shows a notable train-test gap (FC: 0.998 → 0.556). The criticality and weight distribution claims lack the controls needed for full conviction.

- **Significance:** Potentially high if claims hold. Enabling whole-brain connectome-constrained fitting on a single GPU removes a major bottleneck in computational neuroscience. The emergent structure–function links, if validated, would be significant. But the current evidence is not yet conclusive.

- **Clarity:** Adequate but with a critical gap. The main methodological sections are clearly written, but the omission of the training data flow detail from the main text is a significant clarity failure for a paper whose central claims depend on the training procedure.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
