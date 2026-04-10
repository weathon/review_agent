=== CALIBRATION EXAMPLE 21 ===

# Final Consolidated Review
## Summary
This paper introduces an online gradient-based optimization framework (D-RTRL) to fit large-scale, connectome-constrained whole-brain models to neural activity data, overcoming the memory limitations of Backpropagation Through Time (BPTT). The authors demonstrate this by training a *Drosophila* model of over 130,000 neurons with a fixed anatomical scaffold from the FlyWire connectome to match resting-state calcium imaging. They report that the trained model generalizes temporally, yields synaptic weight distributions resembling empirical connectome statistics, and drives network dynamics toward a critical regime.

## Strengths
- **Demonstrated Scalability and Memory Efficiency:** The core technical contribution is convincingly validated. Figure 2A shows memory usage scaling only with batch size, not sequence length, enabling training of a 130K-neuron model on a single GPU—a task explicitly stated to be infeasible with standard BPTT.
- **Effective Integration of Anatomical Priors:** The model successfully integrates multiple biological constraints: binary connectivity and synaptic polarity from the FlyWire connectome, and a synapse-weighted neuropil readout. The result that the connectome-constrained readout outperforms an unconstrained linear readout (Fig. 2C) provides clear evidence for the value of these anatomical priors.
- **Emergence of Biologically Plausible Properties:** The optimization yields two compelling, non-enforced outcomes: the learned synaptic weight distribution develops a heavy tail aligning with empirical connectome statistics (Figs. 4, S6, S9), and the network's spectral radius approaches unity, correlating with the appearance of critical avalanche dynamics (Fig. 5). These suggest the framework can reveal organizational principles.

## Weaknesses
### Major:
- **Invalid Criticality Comparison Undermines a Key Claim:** The analysis in Section 4.4 compares avalanche duration exponents from experimental *neuropil-level* calcium signals with exponents computed from the model's *single-neuron* firing rates. Avalanche statistics are highly sensitive to the level of spatial aggregation; comparing population-level exponents to single-cell exponents is not scientifically sound. The claim that the model "reproduces" the critical dynamics of the resting state is therefore not supported by the presented evidence. A proper comparison requires analyzing avalanches in the model's neuropil-level readouts.
- **Ambiguous Mechanistic Role of the Background Input:** The background input (Eq. 3) is a learned function of global neuropil activity. While the paper clarifies that during autonomous testing the model uses its own predictions (Fig. S8), the design still allows a potentially unconstrained driving signal to compensate for shortcomings in the recurrent circuit. The paper provides no ablation study (e.g., removing recurrent connections or fixing background weights) to demonstrate that the optimized *recurrent connectivity* is necessary or primary for generating the observed dynamics. This limits the mechanistic interpretability of the fitted model.
- **Incomplete Benchmarking of Optimization Efficiency:** The direct performance comparison between online learning and BPTT (Fig. 2B) uses a low-rank weight approximation, not the full connectome-constrained model. While the claim that BPTT is infeasible for the full model is supported by memory arguments, the claim that online learning achieves "comparable" performance is based on an indirect comparison between two different model architectures (low-rank vs. connectome-constrained). A more rigorous evaluation of optimization efficacy on the same task is needed.

### Minor:
- **Simplified Neural Model:** The use of threshold-linear firing-rate units abstracts away spiking mechanisms and richer biophysics. While practical for scale, this choice limits the biological realism of the single-neuron dynamics.
- **Limited Generalization Analysis Beyond Temporal Continuity:** Generalization is evaluated on a temporally held-out segment from the same aggregated dataset. While the test functional connectivity correlation (0.556) exceeds the experimental test-train correlation (0.474), stronger evidence—such as leave-one-fly-out cross-validation or performance on different behavioral states—would bolster claims about learning shared dynamical principles.

### Trivial:
- The difference between the experimental (1.78) and model (1.90) avalanche exponents, while noted, is a minor point if the core comparison were valid.

## Nice-to-Haves
- **Ablation Studies:** Quantifying the contribution of the recurrent connectome versus the background input module to the explained variance in neural dynamics.
- **Extended Criticality Analysis:** Including avalanche size distributions, branching parameters, and sensitivity analysis of the binarization threshold.
- **Biological Interpretation of Learned Parameters:** Analyzing whether the learned background input matrix \(W^{enc}\) or per-neuron time constants align with known biological pathways or cell-type classifications.
- **Comparison with Other Online Learning Methods:** Benchmarking against alternatives like e-prop or FORCE for convergence and final loss.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Weakness: "Background input uses target data during testing, so test is not autonomous."** *Removal Justification:* The paper explicitly describes the autonomous prediction phase in Figure S8 and its caption: "the model generated autonomous activity... it then used its own predicted output from the previous step as input." The critic misread the protocol.
- **Weakness: "No test loss (e.g., MSE) is reported."** *Removal Justification:* While reporting a test MSE would be a clearer quantitative metric, the paper provides a reasonable proxy by showing test functional connectivity correlation that outperforms the experimental baseline. Demanding a specific metric not standard in all neuroscience subfields is a "nice-to-have."
- **Strength: "The paper is well-written."** *Removal Justification:* This is a generic strength that applies to any competently written paper and does not identify something specific this paper does well.
- **Weakness: "Comparison between continuous weights and discrete synapse counts is superficial."** *Removal Justification:* The paper performs Q-Q plot analysis and binned count correlations (Fig. 4C, S9), which is a reasonable quantitative comparison for demonstrating distribution alignment. Requesting a specific statistical test (e.g., KS test) is a methodological preference, not a fundamental flaw.

## Suggestions
- **Revise the Criticality Analysis:** Re-analyze avalanche statistics using the model's *neuropil-level* readout activity to enable a valid comparison with the experimental neuropil data. Discuss the implications of any remaining differences.
- **Add an Ablation Experiment:** Include a experiment that ablates or fixes the recurrent weights to quantify the necessary contribution of the connectome-constrained recurrent dynamics to the model's performance and the emergent properties.
- **Clarify the Benchmark:** In the main text, clearly state that the direct BPTT comparison uses a low-rank approximation due to memory constraints, and discuss the implications for claiming optimization efficiency.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
