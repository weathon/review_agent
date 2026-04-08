=== CALIBRATION EXAMPLE 19 ===

# Final Consolidated Review
## Summary

This paper introduces an online learning framework (D-RTRL) to fit a connectome-constrained whole-brain *Drosophila* model (~138k neurons, ~15M synapses) to resting-state calcium imaging data, overcoming the memory bottleneck of BPTT by reducing memory scaling from sequence length to parameter count. The trained model reproduces neuropil-level dynamics and functional connectivity on held-out data, and the authors report that optimization spontaneously produces heavy-tailed synaptic weight distributions matching empirical connectome statistics and drives the network toward critical dynamics (power-law neuronal avalanches).

## Strengths

- **Scaling achievement with concrete evidence**: The paper successfully trains a 138k-neuron connectome-constrained model on a single GPU, a task that is computationally prohibitive with BPTT. Fig. 2A demonstrates that memory consumption grows with batch size but remains independent of sequence length, directly validating the core engineering claim.
- **Connectome-constrained readout provides genuine inductive bias**: Fig. 2C shows that the connectome-constrained readout (Eq. 4) converges to lower training loss than an unconstrained linear readout, demonstrating that the anatomical prior improves model fit beyond what a generic architecture achieves.
- **Non-trivial emergent alignment with connectome statistics**: The post-training weight distribution closely matches empirical synapse counts (Q-Q plot clustering along y=x; binned count correlation R=0.904 in Fig. S9A vs. R=0.479 for untrained, Fig. S9B). This alignment was not explicitly optimized for, making it a substantive finding rather than a trivial consequence of the loss function.

## Weaknesses

### Major:

- **Missing critical ablation: the background input term ($I^{enc}$) could dominate the dynamics**: Equation 3 introduces a learned, dense feedback pathway from 73 neuropils to all 138k neurons. This term acts as a low-rank recurrent driver capable of fitting dynamics independently of the connectome-constrained weights. The paper shows that connectome-constrained *readout* improves performance (Fig. 2C), but this tests the output mapping, not whether the *recurrent dynamics* depend on the connectome. Without an ablation removing or fixing $I^{enc}$, it is impossible to determine whether the emergent criticality and weight statistics arise from the connectome-constrained recurrent dynamics or from this global feedback pathway. This directly undermines the central claim that connectome structure shapes the observed dynamics.

- **Missing size-matched random connectivity control for emergent properties**: The paper attributes critical dynamics and heavy-tailed weight distributions to the connectome-constrained architecture, but provides no control to rule out that these are generic consequences of training any large recurrent network. The GRU baseline (256 units, Appendix F) is too small to serve this purpose—its failure to exhibit criticality could result from insufficient capacity rather than the absence of connectome structure. A 138k-neuron network with a random (Erdős–Rényi) connectivity mask of equivalent sparsity, trained with the same online objective, would directly test whether the emergent properties are specific to the biological topology. Without this control, the paper cannot distinguish "connectome structure produces criticality" from "training any large recurrent network on oscillatory data produces criticality"—a known ML phenomenon (RNNs trained on temporal tasks naturally approach spectral radius ≈ 1).

- **Criticality claims lack sensitivity analysis**: Avalanche detection (Appendix J) uses a fixed 3σ threshold for binarization. Criticality analysis is well-known to be sensitive to this threshold choice; different thresholds can shift the fitted exponent or even eliminate the power-law regime entirely. Without demonstrating robustness across threshold variations (e.g., 2σ–5σ), the criticality claim rests on a specific parameter choice that could be post-hoc.

### Minor:

- **Ambiguity in the training regime for the background input**: Equation 3 defines $I^{enc}_i(t)$ as a function of $FR^{neuropil}_k(t-1)$, but it is unclear whether this uses the *target* experimental values (teacher forcing) or the *model's own predictions* (autoregressive) during training. Fig. S8 describes a two-phase pipeline (warm-up then prediction), but the main text does not specify which regime governs Eq. 3 during each phase. This ambiguity affects interpretability: if teacher forcing is used, the model has privileged access to ground truth that would not be available during autonomous generation.

- **Moderate test-set functional connectivity correlation**: The training FC correlation of 0.998 drops to 0.556 on held-out data. While the model outperforms the direct train-test data similarity (0.474), a test correlation of 0.556 indicates that the model captures coarse statistical structure but misses substantial variation. Without analyzing *which* neuropils or functional subnetworks are well-captured versus poorly captured, it is unclear whether the model generalizes uniformly or only for specific brain regions.

### Trivial:

- The paper claims the model "spontaneously" develops critical dynamics, but the training target (resting-state calcium data) already exhibits criticality (α = 1.78, Fig. 5A). The word "spontaneously" could be read as implying the dynamics emerge without being driven by the data, when in fact the model is explicitly optimized to match data that is already critical. This is primarily a framing issue.

## Nice-to-Haves

- Comparison against simpler baselines (e.g., linear dynamical systems, VAR models) on held-out prediction to justify the complexity of a 138k-neuron model.
- Intervention on the spectral radius (e.g., constraining it away from 1 during training) to test whether spectral radius approaching 1 is *causally* responsible for the emergence of critical dynamics, rather than merely correlated.
- Cross-fly individual variability analysis: the data come from 18 brains, but no individual-level fitting or cross-fly generalization is shown.
- Comparison against other memory-efficient BPTT variants (gradient checkpointing, activation recomputation) on a smaller-scale connectome where both methods are feasible, to establish that D-RTRL is the right algorithmic choice rather than just a viable one.
- Wall-clock training time benchmark: memory savings are clear, but the computational time cost of D-RTRL at 138k scale is not reported.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: BPTT comparison only uses low-rank approximation, not the actual connectome** (Harsh Critic, Spark Finder): The paper explicitly states that BPTT is infeasible on the full connectome due to memory constraints—this is the entire motivation for the work. The low-rank comparison (Fig. 2B, Appendix C) is designed to validate that online learning achieves comparable *training quality* when both methods can run, while Fig. 2A demonstrates the memory advantage. Criticizing the absence of an infeasible comparison misunderstands the paper's argument.

- **Weakness: D-RTRL is not biologically plausible as a learning rule** (Harsh Critic): The paper does not claim biological plausibility for the optimization algorithm. It claims the *model dynamics* are mechanistic and constrained by biology. Demanding biological plausibility of the gradient-based learning rule is scope creep beyond the paper's stated goals.

- **Weakness: Reproducibility concerns about hyperparameters** (Neutral Reviewer): Specific learning rates, batch sizes, and gradient clipping are implementation details impractical to include in the main text; the supplementary material addresses stability across initializations (Appendix G). Removed per hard rules on reproducibility nitpicks.

- **Weakness: Formatting/clarity nitpicks about equation rendering**: Parser artifacts, not paper issues. Removed per hard rules.

## Novel Insights

The deepest unresolved tension in this paper is between two of its claims: (1) that connectome structure *drives* the emergent biological properties, and (2) that these properties emerge "spontaneously." If heavy-tailed weights and critical dynamics are generic optimization artifacts—produced by training *any* large recurrent network on temporal data—then the paper's significance shifts from "the connectome explains biological organization" to "fitting biological data under *any* connectivity constraint recovers biological organization." The distinction is consequential: in the former case, the connectome's specific topology is causally necessary; in the latter, the training objective alone suffices. The current experiments cannot distinguish these interpretations, and this is the paper's central empirical gap. A single well-designed control experiment—training a size-matched random network on the same data—could resolve this ambiguity and either elevate or fundamentally recast the paper's contribution.

## Suggestions

- Run an ablation removing the $I^{enc}$ background input term (or replacing it with fixed noise) and report the impact on training loss, FC correlation, weight distribution, and avalanche statistics. This is the single most important experiment to validate the connectome's role.
- Train a 138k-neuron model with a random connectivity mask of equivalent sparsity using the same D-RTRL framework and data, then compare avalanche exponents, weight distributions, and spectral radius evolution. This directly tests whether emergent properties are connectome-specific or optimization-generic.
- Report avalanche exponents across a range of binarization thresholds (e.g., 2σ, 3σ, 4σ, 5σ) to establish robustness of the criticality claim.

## Evaluation Axis Summaries

- **Novelty**: Moderate-to-high. The application of online gradient methods to connectome-constrained whole-brain fitting at 138k-neuron scale is novel and addresses a real computational bottleneck. However, the core algorithm (D-RTRL) is from prior work, and the scientific claims about emergent properties remain to be validated against proper controls.
- **Technical soundness**: Moderate. The methodological framework is well-specified and the scalability claim is convincingly demonstrated. However, the scientific claims about connectome-driven emergence of criticality and weight statistics lack the ablations and controls needed to rule out alternative explanations.
- **Empirical support**: Moderate for the engineering contribution (scalability, memory efficiency) and for basic dynamics reproduction. Weak for the stronger emergent-property claims, which lack the critical control experiments discussed above.
- **Significance**: High potential. If the emergent properties are indeed connectome-specific, this would be a striking result linking anatomical structure to dynamical regime. In its current form, the significance of the scientific findings is provisional pending proper controls.
- **Clarity**: Good. The paper is well-organized and the logical flow is clear, though Eq. 3's training regime needs clarification and the word "spontaneously" should be qualified.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
