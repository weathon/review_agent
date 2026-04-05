=== CALIBRATION EXAMPLE 7 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the core contribution: learning delays in recurrent spiking layers. The abstract clearly states the method (DelRec), the technical mechanism (differentiable interpolation within SGL), and the empirical claims (SOTA on SSC and PS-MNIST, competitive on SHD). The claims are well-supported by the subsequent tables. One minor point: the abstract states this is the "first SGL-based method to train axonal or synaptic delays in recurrent spiking layers." While likely accurate given the cited literature (prior works use EventProp, softmax over discrete sets, or focus on feedforward only), a brief clarifying phrase distinguishing continuous gradient-based delay learning from discrete relaxation/EventProp methods would strengthen the novelty claim.

### Introduction & Motivation
The motivation is well-structured and aligns with current SNN research directions. The contrast between neuron-centric complexity (adaptive dynamics, multi-compartment) and structural/combinatorial complexity (delays) is clearly articulated. The reference to Izhikevich’s theoretical work on polychronization and the intuitive explanation of delays acting as temporal skip connections (Fig 1B) effectively grounds the method. However, the introduction could more explicitly address *why* SGL makes learning recurrent delays particularly challenging compared to weights. Specifically, recurrent delays create long-range, non-Markovian dependencies in the computational graph that standard BPTT struggles with. Framing this explicitly would better justify the need for the specific scheduling and relaxation strategy introduced later.

### Method / Approach (Section 2)
This is the most critical section, and several technical aspects require clarification for ICLR-level reproducibility and theoretical soundness:
* **Gradient Computation for Delay Parameters:** The core relaxation relies on the triangle kernel $h_{\sigma,d}(\tau)$ (Eq. 9-11). As $\sigma \to 0$, the kernel approaches a discrete assignment, and $\partial h / \partial d$ becomes discontinuous at integer boundaries. The paper does not discuss how gradient instability is handled near the end of training. Is a Straight-Through Estimator (STE) applied once $\sigma$ falls below a threshold? Are gradients clipped or modified to prevent divergence during fine-tuning? This is crucial, as unhandled discontinuities often cause delay parameters to oscillate or saturate.
* **BPTT & Circular Buffer Mechanics:** Algorithm 1 uses a circular buffer with modular indexing (`pointer`, `mod L`). For surrogate gradient backpropagation, PyTorch’s autograd must maintain the computational graph across time steps. Repeatedly overwriting and cyclically indexing the buffer can inadvertently detach gradients or create memory leaks if not implemented carefully. How is the backward pass constructed? Do you unroll the entire buffer graph, or use a custom `autograd.Function` to manage gradient flow across the modulo operations? Clarifying the memory complexity of BPTT with respect to sequence length $T$ and maximum delay is necessary, especially for long-horizon tasks like PS-MNIST.
* **Missing/Parser-Affected Equation:** Eq. 4 appears garbled or missing in the parsed text. If this is an artifact, it’s fine, but the LIF charging equation should be verifiable in the final PDF. Conceptually, the method’s extension to Eq. 7 is clear, but ensuring the full discrete-time update is formally presented will aid reproducibility.

### Experiments & Results (Section 3)
* **State-of-the-Art Claims (Table 1):** The results are strong and convincingly demonstrate that recurrent delays with simple LIF neurons can outperform more complex neuron models. Notably, `DelRec (only Rec. delays)` achieves **82.58%** on SSC with **0.37M** parameters, outperforming the combined Rec+Ff version (82.19%, 0.55M). This actually strongly supports your core thesis that recurrent delays are highly efficient, but the paper frames the combined version as superior. Please explicitly discuss this: does adding feedforward delays introduce harmful interference or over-parameterization on SSC? A brief ablation sentence on why the combined model underperforms on SSC but excels on SHD would add valuable insight.
* **Ablation & Functional Study (Section 3.2):** The comparison across delay configurations is well-motivated. However, distinguishing the benefit of *learning* delays from the benefit of *heterogeneous* delays remains partially confounded. While you include a fixed random delay baseline, a direct comparison between `fixed heterogeneous delays` (optimized via grid search or random sampling) vs `learned delays` would better isolate the contribution of gradient-based delay optimization. Additionally, reporting statistical significance (e.g., paired t-test or bootstrap confidence intervals) for the SHD ablation would strengthen the claim that learned recurrent delays consistently outperform alternatives under low-parameter constraints.
* **Dataset Saturation & Reporting:** The transparent discussion of SHD saturation and the decision to exclude it from Table 1 is scientifically honest. However, Section 3.2 heavily relies on SHD for the functional study. While acceptable for proof-of-concept ablation, consider acknowledging that SHD’s short sequences (~200ms binned) may not fully capture the long-range dependency advantages promised for PS-MNIST-style tasks.

### Writing & Clarity
The paper is generally well-organized and accessible. The methodological explanation (Fig 2 + Algorithm 1) is intuitive despite the lack of formal pseudocode for the backward pass. Section transitions are logical. Minor phrasing issues (e.g., "whether the high accuracies... stemmed from", "each if the phases") are easily fixable and do not impede understanding. The main clarity gap lies in Section 2.2, where the connection between the mathematical relaxation (spreading function) and the practical buffer implementation needs tighter bridging for readers unfamiliar with custom scheduling mechanisms.

### Limitations & Broader Impact
The paper acknowledges dataset saturation and discusses the energy/accuracy tradeoff between feedforward and recurrent delays. However, two key limitations are under-addressed:
1. **Maximum Delay Constraint & Buffer Size:** The scheduling buffer dimension depends on $\max(d) + \sigma$. What is the practical upper bound on trainable delays before memory/compute overhead becomes prohibitive? Do delays naturally saturate during training, or do they require clipping?
2. **Hardware Mapping:** The abstract mentions deployment on neuromorphic hardware with programmable delays. Real hardware typically enforces integer, bounded delays and often lacks fine-grained programmability per synapse/neuron. Discussing how the continuous-to-discrete rounding at inference time interacts with hardware quantization or delay resolution would strengthen the deployment claim.

---

### Overall Assessment
This paper presents a timely and methodologically sound contribution to SNN research: a practical, SGL-compatible method for learning recurrent axonal delays that achieves strong empirical results on standard temporal benchmarks. The core idea—relaxing integer delays via a decaying triangular scheduling kernel—is elegant, and the results convincingly show that recurrent delays can rival or surpass more complex neuron dynamics. However, for ICLR acceptance, the paper must clarify two critical technical points: (1) how gradients with respect to delay parameters are stabilized as $\sigma \to 0$ and during backward propagation through the circular scheduling buffer, and (2) a clear explanation of why combining feedforward and recurrent delays does not consistently yield gains (as evidenced by Table 1). Addressing these will solidify the reproducibility, theoretical grounding, and practical utility of the method. With these clarifications, the contribution stands as a strong addition to the SNN literature.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces DelRec, a surrogate gradient learning (SGL) method for training axonal and synaptic delays in recurrent spiking neural networks (RSNNs). By employing a differentiable, "future-oriented" spike-scheduling buffer with an annealed triangular interpolation, the technique enables end-to-end optimization of non-integer delays during training that are discretized at inference. The method achieves state-of-the-art accuracy on the SSC and PS-MNIST benchmarks using simple LIF neurons and demonstrates that learned recurrent delays improve parameter efficiency and temporal modeling compared to feedforward-only or fixed-delay baselines.

### Strengths
1. **Methodological Elegance & Practicality:** The scheduling buffer combined with annealed interpolation (Sec. 2.2, Eq. 8-12, Alg. 1) provides a computationally tractable solution to the integer-constraint problem in recurrent delay optimization. It avoids the combinatorial explosion of explicit delay queues while remaining compatible with standard autograd frameworks like PyTorch/SpikingJelly.
2. **Strong Empirical Results with Parameter Efficiency:** DelRec achieves new SOTA on SSC (82.58%) and PS-MNIST (96.21%) using only vanilla LIF neurons and ~0.37M–0.55M parameters (Table 1), outperforming prior methods that rely on complex adaptive neurons, attention, or multi-compartment dynamics. This cleanly isolates the contribution of trainable delays.
3. **Rigorous Functional & Ablation Analysis:** The SHD study (Sec. 3.2, Fig. 3) effectively isolates the impact of recurrent vs. feedforward delays, random fixed delays, and sparsity constraints. The finding that recurrent delays maintain performance better under severe parameter reduction provides strong evidence for their role in efficient temporal information reuse.
4. **High Reproducibility:** The authors provide a public code repository, detailed hyperparameters (Tables 4-5), explicit architecture configurations, dataset preprocessing steps, and a formal reproducibility statement, meeting ICLR's transparency expectations.

### Weaknesses
1. **Missing Backward Pass Formulation & Gradient Stability Analysis:** The paper thoroughly details the forward scheduling mechanism but omits explicit mathematical formulation of how gradients flow backward through the circular buffer and annealed interpolation (Sec. 2). Given SGL's sensitivity to gradient scaling, an analysis of delay-parameter gradient norms, learning sensitivity, or potential instability during early high-sigma phases would strengthen the technical foundation.
2. **Heuristic Annealing & Kernel Choice Without Ablation:** The exponential sigma decay (Eq. 14) and triangular interpolation kernel are introduced without theoretical justification or ablation. It is unclear whether alternative schedules (e.g., cosine, step) or kernels (e.g., Gaussian as used in DCLS) would yield different convergence behavior or final delay distributions.
3. **Unquantified Computational & Memory Overhead:** While the paper discusses eventual neuromorphic deployment, it omits runtime cost analysis. The scheduling buffer's memory footprint scales with sequence length and max delay+sigma, and its repeated updates during BPTT likely increase training FLOPs. Without memory/latency metrics, the claimed "energy efficiency" remains theoretical.
4. **Limited Context Against Strong Non-Spiking Baselines:** Table 1 exclusively compares SNN variants. For an ICLR submission, contextualizing the gains against standard recurrence mechanisms (e.g., GRUs, LSTMs) or modern sequence models on SSC/PS-MNIST would help assess whether recurrent delays close the gap between SNNs and conventional DL, or merely improve within the SNN paradigm.

### Novelty & Significance
**Novelty:** High within the SNN community. While trainable delays exist for feedforward SNNs (e.g., DCLS, SLAYER variants), extending differentiable delay learning to recurrent loops with a practical scheduling mechanism is a distinct and timely contribution. **Clarity:** The forward algorithm and intuition are well-explained, but the backward pass description is insufficient for standalone reproduction without inspecting the code. **Reproducibility:** Strong overall, supported by open code, explicit hyperparameters, and standard datasets, though the missing gradient equations slightly hinder exact methodological replication. **Significance:** Substantial for neuromorphic computing and bio-inspired AI, as it provides empirical validation for long-theorized benefits of axonal delays and offers a hardware-compatible training paradigm. Broader impact on general ML is currently moderated by the lack of non-spiking comparisons and absent efficiency metrics.

### Suggestions for Improvement
1. **Detail the Backward Pass & Gradient Dynamics:** Provide explicit equations for delay gradients through the scheduling buffer and surrogate functions. Include gradient norm trajectories or singular value analysis over time to quantitatively support the claim that recurrent delays mitigate vanishing/exploding gradients (Fig. 1B).
2. **Ablate Annealing & Interpolation Choices:** Test alternative sigma decay schedules and kernel shapes (e.g., Gaussian vs. triangular, linear vs. exponential decay) to demonstrate robustness. Report the final learned delay distributions to show whether the network converges to sparse integer values or relies on fractional delays.
3. **Quantify Computational Cost & Memory Footprint:** Add a section analyzing training/inference time, GPU memory usage, and FLOP counts relative to vanilla RSNNs. Clarify the memory complexity of the buffer (Sec. 2.2) and discuss trade-offs for long-sequence tasks, strengthening the neuromorphic deployment claims.
4. **Expand Baseline Context:** Include a comparison to strong non-spiking recurrent models (e.g., GRUs, LSTMs, or lightweight Transformers) on SSC and PS-MNIST, or discuss performance gaps relative to them. This will better position DelRec's contribution within the broader landscape of sequence modeling.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Report PS-MNIST results over ≥3 random seeds with mean ± std deviation, because a single-seed evaluation undermines the SOTA claim and fails ICLR's minimum statistical rigor.
2. Add a baseline comparison against a straight-through estimator (STE) for discrete delay optimization, because without it, the core contribution of differentiable interpolation vs. standard rounding remains unverified.
3. Include matched-parameter vanilla RSNN baselines (complex or simple neurons) on SSC and PS-MNIST without delays, because accuracy gains could simply stem from parameter reallocation rather than learned delays.
4. Measure gradient norm statistics across unrolled time steps for DelRec vs. vanilla RSNNs, because the central claim that recurrent delays mitigate vanishing/exploding gradients is purely theoretical without empirical training dynamics.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify the memory footprint and compute overhead of the scheduling buffer relative to sequence length, because the $O(N \cdot T)$ storage cost could completely negate the stated energy efficiency and hardware deployment claims on long sequences.
2. Analyze the statistical distribution of learned delays across layers and neurons, because without evidence of structured clustering or timescale alignment, delays may just act as unparameterized reservoir noise.
3. Report the accuracy drop between the interpolated training state and the post-training rounded integer delays, because a significant gap would invalidate the method's robustness and practical deployability on digital neuromorphic hardware.
4. Extend the feedforward vs. recurrent delay comparison to SSC and PS-MNIST, because restricting the key architectural trade-off analysis to the small SHD dataset prevents generalizing the "recurrent delays are superior" conclusion.

### Visualizations & Case Studies
1. Plot gradient flow heatmaps across temporal unrolling with and without delays to visually validate whether delayed connections actually function as temporal skip connections or merely redistribute gradient energy.
2. Map learned neuron delays against input signal temporal features (e.g., phoneme boundaries in SSC) to demonstrate the network exploits meaningful timescale structure rather than converging to arbitrary values.
3. Show buffer wrap-around behavior on long PS-MNIST sequences with artificially high delays to expose potential information truncation or scheduling collisions that the current formulation hides.

### Obvious Next Steps
1. Add full error bars (≥3 seeds) for all datasets immediately, as single-seed or incomplete reporting makes the SOTA and robustness claims non-actionable for peer review.
2. Conduct a standardized efficiency audit reporting FLOPs, peak memory, and total spike count on the main benchmarks, because repeated qualitative claims of energy efficiency and hardware readiness are meaningless without quantitative trade-off analysis against the primary baselines.
3. Include a synthetic long-range dependency benchmark (e.g., sequential MNIST copy task or delayed XOR) to prove the method genuinely captures long temporal horizons rather than just overfitting short, saturated datasets like SHD.

# Final Consolidated Review
## Summary
This paper introduces DelRec, a surrogate gradient learning method for optimizing axonal and synaptic delays in recurrent spiking neural networks. By employing a forward-scheduling buffer with an annealed triangular interpolation kernel, the approach relaxes integer constraints during training and rounds to discrete delays at inference. DelRec achieves new state-of-the-art accuracy on SSC and PS-MNIST using simple LIF neurons with highly competitive parameter counts, while functional ablations on SHD demonstrate that learned recurrent delays outperform feedforward-only and fixed-delay variants under severe parameter constraints, highlighting their role in efficient temporal information reuse.

## Strengths
- **Isolates the architectural contribution of delays:** By achieving top results on SSC and PS-MNIST with vanilla LIF neurons and ~0.37M–0.55M parameters, the paper clearly demonstrates that trainable recurrent delays can outperform methods relying on complex adaptive, multi-compartment, or attention-based neuron dynamics.
- **Elegant and scalable forward mechanism:** The scheduling-buffer approach with decaying triangular interpolation avoids the combinatorial explosion of explicit delay queues and integrates cleanly with PyTorch autograd, offering a practical alternative to exact event-based gradient methods that struggle with scalability.
- **Rigorous functional validation under constraints:** The SHD ablation systematically compares learned recurrent delays against feedforward delays, fixed random delays, and vanilla RSNNs. The clear demonstration that recurrent delays degrade less steeply under parameter reduction and sparsity penalties strongly supports their efficiency in temporal reuse.
- **High reproducibility and transparent benchmarking:** The authors provide open code, exhaustive hyperparameter tables, explicit architecture details, and properly acknowledge dataset saturation on SHD while using rigorous train/val/test splits for SSC, meeting high transparency standards.

## Weaknesses
- **Core motivation regarding gradient mitigation lacks empirical validation:** The introduction and Figure 1B strongly claim that recurrent delays mitigate vanishing/exploding gradients by acting as temporal skip connections that bridge distant timesteps. However, this remains entirely theoretical; the paper provides no gradient norm trajectories, singular value analysis, or effective training-depth metrics across unrolled sequences to empirically validate whether DelRec actually stabilizes BPTT compared to a vanilla RSNN.
- **Inference-time robustness to integer rounding is unreported:** The method relies on continuous relaxation during training and manual rounding at evaluation. While this is standard for differentiable relaxation, the paper does not quantify the accuracy gap (if any) between the interpolated model at $\sigma \approx 0$ and the discretized model. Reporting this drop is critical, as a significant performance degradation would undermine claims about reliable deployment on digital neuromorphic hardware.

## Nice-to-Haves
- Report the distribution of learned delays across layers and neurons to verify whether the network converges to structured, heterogeneous timescales or collapses to uniform values, which would strengthen the biological and functional claims.
- Provide PS-MNIST results over $\geq 3$ random seeds. While single-seed reporting follows prior SNN conventions on this benchmark, full error bars would align better with broader ICLR statistical expectations.
- Briefly discuss why combining feedforward and recurrent delays slightly underperforms the recurrent-only variant on SSC despite higher parameter counts, as this touches on potential interference or over-parameterization trade-offs.
- Clarify the memory/compute footprint of the scheduling buffer during inference on neuromorphic hardware vs. vanilla RSNNs, particularly regarding maximum delay bounds and circular buffer management, to ground the deployment claims.

## Novel Insights
The work effectively bridges theoretical neuroscience (polychronization, myelin plasticity) with practical deep learning by demonstrating that trainable axonal delays, rather than increasingly complex neuronal dynamics, serve as the critical inductive bias for temporal processing in SNNs. The scheduling-buffer relaxation offers a clean pathway to optimize delays under surrogate gradients without discrete softmax bottlenecks. Moreover, the finding that recurrent delays excel in parameter-constrained regimes while feedforward delays achieve comparable accuracy at lower firing rates reveals a fundamental accuracy-energy trade-off, suggesting that hybrid delay configurations could be dynamically tuned for hardware-specific efficiency targets.

## Suggestions
- Add a brief empirical analysis of gradient flow (e.g., gradient norm decay over temporal unrolling or layer-wise gradient variance) for DelRec vs. a vanilla RSNN to substantiate the temporal skip-connection hypothesis.
- Report the test accuracy of the continuous model at the final training epoch alongside the accuracy after integer rounding to confirm deployment robustness.
- Include a histogram or summary statistic of the learned delay parameters post-training to show whether the optimization discovers meaningful timescale diversity.

# Actual Human Scores
Individual reviewer scores: [0.0, 2.0, 2.0]
Average score: 1.3
Binary outcome: Reject
