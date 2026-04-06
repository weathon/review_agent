=== CALIBRATION EXAMPLE 11 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly indicates the method (DelRec) and its purpose. The abstract is well-structured and makes strong, specific claims (new SOTA on SSC and PS-MNIST, matching SOTA on SHD) which are supported in the main text. The claim that DelRec is "the first SGL-based method to train axonal or synaptic delays in recurrent spiking layers" is plausible and sets a clear novelty. However, it would be helpful if the abstract explicitly qualified that the SOTA claims are for models using LIF neurons, as more complex neuron models (e.g., multi-compartment) report higher absolute numbers (as noted in the footnote).

### Introduction & Motivation
The introduction effectively motivates the problem: RSNNs suffer from gradient issues, and while recent work focuses on neuron dynamics, delays offer a complementary, biologically-inspired path. The gap in learning recurrent delays with SGL is clearly identified. The potential benefits (expressive dynamics, gradient flow improvement) are well-argued and supported by references and Figure 1. The contributions are clearly stated. A minor weakness is the comparison to Xu et al. (ASRC-SNN). The text says Xu et al. learn "a single recurrent delay parameter per layer," but it is not immediately clear how this fundamentally differs from DelRec's per-neuron delays. A more explicit contrast in the introduction (e.g., per-layer vs. per-neuron delays, and the interpolation method vs. their softmax selection from a fixed set) would strengthen the positioning.

### Method / Approach
The core method is clearly presented: extending the recurrent input equation (Eq. 7) to include per-neuron delays, using a differentiable triangular interpolation (`h_σ,d`) for non-integer delays during training, and annealing σ to zero. The use of a scheduling matrix and circular buffer (Algorithm 1) is sensible. The method appears reproducible.

**Major Concerns:**
1.  **Clarity and Correctness of Algorithm 1:** The algorithm is central but has confusing aspects. Line 4 initializes a single `spread` array, but `h_σ,d` depends on the individual delay `d_j` of the spiking neuron. Therefore, `spread` should be a matrix or be computed per neuron. As written, the algorithm is incorrect/ambiguous. This must be clarified.
2.  **Gradient Flow for Delays:** The method learns delays via the interpolation of scheduled spikes. However, the gradient path from the loss through the spike `S_j[t]` (a non-differentiable Heaviside) to the delay `d_j` relies entirely on the surrogate gradient of Θ. This is standard for SGL, but the paper does not discuss or analyze whether this provides effective gradients for the delay parameters specifically. A brief discussion or citation justifying this choice would be helpful.
3.  **Implementation Detail Omission:** The description of the pointer mechanism and buffer update (lines 19-30 in Algorithm 1) is complex. A clearer explanation or a small schematic in the appendix would aid understanding. The condition `if pointer + τ ≤ L - 1` seems to assume τ is positive; given Eq. 13, this holds, but it should be stated.

**Minor Points:**
- The annealing schedule for σ (Eq. 14) is a heuristic. A brief justification or an ablation on its necessity (e.g., vs. fixed small σ) would strengthen the method.
- The modified spread function for SSC (Eq. 15) with per-neuron `p_i` is introduced in the appendix without motivation in the main text. What problem does this solve? Does it affect the final integer rounding?

### Experiments & Results
The experimental design is comprehensive, using three standard SNN benchmarks. The results are impressive, showing clear improvements over strong baselines.

**Strengths:**
- SOTA claims on SSC and PS-MNIST are well-supported by Table 1, using a clean comparison against other LIF-derived models. The exclusion of more complex neuron models is justified in a footnote.
- The ablation study on SHD (Fig. 3) is excellent. It convincingly shows: a) the benefit of delays over no delays, b) that learned recurrent delays outperform fixed random ones, and c) the interesting trade-off between recurrent delays (higher accuracy) and feedforward delays (higher accuracy per spike, i.e., more energy-efficient).
- The study combines feedforward and recurrent delays, showing complementary benefits on SHD (Table 2).
- Reproducibility is strong, with code and detailed hyperparameters provided.

**Concerns and Questions:**
1.  **Statistical Significance and Seeds:** For PS-MNIST, results are reported for a single seed, while for SSC, three seeds are used. The authors note that previous SOTA on PS-MNIST also used one seed, which is a community standard but not rigorous. Running multiple seeds for PS-MNIST would bolster the claim. The standard deviations reported for SSC are reassuringly small.
2.  **Baseline Implementation Details:** The paper compares against DCLS for feedforward delays. It states (Appendix A.2.5) that for DCLS, they use linear interpolation instead of Gaussians. This is a modification to the original DCLS method. How does this choice affect the comparison? Is it necessary for fairness (e.g., to match DelRec's interpolation)? This should be justified.
3.  **Computational Cost:** The method introduces a scheduling buffer of size `N x L`. What is the typical memory and runtime overhead compared to a vanilla RSNN? This is relevant for the claim of "efficient deployment."
4.  **Delay Value Ranges:** The initialized and learned delay ranges are not analyzed. What typical delay values emerge? Are they biologically plausible (ms range)? Do they saturate bounds? A short analysis would provide insight into what the network learns.
5.  **Failure to Ablate σ Annealing:** The annealing of σ is a key component but is not ablated. Does performance degrade without it?

### Writing & Clarity
The paper is generally well-written. The figures effectively illustrate the concepts (though Fig. 1 has parsing artifacts). The method section, while technically sound, would benefit from a more intuitive walk-through of the scheduling process before presenting the algorithm. The distinction between "axonal" (per-neuron) and "synaptic" (per-connection) delays is mentioned but could be clarified earlier, as all experiments use axonal delays.

### Limitations & Broader Impact
The paper lacks a dedicated limitations section. Key limitations that should be acknowledged include:
- The computational and memory overhead of the buffer mechanism.
- The reliance on surrogate gradients for delay learning, which is not theoretically guaranteed.
- The evaluation is limited to classification tasks; generalization to other temporal tasks (e.g., prediction, robotics) is untested.
- The rounding of delays to integers at inference creates a train-test mismatch. The impact of this should be discussed (though the annealing of σ likely mitigates it).
- Broader impact is not discussed. Given the focus on energy-efficient SNNs, a brief positive statement about potential environmental benefits is appropriate, along with a standard note about potential misuse.

### Overall Assessment
This paper presents a novel and well-executed contribution: the first SGL-based method for learning recurrent delays in SNNs. The core idea is elegant, and the empirical results are strong, achieving new SOTA on two major temporal benchmarks using simple LIF neurons. The ablation study provides valuable insights into the functional role of delays. The main weakness is the lack of clarity in the algorithm description (specifically the handling of per-neuron spread functions), which must be corrected. Additionally, a more thorough analysis of computational cost, learned delay distributions, and the effect of the σ annealing schedule would strengthen the paper. Despite these issues, the central contribution is significant and likely to influence the field. With revisions addressing the major concerns, this paper meets the high standards of ICLR.

**Recommendation: Accept (Post Rebuttal / Revision).** The contribution is novel, the methodology is sound, and the results are compelling. The authors must clarify the algorithm and address the other major points raised above in a rebuttal and/or final revision.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces DelRec, a novel method for training axonal or synaptic delays in recurrent spiking neural networks (RSNNs) using surrogate gradient learning (SGL). The core innovation is a differentiable interpolation technique that allows optimizing non-integer delays during training, which are then rounded to integers for efficient inference. The authors demonstrate that learning recurrent delays significantly improves temporal processing, achieving new state-of-the-art results on the Spiking Speech Commands (SSC) and Permuted Sequential MNIST (PS-MNIST) datasets using simple Leaky-Integrate-and-Fire (LIF) neurons, and matches SOTA on the Spiking Heidelberg Digits (SHD) dataset.

### Strengths
1. **Novel and well-motivated contribution**: DelRec is the first method to successfully train delays in recurrent connections using SGL, addressing an underexplored area with strong biological and theoretical motivation (e.g., enabling richer dynamics and mitigating gradient issues). The method is clearly presented with algorithmic details and integration into existing frameworks (SpikingJelly).
2. **Strong empirical results**: The paper demonstrates new SOTA results on two challenging temporal benchmarks (SSC and PS-MNIST) using simple LIF neurons, outperforming more complex neuron models. The ablation studies on SHD convincingly show that recurrent delays outperform feedforward delays under low-parameter constraints and provide insights into trade-offs between accuracy and sparsity.
3. **Comprehensive evaluation and reproducibility**: The authors provide extensive experiments across multiple datasets, careful comparisons with prior work (including recent SOTA methods), and a thorough ablation study. The code is publicly available, and hyperparameters are detailed in the appendix, enhancing reproducibility.

### Weaknesses
1. **Clarity and presentation issues**: While the core method is explained, some parts of the paper are difficult to follow due to garbled text and formatting artifacts from the PDF parser (e.g., broken tables and figure captions, misaligned equations). Although these are parser issues, they impede readability. The description of the scheduling mechanism (Algorithm 1) and the modified spread function (Eq. 15) could be clarified.
2. **Limited comparison with closest related work**: The comparison with ASRC-SNN (Xu et al.) and EventProp (Mészáros et al.) is somewhat superficial. A deeper discussion of how DelRec differs in terms of scalability, gradient computation, and performance on long sequences would strengthen the contribution. The claim of being "first" should be more precisely qualified relative to these approaches.
3. **Insufficient analysis of limitations and overhead**: The computational and memory overhead of the scheduling buffer (especially for long sequences or large networks) is not discussed. The method's sensitivity to hyperparameters (e.g., initial spread σ) and its behavior on very long-range dependencies (beyond the tested datasets) remain unexplored.

### Novelty & Significance
**Novelty**: The paper introduces a novel method for learning delays in recurrent SNNs with SGL, a significant advance over prior work that focused on feedforward delays or used alternative optimization methods (e.g., EventProp). The differentiable interpolation approach for non-integer delays is clever and well-executed.
**Significance**: The results demonstrate that recurrent delays are critical for temporal processing and can achieve SOTA performance even with simple neuron models. This opens new directions for efficient SNN design and neuromorphic hardware deployment. The work meets ICLR's expectations for solid methodological innovation and rigorous empirical validation in an emerging field.

### Suggestions for Improvement
1. **Improve clarity and fix presentation issues**: Revise the manuscript to ensure all figures, tables, and equations are clearly legible. Consider adding a schematic diagram to illustrate the spike scheduling buffer and the interpolation process more intuitively. Expand the explanation of Algorithm 1 and the role of the circular buffer.
2. **Deepen the related work comparison**: Explicitly compare DelRec with ASRC-SNN and EventProp in terms of gradient approximations, scalability, and performance trade-offs. Discuss why SGL-based delay learning is more effective than event-based methods for the studied tasks.
3. **Discuss limitations and broader impact**: Address the computational overhead of the scheduling buffer, potential challenges in scaling to very long sequences, and sensitivity to hyperparameters. Include a brief discussion on the neuromorphic hardware implications (e.g., how programmable delays could be exploited) and potential applications beyond classification.
4. **Strengthen the empirical analysis**: Consider adding an experiment on a dataset with very long temporal dependencies (e.g., sequential pixel-level vision tasks) to further validate the method's ability to capture long-range interactions. Analyze the learned delay distributions to provide insights into what temporal patterns are captured.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Direct, controlled comparison between feedforward-only and recurrent-only delays on the main benchmarks (SSC, PS-MNIST).** The claim that recurrent delays outperform feedforward ones is central, but the paper only shows this on the smaller SHD dataset with simplified models. A comparison using the same architecture and parameter budget on SSC and PS-MNIST is essential to validate this claim for the reported SOTA.
2. **Ablation study on SSC and PS-MNIST isolating the contribution of learned recurrent delays.** The SOTA results are presented without ablating the key component (learned recurrent delays) against fixed or no delays on these datasets. Without this, the improvement cannot be confidently attributed to DelRec rather than other architectural choices.
3. **Inclusion of recent SOTA models with complex neurons in the main comparison table.** The paper claims SOTA using simple LIF neurons but excludes models with multi-compartment neurons, attention, or GRUs (which report higher scores on SSC). A direct comparison is necessary to properly contextualize the claimed SOTA and demonstrate the value of delays versus neuron complexity.
4. **Hardware or simulation-based efficiency evaluation.** The paper motivates the work with energy efficiency and deployment on neuromorphic hardware but provides no experiments measuring actual energy consumption, latency, or memory footprint compared to baselines. This undermines the practical contribution claim.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of the learned delay distributions and their correlation with task timescales.** Without examining what delay values are learned and whether they align with meaningful temporal patterns in the data (e.g., phoneme durations in SSC), it is unclear if the method captures interpretable structure or is just optimizing arbitrary numbers.
2. **Quantitative analysis of gradient flow to support the claim that delays mitigate vanishing/exploding gradients.** The paper hypothesizes that recurrent delays act as temporal skip connections but provides no evidence (e.g., gradient norm plots over time or layers) comparing training dynamics with and without delays.
3. **Mechanistic analysis of why recurrent delays outperform feedforward delays under low parameter counts.** The functional study on SHD shows a performance advantage but does not investigate the underlying reason (e.g., analysis of temporal receptive fields or information reuse). This limits understanding of the method's strengths.

### Visualizations & Case Studies
1. **Visualization of spike patterns and corresponding learned delays for sample inputs.** Showing raster plots of network activity with and without learned delays for a few input samples (e.g., specific speech commands) would concretely illustrate how delays shape temporal processing and improve classification.
2. **Training dynamics of the delay parameters.** Plotting how the delay values evolve over training epochs for a subset of neurons would reveal optimization behavior and whether delays converge to stable, interpretable values.

### Obvious Next Steps
1. **Combining DelRec with advanced neuron models (e.g., AdLIF, multi-compartment).** The paper shows SOTA with simple LIF but notes that higher performance could come from combining delays with complex neurons. This combination is a logical next step that should have been explored to push performance further and validate the method's generality.
2. **Comparison between axonal and synaptic delay learning.** The method is stated to support synaptic delays (one per synapse) but all experiments use axonal delays (one per neuron). A comparison of these two settings on a benchmark would provide insight into the flexibility and benefits of fine-grained delay tuning.

# Final Consolidated Review
## Summary
This paper introduces DelRec, the first method to train axonal or synaptic delays in recurrent spiking neural networks using surrogate gradient learning. The core innovation is a differentiable interpolation technique that allows optimizing non-integer delays during training, which are rounded for efficient inference. DelRec achieves new state-of-the-art results on the Spiking Speech Commands (SSC) and Permuted Sequential MNIST (PS-MNIST) datasets using simple LIF neurons, and demonstrates through ablation that recurrent delays offer advantages over feedforward delays for temporal processing.

## Strengths
- **Novel methodological contribution**: DelRec is the first successful integration of trainable delays in recurrent connections with surrogate gradient learning, addressing a significant gap. The method is elegant, using a differentiable interpolation and annealing schedule to handle non-integer delays, and is implemented within a standard framework (SpikingJelly).
- **Strong and comprehensive empirical validation**: The method sets new state-of-the-art performance on two challenging, non-saturated temporal benchmarks (SSC, PS-MNIST) using only simple LIF neurons, outperforming models with more complex neuronal dynamics. A thorough functional study on SHD provides convincing evidence that learned recurrent delays outperform fixed delays and offer distinct benefits compared to feedforward delays, particularly under low-parameter constraints.

## Weaknesses
- **Ambiguity in the core algorithm description**: Algorithm 1 is central to the method but is presented ambiguously. It initializes a single `spread` array, yet the spread function `h_σ,d` depends on each neuron's specific delay `d_j`. This makes the algorithm as written incorrect or incomplete, hindering reproducibility and understanding.
- **Incomplete ablation on main benchmarks**: The paper's central claim—that recurrent delays are more powerful than feedforward delays—is supported by a functional study on the smaller SHD dataset with simplified models. A direct, controlled comparison between recurrent-only and feedforward-only delays using the same architectures and budgets on the primary SSC and PS-MNIST benchmarks is missing, weakening support for this key claim.

## Nice-to-Haves
- Analysis of the learned delay distributions to see if they correspond to interpretable temporal structures in the data (e.g., phoneme durations).
- Visualization case studies showing how spike patterns and classification change with learned delays for sample inputs.
- A brief discussion of the memory/runtime overhead of the scheduling buffer mechanism, though this is not a core evaluation metric for the paper.

## Novel Insights
The paper provides compelling evidence that optimizing delays in recurrent connections is more effective for temporal processing than optimizing feedforward delays, especially when network capacity is limited. This is a novel and valuable insight for the SNN community, suggesting that the recurrent topology is a crucial site for learning temporal skip connections. The functional study further reveals an interesting trade-off: while recurrent delays achieve higher accuracy, feedforward delays can achieve comparable performance with lower firing rates, pointing to a potential accuracy-efficiency tradeoff for hardware deployment.

## Suggestions
- Revise Algorithm 1 and the surrounding text to unambiguously describe how the per-neuron spread is computed and applied during the buffer update. A clarifying schematic or pseudo-code correction is essential.
- Perform an ablation on SSC and/or PS-MNIST comparing a DelRec model (recurrent delays only) against a strong feedforward-delay-only baseline (e.g., DCLS) with matched parameter counts to solidify the claim about the superiority of recurrent delays.

# Actual Human Scores
Individual reviewer scores: [0.0, 2.0, 2.0]
Average score: 1.3
Binary outcome: Reject
