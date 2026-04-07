=== CALIBRATION EXAMPLE 16 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title is clear and accurately reflects the paper's core contribution. The abstract succinctly summarizes the method (DelRec), its novelty (first SGL-based method for recurrent delays), and the key results (new SOTA on SSC and PS-MNIST, matching SOTA on SHD). The claims are specific and appear supported by the results presented later. However, the abstract does not mention any limitations or potential drawbacks of the method.

### Introduction & Motivation
The introduction effectively motivates the problem: challenges in training RSNNs, the biological plausibility and theoretical benefits of synaptic delays, and the identified gap—the lack of a practical SGL-based method for learning delays in recurrent connections. The related work is adequately surveyed, distinguishing between feedforward delay learning and the few existing recurrent delay methods (e.g., EventProp-based). The contributions are clearly stated: DelRec, a novel SGL method for recurrent delays using differentiable interpolation.

**Major Concern:** The claim that DelRec is "the first SGL-based method to train axonal or synaptic delays in recurrent spiking layers" requires careful qualification. The paper cites "Xu et al." (ASRC-SNN) as learning a single recurrent delay per layer using backpropagation. It is unclear whether Xu et al. uses SGL or a different gradient approximation. If Xu et al. does use SGL, the "first" claim might be overstated. The authors must clarify the exact relationship and novelty relative to this concurrent work.

### Method / Approach
The neuron model and the formulation of delayed recurrent connections (Eq. 7) are standard and clear. The core idea—using a differentiable interpolation (triangle function) with annealing width (σ) to handle non-integer delays during training—is well presented. Algorithm 1 provides a concrete implementation sketch.

**Critical Issues:**
1. **Notational Inconsistency:** There is a potential transpose error in the weight notation between the equations and Algorithm 1. In Eq. 7 and Eq. 10, the recurrent input to neuron *i* sums over *j* using weight *w_{ij}^{rec}* (presumably from neuron *j* to *i*). In Algorithm 1, line 21, the update uses *w_{ji}^{rec}* (from *i* to *j*). This must be corrected and clarified to ensure reproducibility.
2. **Memory and Computational Complexity:** The method requires maintaining a scheduling buffer of size *N × L*, where *L* scales with σ and the maximum delay. The computational cost of the interpolation and buffer management is not discussed, nor is its impact on training speed or memory compared to a vanilla RSNN.
3. **Annealing Schedule:** The annealing of σ (Eq. 14) is a heuristic. The paper does not ablate the sensitivity of results to the choice of σ_init, decay schedule, or the need for annealing at all. A comparison to a straight-through estimator for integer delays would strengthen the methodological contribution.
4. **Dynamic Buffer Size:** The buffer size *L* is computed based on the current maximum delay and σ (Eq. 13). The paper does not explain how the buffer is resized if delays grow beyond the initial allocation during training. Algorithm 1 implies a fixed *L* per forward pass, which could break if delays increase.

### Experiments & Results
The experimental evaluation is comprehensive, covering three standard SNN benchmarks. The SOTA comparisons on SSC and PS-MNIST are compelling, especially given that DelRec uses simple LIF neurons, outperforming models with more complex neuronal dynamics. The functional study on SHD provides valuable insights into the relative benefits of recurrent vs. feedforward delays.

**Critical Issues:**
1. **Statistical Significance:** Results on SSC are reported over 3 seeds. While standard deviations are small, 3 seeds are insufficient for robust statistical claims, especially given the small performance margins (e.g., 82.58% vs. 82.03%). More seeds (e.g., 5-10) are needed to confirm significance. Confidence intervals should be reported.
2. **Apples-to-Oranges Comparison in Ablation:** In the functional study (Fig. 3), the comparison between "feedforward delays" (DCLS, synaptic delays) and "recurrent delays" (DelRec, axonal delays) is problematic because the parameterization differs fundamentally. DCLS has *O(N²)* delay parameters for a fully connected layer, while DelRec uses *O(N)*. When controlling for total parameter count, the models necessarily have different numbers of neurons, making the comparison less direct. The authors should discuss this discrepancy and its implications.
3. **Missing Ablations:** There is no ablation study on the core components of DelRec itself (e.g., the effect of the annealing schedule σ, the impact of the triangle interpolation vs. other kernels, the benefit of learning delays vs. using fixed random delays). Such analyses are crucial for understanding what drives the performance gains.
4. **Energy Efficiency Claims:** The discussion of spiking rates (Fig. 3C, bottom) is interesting but preliminary. The claim that feedforward delays are "more energy-efficient" is not sufficiently supported; a more rigorous analysis of energy-accuracy trade-offs (e.g., accounting for the cost of implementing delays on hardware) is needed.
5. **Dataset Saturation and Overfitting:** The authors correctly note that SHD is saturated and used augmentations for their best models. However, for the SSC and PS-MNIST results, it is unclear if similar overfitting concerns were addressed (e.g., via proper validation splits). The paper should clarify the validation procedures for these datasets.

### Writing & Clarity
The paper is generally well-written. However, several sections need clarification:
- The weight notation inconsistency (Method section) must be fixed.
- The modified spread function *h_{σ,d,p}* (Eq. 15, Appendix) is used for SSC but not explained in the main text. This is a significant detail for reproducibility.
- The description of the circular buffer mechanism in Algorithm 1 could be more intuitive (e.g., a brief explanation of why it's circular).

### Limitations & Broader Impact
The paper lacks a dedicated limitations section. Key limitations that should be acknowledged include:
- The discrete-time formulation and its implications for event-based or neuromorphic deployment.
- The scalability of the method to larger networks and longer temporal sequences (both in terms of computation and memory).
- The heuristic nature of the σ annealing schedule and its potential sensitivity.
- The comparison with prior work on recurrent delays (Xu et al., Mészáros et al.) needs more precise differentiation.
- Societal impact is not discussed. While energy efficiency is a potential positive, a brief statement on broader impacts is expected by ICLR.

### Overall Assessment
This paper presents a timely and significant contribution to the field of spiking neural networks. DelRec is a novel and practical method for learning delays in recurrent connections using SGL, achieving new state-of-the-art results on two challenging benchmarks with simple neuron models. The core idea of differentiable interpolation with annealing is sound and well-executed. However, the paper has notable weaknesses: insufficient statistical validation, unclear novelty claim relative to concurrent work, a critical notation error in the algorithm, and missing ablations of the method's components. The contribution is strong enough for ICLR conditional on a revision that addresses these concerns, particularly clarifying the novelty, fixing the methodological description, providing more robust statistical evidence, and adding a thorough ablation study.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces "DelRec", a method to train axonal or synaptic delays in recurrent spiking neural networks (RSNNs) using surrogate gradient learning (SGL). The key technical contribution is a differentiable interpolation technique that allows for gradient-based optimization of continuous-valued delays, which are then rounded to integers for inference. The authors demonstrate that RSNNs with learned recurrent delays outperform models with only feedforward delays, achieving new state-of-the-art results on Spiking Speech Commands (SSC) and Permuted Sequential MNIST (PS-MNIST) datasets using simple Leaky-Integrate-and-Fire (LIF) neurons.

### Strengths
1.  **Clear Novelty and Technical Contribution:** The paper presents the first SGL-based method for training delays specifically in *recurrent* connections of SNNs. The proposed differentiable interpolation scheme (using a triangular spread function with a decreasing width parameter σ) is a well-motivated and technically sound solution to handling non-integer delays during training.
2.  **Strong and Thorough Empirical Results:** The method achieves new SOTA results on two challenging, non-saturated temporal benchmarks (SSC and PS-MNIST) using simple LIF neurons. The work includes a comprehensive ablation study on the SHD dataset, convincingly showing the advantages of recurrent delays over feedforward delays and fixed-delay baselines under parameter and sparsity constraints. The use of multiple seeds and proper train/validation/test splits (notably for SSC and SHD) adds rigor.
3.  **High Clarity and Reproducibility:** The paper is generally well-written, with a clear explanation of the method (aided by Algorithm 1 and Figure 2). The authors provide a public code repository and detail their hyperparameters and architectures in the appendix, facilitating reproducibility.

### Weaknesses
1.  **Insufficient Analysis of Combined Delay Types:** While the paper shows that recurrent delays alone outperform feedforward delays alone on small models (Fig. 3), the benefit of combining both types in larger models is asserted but not deeply analyzed. Table 1 shows the combined model underperforms the recurrent-only model on SSC, and the text notes "no advantage" in small configurations. A more detailed analysis of when and why these components complement or interfere with each other is needed.
2.  **Under-Supported Claims on Hardware Efficiency and Gradient Mitigation:** The claims that recurrent delays facilitate "efficient deployment on neuromorphic hardware" and "mitigate gradient challenges" (Fig. 1B) are not substantiated with evidence. No analysis of computational overhead, memory footprint on delay buffers, or actual gradient flow (e.g., gradient norm plots) is provided. These remain interesting but speculative motivations.
3.  **Limited Comparison to Non-Delay SOTA Methods:** The paper excellently compares to other delay-based methods. However, for ICLR's high bar, a more direct comparison to the strongest non-delay SOTA methods (e.g., SE-adLIF, SiLIF) is warranted, especially to disentangle the contribution of delays from other architectural choices. The justification for excluding some models from Table 1 (footnote 1) is reasonable but leaves a gap in the competitive landscape.

### Novelty & Significance
**Novelty:** High. The work is the first to successfully train recurrent delays in SNNs using the dominant SGL framework. The proposed interpolation and scheduling mechanism is a novel adaptation of ideas from feedforward delay learning to the recurrent setting.
**Significance:** Moderately High. The results demonstrate that learnable recurrent delays are a powerful and perhaps more effective alternative to complex neuron dynamics for temporal processing in SNNs. This could influence the design of future neuromorphic algorithms. However, the significance is somewhat tempered by the incremental nature of the advance over prior work on feedforward delays and the need for further validation of the hardware efficiency claims.

### Suggestions for Improvement
1.  **Provide a Deeper Analysis of Delay Interactions:** Conduct experiments to systematically analyze the interaction between feedforward and recurrent delays. For instance, visualize the learned delay distributions in different layers/tasks when both are active, or perform a Pareto analysis of accuracy vs. parameter count for the combined model to clarify its utility.
2.  **Substantiate Hardware and Gradient Claims:** To support the hardware efficiency claim, provide a basic analysis of the memory and computation requirements of the scheduling buffer relative to network size. To support the gradient propagation claim, include a plot of gradient norms or losses during training for networks with and without learned recurrent delays.
3.  **Strengthen the Comparative Benchmarking:** Include a dedicated experiment or expanded discussion section that directly compares DelRec (with LIF) against the top non-delay SOTA methods (e.g., SE-adLIF, SiLIF) on one of the main datasets. Analyze parameter counts, computational cost, and performance to clearly position the contribution of delays versus neuron complexity.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Direct comparison of axonal vs. synaptic delays on the same model and dataset.** The paper compares its axonal recurrent delays (one per neuron) with feedforward synaptic delays from DCLS (one per synapse). This is an unfair comparison of model capacity and expressivity. A fair ablation must test feedforward *axonal* delays and recurrent *synaptic* delays to isolate the benefit of recurrence vs. delay granularity.
2. **Quantitative analysis of gradient flow (vanishing/exploding) claims.** The paper claims recurrent delays act as "temporal skip connections" to mitigate gradient issues (Fig. 1B), but provides no empirical measurement of gradient norms or correlation lengths across time. Without this, the theoretical motivation is unsupported.
3. **Energy consumption benchmark on neuromorphic hardware or simulators.** The abstract claims "potential for substantially higher energy efficiency" and mentions hardware deployment, but the only energy-related metric is spike rate. For an SNN paper at ICLR, a rigorous analysis of energy-delay trade-offs on a hardware-compatible simulator (e.g., using synaptic operations) is necessary to substantiate efficiency claims.
4. **Comparison with the most recent and relevant method (Mészáros et al., 2025 EventProp).** The authors cite this work but do not compare DelRec's performance against it on the same datasets (SSC, PS-MNIST) under comparable settings. This is a critical omission for establishing state-of-the-art status.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of what temporal patterns the learned delays capture.** The paper lacks any analysis of the *distribution* or *function* of learned delays. Do they align with specific temporal features of the datasets (e.g., phoneme durations in SSC)? A histogram of final learned delays and their correlation with network dynamics is needed to show the method learns meaningful structure.
2. **Ablation on the contribution of the "progressive sharpening" schedule (`σ`).** The training uses a decreasing `σ` for the interpolation kernel. An ablation is needed to show this schedule is necessary for stable optimization and better than, e.g., a fixed small `σ` or direct rounding. Without this, the training method's design is not justified.
3. **Interaction analysis between learned delays and learned weights.** Do delays and weights co-adapt, or do delays simply compensate for poor weight learning? A sensitivity analysis (e.g., fixing delays after some epochs) would reveal if delays are primarily easing optimization or encoding task-specific timing.

### Visualizations & Case Studies
1. **Visualization of spike rasters with and without learned delays for a few input samples.** The current figures are conceptual. To show that delays create useful temporal dynamics, plot actual spike trains from the recurrent layer for sample inputs, comparing a vanilla RSNN, a fixed-delay RSNN, and DelRec. This would visually demonstrate "pattern generation" (Fig. 1A) in practice.
2. **Case study on a simple synthetic temporal task (e.g., spike sequence detection).** To convincingly demonstrate the expressivity of recurrent delays, apply DelRec to a canonical temporal computation task (e.g., detecting a specific inter-spike interval pattern). This would provide a clear, interpretable proof-of-concept beyond aggregate accuracy on complex datasets.

### Obvious Next Steps
1. **Test on actual neuromorphic hardware with programmable delays.** The paper's conclusion mentions "efficient deployment on neuromorphic hardware." A logical next step within the paper's scope would be to demonstrate a speed/energy benchmark on one such platform (e.g., SpiNNaker, Loihi) or a detailed simulation thereof, rather than just stating it as future work.
2. **Combine DelRec with adaptive neuron models (AdLIF, etc.).** The paper notes that combining delays with sophisticated neuron models could yield further gains. An obvious step is to integrate DelRec with a state-of-the-art adaptive neuron (e.g., SE-adLIF) and show if performance improves synergistically or if delays alone on LIFs are sufficient.
3. **Explore learned delays in deeper recurrent architectures (multi-layer RSNNs).** All experiments place recurrent delays only in the last hidden layer (Fig. 3A). A natural extension is to enable delays in all recurrent layers and study the effect on depth, gradient propagation, and performance.

# Final Consolidated Review
## Summary
This paper introduces DelRec, a method for training axonal or synaptic delays in recurrent spiking neural networks (RSNNs) using surrogate gradient learning. The core idea is a differentiable interpolation technique that allows continuous-valued delays to be optimized via backpropagation, which are then rounded to integers for inference. The authors demonstrate that RSNNs with learned recurrent delays achieve new state-of-the-art performance on the Spiking Speech Commands (SSC) and Permuted Sequential MNIST (PS-MNIST) datasets using simple leaky integrate-and-fire neurons.

## Strengths
- **Clear technical novelty:** DelRec is the first method to successfully train delays in recurrent connections of spiking networks using the dominant surrogate gradient learning framework, effectively adapting ideas from feedforward delay learning to the recurrent setting.
- **Strong empirical results:** The method sets new state-of-the-art accuracy on two challenging, non‑saturated temporal benchmarks (SSC and PS‑MNIST) while using simple neuron models, demonstrating that learnable delays can rival or surpass more complex neuron dynamics.
- **Thorough functional analysis:** The ablation study on the SHD dataset provides compelling evidence that recurrent delays outperform feedforward delays under parameter and sparsity constraints, and that even fixed random recurrent delays improve over a vanilla RSNN, highlighting the value of delay heterogeneity.

## Weaknesses
- **Clarity of novelty claim:** The paper claims to be “the first SGL‑based method” for recurrent delays, but cites concurrent work (Xu et al.) that also learns a recurrent delay via backpropagation. The relationship and distinctiveness of DelRec need clearer articulation to avoid overstatement.
- **Notational inconsistency in the algorithm:** There is a discrepancy between the weight notation in the equations (e.g., Eq. 7, 10) and Algorithm 1. In the equations the recurrent input to neuron *i* sums over *j* with weight *w_{ij}^{rec}*, while Algorithm 1 line 21 uses *w_{ji}^{rec}*. This ambiguity must be resolved for reproducibility.
- **Insufficient statistical validation:** Results on SSC are reported over only three seeds. Although standard deviations are small, the performance margins over competing methods are also small; more seeds (e.g., 5‑10) are required to robustly establish statistical significance.
- **Unfair comparison in the ablation study:** The functional study compares feedforward delays (which are synaptic, with O(N²) parameters) against recurrent delays (which are axonal, with O(N) parameters). Because the parameterizations differ fundamentally, the comparison does not isolate the benefit of recurrence versus delay granularity.
- **Missing ablation of method components:** The paper does not ablate key design choices of DelRec itself, such as the annealing schedule for the spread width σ, the choice of triangular interpolation versus other kernels, or the necessity of the progressive sharpening. Without these experiments, it is unclear what drives the optimization success.
- **Unsupported claims about hardware efficiency and gradient mitigation:** The abstract and introduction claim that recurrent delays enable “efficient deployment on neuromorphic hardware” and act as “temporal skip connections” to mitigate vanishing/exploding gradients, but no evidence is provided—no analysis of memory/computation overhead, energy‑delay trade‑offs, or measurements of gradient flow (e.g., gradient norms).

## Nice-to-Haves
- Visualization of learned delay distributions and their correlation with temporal features of the tasks (e.g., phoneme durations in SSC) to better interpret what the method captures.
- A case study on a simple synthetic temporal task (e.g., spike‑sequence detection) to provide an interpretable demonstration of the expressivity granted by recurrent delays.
- Direct comparison with the EventProp‑based recurrent delay method (Mészáros et al., 2025) on the same datasets to more completely situate DelRec within the landscape of delay‑learning techniques.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Dynamic buffer resizing:** A concern that the buffer size L might need to change if delays grow beyond the initial allocation during training. The paper indicates L is recomputed per epoch (Eq. 13), so this is an implementation detail rather than a fundamental flaw.
- **Explanation of modified spread function h_{σ,d,p}:** The function appears in the appendix and is used for SSC; its absence from the main text is a clarity issue but not a substantive weakness.
- **Request for theoretical proofs:** The paper is empirically focused; demanding theoretical guarantees is outside the expected scope for this type of contribution.
- **Criticism of missing related work:** As per instructions, we do not add or comment on missing references.

## Novel Insights
None beyond the paper’s own contributions.

## Suggestions
- Clarify the novelty claim relative to Xu et al. by explicitly stating whether that method uses surrogate gradients and how DelRec differs (e.g., per‑neuron delays vs. per‑layer, interpolation scheme).
- Correct the weight notation in Algorithm 1 to match the equations (use *w_{ij}^{rec}* for the connection from neuron *j* to *i*) and provide a brief textual explanation of the circular buffer mechanism.
- Run additional seeds (at least 5) for the SSC experiments and report confidence intervals to strengthen the statistical validity of the SOTA claim.
- In the functional study, add a comparison between feedforward axonal delays and recurrent synaptic delays to decouple the effects of recurrence from delay granularity.
- Include a basic ablation study on the DelRec components: test a fixed small σ, alternative interpolation kernels, and a schedule that anneals σ more rapidly to validate the design choices.

# Actual Human Scores
Individual reviewer scores: [0.0, 2.0, 2.0]
Average score: 1.3
Binary outcome: Reject
