# Biologically Plausible Online Hebbian Meta‐Learning: Two‐Timescale Local Rules for Spiking Neural Brain Interfaces

- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Brain-Computer Interfaces face neural signal instability and tight memory budgets in real-time implantable settings. We introduce an online intracortical SNN decoder trained with a temporally local, layer-local but spatially non-local three-factor rule with dual-timescale eligibility traces, avoiding backpropagation through time and requiring memory that is constant in sequence length. On two primate datasets, this Online SNN attains Pearson correlations of at least $R \geq 0.63$ on Zenodo Indy and $R \geq 0.81$ on MC Maze while converging faster in early training than a BPTT-trained SNN and reducing measured training memory by 63-86\% at sequence length $T=1000$. The learning rule combines synapse-specific dual-timescale Hebbian accumulators, error-modulated updates, and integer-friendly RMS homeostasis, and operates without unrolled computational graphs or adaptive optimizer state. Closed-loop simulations with synthetic neural populations demonstrate online supervised adaptation to neural disruptions and learning from scratch without offline calibration. Overall, the method provides a memory-efficient, continuously adaptive decoder that is temporally local and Hebbian but still relies on spatial backpropagation across layers, yielding a partially biologically plausible algorithm that is competitive with BPTT-trained SNN and Kalman baselines while trading some offline accuracy relative to LSTM/GRU decoders for online learning and deployment-oriented properties.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an online Spiking Neural Network (SNN) decoder for Brain-Computer Interfaces (BCIs) that uses local three-factor learning rules with dual-timescale eligibility traces. The approach avoids backpropagation through time (BPTT) while achieving satisfying performance in terms of online learning and closed-loop simulation. While I appreciate the engineering effort and experimental thoroughness in this work, I have concerns about its fit for ICLR. The contribution appears to be primarily `empirical and engineering-oriented` rather than offering the conceptual breakthroughs or theoretical depth.

### Strengths
1. The biological plausibility argument for the local three-factor learning strategy is compelling for neuromorphic implementation.
2. Memory complexity analysis clearly shows the advantages of the local three-factor learning strategy over BPTT-based methods.

### Weaknesses
1. In its current form, this reads more as a *well-executed technical report* documenting an implementation of a local three-factor learning strategy on SNNs and its performance on an online BCI task rather than a paper presenting novel research findings. I can not find any special design in the `Methodology` section to tackle the challenges raised by the EEG signal, such as `signal non-stationarity`, `high dimensionality and noise` and `cross-session/subject variability` as the authors mentioned in the `Introduction` section. The capability to resolve these issues seems originate from the local three-factor learning strategy itself rather than any specific methodological innovation. Therefore, I think the insights offered by this manuscript are somewhat limited and some of the claims appear overstated to some extent.

2. The paper presentation can be improved
   - The three bullet points in the contribution part of the `Introduction` section are not parallel.
   - Some figures are presented with low resolution and with a small font size, making it hard to read.
   - I suggest moving Table 1 to the supplementary material.

### Questions
1. Binning a continuous pulse sequence into a fixed time window will result in the loss of precise timing information. The chosen fixed time window can be very tricky. Could you elaborate on how you determine the time window size? Have you tried different time window sizes, and how does it affect the performance?
2. Could you provide an analysis of the generalization ability of the proposed local learning rule across different SNN architectures and network scales?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an online spiking neural network (SNN) decoder for brain-computer interfaces trained with local three-factor learning rules. The main claim is to achieve O(1) memory that avoids O(T) memory of BPTT, while maintaining competitive performance. Experiments on two primate datasets show comparable decoding accuracy with memory reduction and faster convergence, and closed-loop simulations with synthetic neural populations demonstrate adaptation to disruptions.

### Strengths
1. This paper unifies three-factor rule, two-timescale traces and updates, and meta-learning to build an online adaptive SNN decoder for efficient BCI, enabling lower training costs with competitive performance.

2. The paper conducts component-wise ablations to clarify the importance of each ingredient.

### Weaknesses
1. Poor writing and organization. The paper fails to describe the motivation clearly in the short introduction: why SNN is required and capable to solve the discussed challenges? There are logic gaps from BCI to SNN-based BCI to adaptive and online SNN-based BCI, and the paper directly jumps to the latter without answering the basic former question. The description of the method did not show advantages of SNNs, and experiments just assumed the context of SNN-based BCI and did not compare with state-of-the-art ANNs.
2. Limited novelty. The three-factor rule and meta-learning are existing techniques and have been widely used. It is unclear what is the novel perspective of this paper apart from combining these techniques.
3. Insufficient comparisons. The paper claims to compare with state-of-the-art methods but the baselines are relatively outdated. There is also no comparison to show advantages of SNNs considering any perspective.
4. Inconsistent claims. The contribution claims to provide “per-sample online adaptation to neural non-stationarities”, however, the method is not related to test-time adaptation, but only focuses on training given per-sample supervision.

### Questions
1. What is the justification for using SNNs? How can SNNs solve the stated problems, such as signal non-stationarity, high dimensionality, cross-subject variability, etc?
2. How is the performance of the proposed method compared with more advanced ANNs?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a fully online learning framework for Spiking Neural Networks. The core of the method lies in a local three-factor learning rule with dual-timescale eligibility traces. The authors reinterpret the eligibility traces as pure Hebbian accumulators, decoupling them from the error signal, rather than treating them as surrogates for BPTT gradients. Combined with techniques like RMS normalization and weight normalization, the method aims to achieve a biologically plausible, computationally efficient, and hardware-friendly online learning mechanism. Experiments conduct ablation studies on the Zenodo Indy brain-computer interface dataset to validate the effectiveness of its components.

### Strengths
1.The conceptual shift of decoupling eligibility traces from the error signal and redefining them as Hebbian accumulators is the most significant contribution of this paper. This perspective provides a clear and compelling new direction for designing non-BPTT online learning algorithms.

2.The entire learning framework is well-structured and clear, decomposing the learning process into two independent modules: "memory of history" (eligibility traces) and "guidance for learning" (error modulation). This modular design is not only conceptually easy to understand but also provides a solid foundation for future research and improvements.

### Weaknesses
1.Although the paper emphasizes biological plausibility, its core reliance on the global error signal δ(t) lacks a reasonable explanation in real biological neural systems.

2.The method introduces too many hyperparameters (e.g., decay coefficients for two timescales, two learning rates, mixing coefficient, normalization parameters, etc.). This high-dimensional hyperparameter space could lead to difficult tuning, and the method's performance might be highly sensitive to the choice of these parameters.

### Questions
1.The paper claims the method runs in O(1) memory, which is indeed a great advantage. However, the eligibility traces e_fast and e_slow themselves require storage. Could you please elaborate on the specific advantage in actual memory footprint (in bytes) compared to storing an unfolded BPTT computational graph? How do the memory costs of the two methods scale as the network size increases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a biologically plausible online Hebbian meta-learning rule for spiking neural networks (SNNs), aiming to replace backpropagation-through-time (BPTT) in brain-computer interface (BCI) decoders. The method combines a three-factor Hebbian rule (pre/post activity with an error signal), dual-timescale eligibility traces (fast/slow), and a simple meta-adaptation mechanism for learning rates. The authors claim their rule achieves comparable decoding accuracy to BPTT-trained SNNs while using O(1) memory and adapting online to signal non-stationarities. Experiments include results on two public primate datasets (MC Maze, Zenodo Indy) and synthetic closed-loop simulations under simulated disruptions (remapping, drift, dropout).

### Strengths
The paper touches on an important open problem, developing online, energy-efficient, biologically grounded learning for SNNs and BCIs, an area of genuine interest for both neuroscience-inspired AI and neuromorphic computing. The introduction clearly articulates why BPTT is biologically implausible and memory-inefficient, and motivates local three-factor rules as an alternative.

### Weaknesses
While the manuscript addresses an interesting and relevant problem, it suffers from numerous weaknesses related to unclear novelty, overstated claims, weak empirical validation, and poor presentation.

The proposed learning rule largely reassembles existing ideas from e-prop, SuperSpike, and meta-plastic SNNs, adding only minor heuristics such as a fixed fast/slow mixing coefficient and a hand-tuned learning-rate controller. The manuscript lacks a rigorous theoretical justification or empirical comparison that would clarify how the proposed method fundamentally differs from previous approaches. Moreover, the experimental section omits comparisons with established local learning methods such as e-prop or SuperSpike, and fails to reference other relevant recent work on local learning in spiking neural networks, such as OTTT [1], S-TLLR [2], SLTT [3], and TESS [4], which already provide efficient, biologically inspired alternatives to BPTT.

The manuscript also misrepresents the proposed approach as a “local learning rule.” While the update is local in time, it is not local in space, since it still relies on backpropagation across layers to compute error signals. Furthermore, the computational complexity analysis is superficial and somewhat misleading. The authors claim O(1) complexity, yet the per-timestep cost remains O(n²) in the number of neurons n. Although this cost is independent of sequence length, it can easily exceed that of BPTT for large models, and contrasts sharply with the linear O(n) complexity achieved by other recent local learning rules [1–4].

The so-called “meta-learning” component is another weak aspect. Rather than an adaptive or learned mechanism, it consists of a hand-designed heuristic that increases or decreases the learning rate depending on whether the loss improved in the previous window. There is no analysis explaining why this update scheme is beneficial, nor any demonstration that it contributes meaningfully to the model’s performance. Overall, the algorithm appears to be a collection of loosely motivated heuristics rather than a principled or biologically grounded model of meta-plasticity.

Several claims in the manuscript are also overstated. The authors repeatedly assert that their method achieves “comparable decoding to state-of-the-art” and exhibits “biological plausibility,” yet neither is convincingly supported. The performance is not competitive with modern recurrent or transformer-based decoders used in BCI research, and no experiments demonstrate the claimed hardware or energy efficiency. The discussion of memory efficiency is similarly unconvincing: Table 2 assumes the total memory footprint is three times the parameter count (weights plus two eligibility traces), but the algorithm also requires additional buffers for Hebbian updates (Eq. 1) and the momentum term G, effectively raising the memory requirement to roughly five times the number of parameters.
The ablation studies are descriptive rather than analytical, providing little insight into the model’s mechanisms. Most reported differences are small or statistically insignificant, suggesting that the proposed components may not substantially affect performance.
Finally, the manuscript’s presentation requires significant improvement. The paper is difficult to follow and not self-contained, with many key details relegated to the appendices. As a result, the main text cannot be fully understood without constant cross-referencing. Moreover, the structure of certain sections, for instance, Section 4.2, makes it unclear where one experiment ends and another begins, as the narrative blends multiple results into disconnected paragraphs. The lack of theoretical intuition or clear motivation behind several design choices further weakens the overall clarity and impact of the work.

[1] Xiao, M., Meng, Q., Zhang, Z., He, D. and Lin, Z., 2022. Online training through time for spiking neural networks. Advances in neural information processing systems, 35, pp.20717-20730.

[2] Apolinario, M.P.E. and Roy, K., 2025. S-TLLR: STDP-inspired Temporal Local Learning Rule for Spiking Neural Networks. Transactions on Machine Learning Research.

[3] Meng, Q., Xiao, M., Yan, S., Wang, Y., Lin, Z. and Luo, Z.Q., 2023. Towards memory-and time-efficient backpropagation for training spiking neural networks. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 6166-6176).

[4] Apolinario, M.P.E., Roy, K. and Frenkel, C., 2025. TESS: A Scalable Temporally and Spatially Local Learning Rule for Spiking Neural Networks. arXiv preprint arXiv:2502.01837.

### Questions
•	How does the proposed learning rule differ mathematically or conceptually from existing local learning methods such as e-prop, SuperSpike, and other three-factor rules?

•	What is the actual novel contribution of this work beyond combining previously known mechanisms like dual-timescale traces and heuristic meta-learning?

•	The paper does not include comparisons with recent local learning methods such as OTTT [1], S-TLLR [2], SLTT [3], and TESS [4]. Could the authors provide such comparisons or discuss how their approach relates to these methods?

•	The method still uses backpropagation across layers to compute error signals. In what sense can it be considered spatially local?

•	The claimed O(1) complexity appears inconsistent with the O(n²) per-timestep computational cost implied by the equations. Could the authors clarify the actual time and memory complexity relative to BPTT and other local update schemes?

•	The meta-learning component seems to be a simple heuristic adjusting learning rates based on loss improvement. How is this different in practice from using a standard adaptive optimizer such as Adam or RMSProp?

•	What is the quantitative contribution of the meta-learning mechanism to overall performance?

•	The claims of “comparable decoding to state-of-the-art” and “biological plausibility” are not well supported. Could the authors provide evidence or specific metrics that substantiate these claims?

•	The memory analysis in Table 2 appears incomplete, as additional buffers for Hebbian updates and momentum terms are not included. Could the authors provide a corrected breakdown of total memory usage?

•	The ablation studies show minimal differences across variants. How do these results support the importance of the proposed components?

### Soundness
2

### Presentation
1

### Contribution
1
