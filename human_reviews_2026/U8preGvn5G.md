# Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
Spiking Neural Networks (SNNs) have garnered significant attention as a central paradigm in neuromorphic computing, owing to their energy efficiency and biological plausibility. However, training deep SNNs has critically depended on explicit normalization schemes, leading to a trade-off between performance and biological realism. To resolve this conflict, we propose a normalization-free learning framework that incorporates lateral inhibition inspired by cortical circuits. Our framework replaces the traditional feedforward SNN layer with distinct excitatory (E) and inhibitory (I) neuronal populations that capture the key features of the cortical E-I interaction. The E-I circuit dynamically regulates neuronal activity through subtractive and divisive inhibition, which respectively control the excitability and gain of neurons. To stabilize end-to-end training of the biologically constrained SNNs, we propose two key techniques: E-I Init and E-I Prop. E-I Init is a dynamic parameter initialization scheme that balances excitatory and inhibitory inputs while performing gain control. E-I Prop decouples the backpropagation of the circuit from the forward pass, regulating gradient flow. Experiments across multiple datasets and network architectures demonstrate that our framework enables stable training of deep normalization-free SNNs with biological realism, achieving competitive performance. Therefore, our work not only provides a solution to training deep SNNs but also serves as a computational platform for further exploring the functions of E-I interaction in large-scale cortical computation. Code is available at https://github.com/vwOvOwv/DeepEISNN.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a biologically-inspired framework to enable the stable training of deep Spiking Neural Networks (SNNs) without explicit normalization layers. This is achieved by introducing a lateral inhibition circuit composed of distinct excitatory and inhibitory populations as a substitute for standard normalization. To this end, the authors have developed a comprehensive suite of techniques, including a principled initialization scheme (E-I Init) and a stabilization method for propagation (E-I Prop).

### Strengths
This work addresses a critical challenge in the field of neuromorphic computing that training deep SNNs while preserving biological plausibility with a novel and well-conceived solution. The proposed E-I dynamic regulation mechanism is compelling, and the thorough ablation studies convincingly demonstrate the framework's effectiveness and the necessity of its individual components.

### Weaknesses
1.The simplification of the inhibitory neuron model to a static, ReLU-like unit undermines the network's temporal dynamics. While computationally convenient, its validity is questionable, and it may introduce significant errors, especially when the number of timesteps (T) is small.

2.The 1/d gradient scaling for the inhibitory weights W_EI appears to be an ad-hoc choice. The paper lacks a theoretical analysis or extensive empirical validation explaining why this specific scaling factor is optimal, which may raise concerns about its generalizability.

3.The paper is missing a direct performance comparison with state-of-the-art (SOTA) SNNs that use Batch Normalization (BN) within the same architectures. This weakens the claim of achieving "competitive performance."

4.All experiments are conducted on static image classification tasks. To truly validate the framework's temporal processing capabilities, it should be evaluated on event-based datasets (e.g., DVS-Gesture) or other sequential tasks where SNNs are expected to excel.

5.As depicted in Figure 2, the model incorporates both subtractive and divisive inhibition. However, the manuscript (e.g., around Equation 14) fails to provide a clear mathematical formula describing how these two operations are combined. This ambiguity severely hinders the method's reproducibility.

### Questions
1.Does the proposed method alter the standard forward-pass rules of SNNs? Can the operations within the E-I circuit, particularly the divisive inhibition, be efficiently deployed on existing neuromorphic hardware platforms?

2.A core motivation of this work is the biological implausibility of BN. However, during inference, BN parameters can be folded into the weights of the preceding convolutional or fully-connected layers, meaning no standalone BN operation exists in the final network. Could you please elaborate on why BN is still considered a critical barrier to the biological plausibility of SNNs in this context?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a normalization-free biologically inspired method to train deep SNNs. The authors argue that most SNNs rely on non-biological normalization layers (BatchNorm, LayerNorm, etc.) to maintain activation stability, which limits biological plausibility. To overcome this, the authors propose an Excitatory-Inhibitory (E-I) architecture, including E-I init and E-I Prop, that enforces Dale’s law, meaning each neuron has a fixed excitatory or inhibitory role throughout training.

### Strengths
1. The aim to find a biologically plausible stabilization mechanism is well motivated. The integration of Dale’s law into deep SNNs is technically challenging and conceptually significant.
2. E-I Init and E-I Prop are technically well described and appear easy to integrate.
3. The paper could inspire further exploration into lateral inhibition and homeostatic balance as normalization-free learning tools.

### Weaknesses
1. Experiments are restricted to CIFAR-10/100; it remains unclear if the framework scales to large-scale or event-based datasets (e.g., ImageNet, DVS-Gesture).
2. The experiments do not include ablations isolating the impact of Dale’s-law enforcement itself. It is unclear whether improvements arise from E-I Prop (adaptive inhibition) or from Dale’s constraint.
3. The paper overclaims biological fidelity while omitting discussions of modern findings that contradict the core assumptions.
4. The training costs are not claimed. The overhead of maintaining separate excitatory and inhibitory populations and additional inhibitory weight updates is not quantified in runtime or memory terms.

### Questions
1. How does your method perform without enforcing Dale’s law, i.e., allowing all weights to freely take positive/negative values or other cases? 
2. Could you cite recent neuroscience literature justifying Dale’s law as a valid and necessary constraint for modern models? Otherwise using a seemingly outdated theory is not sufficient for this paper.
3. What are the computational overheads (training time, memory costs, etc.) compared to normalization-based baselines? The convergence is essential for training deep SNNs.
4. Have you tested on event-based neuromorphic datasets or other dataset to validate real-world relevance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
To resolve the performance-biological realism trade-off in deep Spiking Neural Networks, this work proposes a normalization-free framework incorporating lateral inhibition.

### Strengths
1. It presents a novel and practical solution to a core challenge in neuromorphic computing by replacing artificial normalization with bio-plausible E-I circuits, enabling both high performance and biological fidelity.

2. The work makes substantial contributions to brain-inspired AI by rigorously implementing Dale's law and cortical microcircuit principles, providing a valuable platform for computational neuroscience exploration.

3. The framework achieves competitive accuracy on benchmark datasets while demonstrating sophisticated neural regulation through emergent bimodal activity patterns, offering a biologically grounded alternative to conventional normalization.

### Weaknesses
The presentation of the core methodologies, particularly the derivation of E-I Init and the custom backward pass of E-I Prop, is excessively condensed. A more structured, step-by-step exposition is needed, as the current description lacks clarity and logical progression, consequently hindering both readability and reproducibility.

### Questions
1. Could you please provide further explanation of the statement: "In our discrete-time simulation, the time step ∆t = 1 is chosen to be on a similar scale as the time constant of excitatory neurons (e.g., τE = 2), which implies τI ≪ ∆t. Under this condition, the dynamics of inhibitory neurons can reach a steady state almost instantaneously within a single time step"?

2. Should the weight matrices W mentioned in the text be understood as fully connected? This point requires clarification.

3. Is the variable 'g' in Equation 13 a floating-point parameter? Does the formulation involve a direct multiplication between a floating-point parameter and a floating-point weight matrix?

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper revisits the important role of lateral inhibition in the cortical circuit and develops a normalization-free learning framework. The authors also developed a useful dynamic initialization scheme to ensure effective learning. The proposed framework supports the effective training of deep networks and shows competitive results on multiple benchmarks. Overall, I find this paper is quite interesting and the main idea is easy to follow.

### Strengths
- The work addresses the critical challenge of training deep SNNs without biologically implausible normalization schemes like Batch Normalization. The proposed framework, inspired by lateral inhibition in cortical circuits, is a significant step towards bridging the gap between high-performance machine learning and biological realism.

- The paper is well-written, and the core concepts are presented in a way that is easy to understand. The motivation for replacing standard normalization with a brain-inspired alternative is clearly articulated.

### Weaknesses
- While the performance has been validated on static image classification tasks, I think it would be better to further examine the performance on more extensive neuromorphic datasets for a broader impact.
- It would be great to discuss the potential deployment on existing neuromorphic hardware.
- Table 1 lacks statistical significance analysis.

### Questions
The authors incorporate Dale’s law into their models. I fully understand their consideration from a biological perspective. Beyond this, I wonder if the authors observed any other functional implications or accuracy gains by following this biological law?

### Soundness
4

### Presentation
4

### Contribution
3
