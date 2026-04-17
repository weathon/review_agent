# Online Pseudo-Zeroth-Order Training of Neuromorphic Spiking Neural Networks

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Brain-inspired neuromorphic computing with spiking neural networks (SNNs) is a promising energy-efficient computational approach. However, successfully training deep SNNs in a more biologically plausible and neuromorphic-hardware-friendly way is still challenging. Most recent methods leverage spatial and temporal backpropagation (BP), not adhering to neuromorphic properties. Despite the efforts of some online training methods, tackling spatial credit assignments by alternatives with competitive performance as spatial BP remains a significant problem. In this work, we propose a novel method, online pseudo-zeroth-order (OPZO) training. Our method only requires a single forward propagation with noise injection and direct top-down signals for spatial credit assignment, avoiding spatial BP's problem of symmetric weights and separate phases for layer-by-layer forward-backward propagation. OPZO solves the large variance problem of zeroth-order methods by the pseudo-zeroth-order formulation and momentum feedback connections, while having more guarantees than random feedback. Combining online training, OPZO can pave paths to on-chip SNN training. Experiments on neuromorphic and static datasets with both fully connected and convolutional networks demonstrate the effectiveness of OPZO with competitive performance compared with spatial BP, as well as estimated low training costs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a method named OPZO for training Spiking Neural Networks (SNNs), which replaces spatial backpropagation with a single noised forward pass and momentum-updated feedback connections, aiming to achieve efficient and hardware-friendly credit assignment.

### Strengths
The OPZO method is novel and effective, skillfully addressing the high variance problem of traditional zeroth-order methods by retaining the first-order information of the loss function. Through extensive experiments, the paper demonstrates that OPZO's performance is significantly superior to other bio-plausible methods (like DFA) and can approach or even match the level of backpropagation (BP), offering a highly promising direction for efficient on-chip SNN training.

### Weaknesses
1.The provided ImageNet experiment is limited to fine-tuning a pre-trained network, which fails to demonstrate its performance on large-scale networks and datasets trained from scratch. For large-scale networks, OPZO might heavily rely on Local Learning to achieve performance comparable to BP.

2.Table 8 shows that OPZO requires an extremely large $\lambda$ for stable training. According to Equation 6, this implies that the feedback matrix M is updated very slowly. This raises the question: what would the performance be if a static M (i.e.,$\lambda=1$ after an initial phase) were used?

3.The paper lacks details on the initialization of the feedback matrix M, which weakens the method's reproducibility.

### Questions
1.OPZO and all compared methods are built upon the OTTT framework, which truncates the gradient flow across the time dimension. Is this a fundamental limitation for OPZO? Could OPZO be applied in a setting without truncating the temporal gradient flow (e.g., with full STBP)?

2.The paper mentions that the pseudo-zeroth-order method introduces bias but does not sufficiently analyze its impact on the final performance. Could you provide more theoretical analysis or experimental evidence on how this bias affects training stability and final accuracy?

3.Could you explain the rationale for removing the $1/\alpha$ factor in the practical implementation and discuss its impact on convergence?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes online pseudo-zeroth-order (OPZO) training for spiking neural networks. Existing zero-order optimization methods have large variances. To address this issue, the authors introduced a pseudo-zero-order formulation and used momentum feedback connections. Moreover, to apply this method to SNNs, the authors replace the gradient estimation in the OTTT method with the proposed method. To demonstrate the effectiveness of their method, the authors conducted experiments on static and neuromorphic datasets.

### Strengths
1. The experimental results demonstrate that the proposed method effectively reduces variance.
2. Experimental results demonstrate that the proposed method exhibits robustness against different gradient noise injections.

### Weaknesses
1. The novelty of introducing a momentum mechanism in error propagation is limited.
2. It does not seem that the proposed method takes into account the unique properties of SNNs.
3. The performance of the proposed method is not consistently superior. The proposed method performs worse than OTTT [1], especially on static datasets such as CIFAR and ImageNet. The authors did not even compare the performance of the proposed method against OTTT on ImageNet. Furthermore, there is a lack of comparisons to other SOTA online learning or acceleration methods for SNNs.


```
[1] Online Training Through Time for Spiking Neural Networks. NeurIPS. 2022.
```

### Questions
See Weakness.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This work proposes a pseudo-online zeroth-order training method that requires only a single forward pass, unlike existing approaches that rely on two forward passes. The authors achieve this by decoupling the loss function from the model and using the error signal (potentially a vector) computed at the final layer to update the model parameters. Additionally, a momentum feedback mechanism is introduced, which utilizes feedback signals from previous iterations to refine parameter updates. The authors claim that these modifications lead to improved performance and reduced variance in the training signal.

Experimentally, the proposed method demonstrates enhanced performance across multiple datasets and is compared against a range of baseline methods.

I am not an expert in zeroth-order optimization methods for SNN models, although I do have experience working with spiking neural networks. Therefore, I am assigning my score with low confidence.

### Strengths
The presentation of the paper is clear and well-organized.

This work addresses an important issue: the high variance problem in zeroth-order optimization methods.

The proposed decoupling approach is particularly interesting and provides a novel perspective on improving training stability and efficiency.

### Weaknesses
My main concern is that existing backpropagation or surrogate-gradient-based methods already achieve strong performance in similar settings. It is therefore not entirely clear why one should prefer the proposed zeroth-order approach over these established alternatives.

### Questions
Please see the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Online Pseudo-Zeroth-Order (OPZO) training, a novel method for supervised learning of Spiking Neural Networks (SNNs). OPZO aims to be more biologically plausible and hardware-friendly than standard backpropagation (BP). It achieves spatial credit assignment using only a single forward pass with injected noise and direct top-down feedback signals via momentum-based feedback connections, avoiding the biologically implausible weight symmetry and separate forward-backward phases of BP. Experiments on various datasets show OPZO achieves competitive accuracy compared to spatial BP while being estimated to have lower computational costs on neuromorphic hardware.

### Strengths
1.  The proposed momentum feedback connections successfully address the high variance problem typical of zeroth-order methods, enabling effective training where a basic single-point zeroth-order method fails. This is supported by theoretical analysis and empirical results showing gradient variances comparable to BP.

2.  The method is designed with neuromorphic computing constraints in mind. It requires only one forward pass, enables parallel updates, and aligns with three-factor Hebbian learning rules, potentially leading to lower memory and operation costs for on-chip training compared to BP.

### Weaknesses
1.  The biological plausibility claims lack specificity. The paper frequently states that OPZO is more biologically plausible but does not sufficiently elaborate on how specific components, like the momentum feedback connections, map to known biological mechanisms. A more detailed discussion comparing the algorithm's components (e.g., noise injection, top-down signals) to neuroscientific evidence would strengthen this claim.

2.  Theoretical grounding for momentum feedback could be stronger. While Proposition 4.1 analyzes variance reduction, the paper acknowledges that the momentum-based Jacobian estimation can introduce bias. Proposition 4.3 discusses conditions for providing a descent direction, but it would be strengthened by empirical analysis of this bias during training on complex tasks to show it does not hinder convergence.

3.  Integration with local learning is loosely defined. The method of combining OPZO with Local Learning (LL) and Intermediate Global Learning (IGL) is described at a high level. The paper states that for LL, "we assume the weight symmetry for propagating errors," but provides limited justification for this assumption in a biologically plausible or hardware-friendly context. A more detailed mechanism for how these global and local signals are integrated without violating the intended constraints would improve the methodology.

4. The experiment with a 9-layer convolutional network uses Local Learning and IGL. However, the results in Appendix D.5 show that pure OPZO without residual connections performs poorly on deeper networks. This suggests the core method may have scalability limitations, which are mitigated by auxiliary techniques. A clearer discussion of this dependency and more experiments probing the depth limitations of the core OPZO method would provide a more complete picture.

5.  Hardware Cost Analysis is Speculative. The analysis of computational costs (Table 5) is a theoretical estimate for potential neuromorphic hardware. The paper states that "no implementation on neuromorphic hardware is included due to our limited access," which is acknowledged as a limitation. Concrete measurements or more detailed simulations based on specific hardware architectures (e.g., Loihi, SpiNNaker) would make the claims about efficiency much stronger.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
