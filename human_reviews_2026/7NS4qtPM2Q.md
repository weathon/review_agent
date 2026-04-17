# Unveiling the Scaling Law of PINNs under Non-Euclidean Geometry

- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Physics-informed neural networks (PINNs) have emerged as a powerful framework for solving partial differential equations (PDEs) by embedding physical laws into the training process. In theory, PINNs admit optimal polynomial convergence rates in approximation and generalization. However, these results rely on the unrealistic assumption of global optimization, which is intractable in practice. Consequently, scaling PINNs to large architectures remains a major challenge, as network width increases and thereby the condition number of the underlying optimization problem grows rapidly, making training increasingly difficult and creating a fundamental bottleneck.
In this work, inspired by the MUON framework, we propose a descent strategy that adapts to the geometry of the optimization landscape. The new optimization algorithm does not degrade as the network size increases. As a result, we establish—for the first time—a scaling law for PINNs that predicts how performance improves systematically with model size. Using this framework, we successfully trained a PINN with more than 1,000 neurons per layer, surpassing the previous state-of-the-art limit of 200–400. This scaling perspective bridges the gap between theoretical guarantees and practical optimization, opening the door to pushing PINNs toward machine precision at unprecedented scales.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The primary contribution of the paper is to demonstrate that a specific optimization strategy, Scaled MUON, enables the training of much wider PINNs, revealing a new scaling law. The paper uses comparisons and a variety of PDEs to support this claim, though its architectural focus is narrow.

### Strengths
The primary contribution of the paper is to demonstrate that a specific optimization strategy, Scaled MUON, enables the training of much wider PINNs, revealing a new scaling law. The paper uses comparisons and a variety of PDEs to support this claim. The experiments repeatedly compare the proposed Scaled MUON optimizer against Adam and SOAP. Figures 1 and 4 show that standard optimizers like Adam and SOAP fail to maintain performance as network width increases, whereas Scaled MUON succeeds in training networks up to a width of 1024, surpassing the previous limit of 200–400 neurons.

### Weaknesses
1. Lack of comparison with other optimizers.

The experiments repeatedly compare the proposed Scaled MUON optimizer against Adam and SOAP. But other good optimizers should also be compared. See a survey on PINN optimizer: Optimizing the optimizer for physics-informed neural networks and Kolmogorov-Arnold networks.

2. Test on only very simple PDEs.

This paper only experimented with simple, low-dimensional PDEs, where making the network wide does not make much sense given the low input dimension. I suggest the authors try higher-dimensional cases, like 1000 dimensions. Authors can find out these interesting cases in this paper: Tackling the Curse of Dimensionality with Physics-Informed Neural Networks. In these high-dimensional cases, it is indispensable for the PINN network to be large; otherwise, information is compressed, where a wide PINN net makes more sense.

3. This paper fails to consider various PINN network structures.

Other approaches, such as improved network architecture and activation functions, can also mitigate the problem identified in the paper. But such a discussion is lacking.

### Questions
See weakness.

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
5

### Summary
The manuscript proposes a new scaling for Muon, which results in new optimizer for neural network training. The scaled version of Muon is tested with a physics-informed loss on a variety of partial differential equations. Additionally, an approximation result with restricted spectral norms of the trainsition matrices is presented.

### Strengths
+ The manuscript addresses the timely and important problem of improving optimization in physics-informed machine learning. 
+ The proposed method is scalable to architectures of considerable size. 
+ The heuristic about using a different scaling of the hidde-layer gradients could become useful for PINN training.

### Weaknesses
+ The manuscript is not very well organized, in particular: 
	+ Overall, the manuscript reads unfinished with a variety of typos, not careful writing. 
	+ Figure 1 and 2 already contain results although they are presented in Section 2 discussing preliminaries. 
	+ It is not clear to me why the paragraph **Difference with MUON.** comparing the scaled to the unscaled Muon is not in the description of the method given in Section 3. 
+ Motivation for the choice of scaling: The choice of matrix norms is not motivated at all in the section introducing the methodology. 
+ The computational cost of the method is nowhere mentioned. Further a description of how $\textup{signm}(\tilde G)$ is computed is missing. 
+ Comparison to (approximate) second-order optimizers: A comparison of the method to approximate second-order methods is missing: 
	+ From my understanding the scaled Muon optimizer has complexity cubic in the layer width of the network. This seems to be on the same order with the Kronecker-factored approximation (KFAC) of energy natural gradients or Gauss-Newton method proposed by Dangel et al. (2024). However, despite these recent advances, natural gradients are described as being hopeless due to their cubic implementation cost, which does not describe the state of research on optimization PINNs accurately. 
	+ Next to the comparison in computational complexity, a comparison to second order optimizers in terms of accuracy is completely missing. However, such a comparison is required for publication as it is the accuracy that a given training method for PINNs is able to achieve that is important, not the size of the trained network. Here, a comparison to KFAC as well as efficient implementation based on the Woodbury matrix identity as described by Dangel et al. (2025) should be added.
+ Relation of theoretical results to practical algorithm: I am failing to see the immediate connection of Theorem 1 and Theorem 2 to the proposed scaled Muon optimizer. 
+ Overly big claims: 
	+ The manuscript has the title *UNVEILING THE SCALING LAW OF PINNS UNDER NON-EUCLIDEAN GEOMETRY* and states that it *This work establishes, for the first time, a scaling law for physics-informed neural networks (PINNs).* From my understanding, not a scaling law is established, but fitted to experimental data produced from one specific algorithm. Hence, I don't think the claim that it reveals an actual scaling law inherent to PINNs. Further, from my understand there is no such thing as **the** scaling law of PINNs, but rather a scaling law for every optimizer and PDE. Therefore, I find the title slightly misleading. 
	+ Claim of empirical performance: *Using this framework, we successfully trained a PINN with more than 1,000 neurons per layer, surpassing the previous state-of-the-art limit of 200–400.* I can not verify this claim. As described above, from my understanding, the computational complexity of the proposed method is not smaller than of certain approximate second-order methods. Note Dangel et al. (2024) use KFAC to train a PINN of architecture $100 \to 768 \to 768 \to 512 \to 512 \to 1$ with $1.3\cdot10^6$ parameters and Woodbury as well as tools from randomized numerical linear algebra have achieved comparable speedups, see Dangel et al. (2025). 
	+ In the abstract, it is stated that *Using this framework, we successfully trained a PINN with more than 1,000 neurons per layer, surpassing the previous state-of-the-art limit of 200–400.* I do not know what is meant by this. First, Ibelieve that the notion of "successful" training can not be defined for a PINN. Much rather, the performance of different optimizers can be compared. Further, as already mentioned above, it is not about the size of the networks being trained, but about the accuracy achieved in doing so.

### Questions
+ Can you give an intuition for the choice of the matrix norms? 
+ Can you add an explanation how the explicit form of the update in line 283 arises from the chosen matrix norm? 
+ Can you comment on the computational complexity of the proposed method? In particular, can you comment how this relates to approximate second-order methods like KFAC? 
+ Can you comment on the accuracy of the proposed method with state-of-the-art second optimizers (KFAC, E-NGD with woodbury, SOAP, SHAMPOO, S-Broyden)? 
+ Can you comment on the number of parameters of the networks used in your experiments? In particular the depth is not given. 
+ Can you elaborate why you put the emphasis of the work on the scaling law rather than the design of an optimizer? 
+ Can you elaborate what you mean by *we successfully trained a PINN with more than 1,000 neurons*? What do you regard as successful training? 
+ Can you elaborate on the implications of Theorem 1 and 2 to the proposed scaled version of the Muon optimizer? 
+ Regarding the Proof of Theorem: How do you control the $N$-dependence of magnitude of the coefficients $\lambda_i(f)$ of the splines? 
+ In line 428, what is meant by *nuclear norm*? Nuclear norm usually refers to $l^1$ norm of the singular values of a matrix. In line 427 it is applied to a gradient. 
+ Can you comment why you don't choose a scaling in the matrix norm that further reduced the bound given in (8)? 

**References**
1. Kronecker-Factored Approximate Curvature for Physics-Informed Neural Networks, Felix Dangel, Johannes Müller, Marius Zeinhofer, NeurIPS 2024
2. Improving Energy Natural Gradient Descent through Woodbury, Momentum, and Randomization, Andrés Guzmán-Cordero, Felix Dangel, Gil Goldshlager, Marius Zeinhofer, 2025

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of training large-scale Physics-Informed Neural Networks (PINNs), which suffer from ill-conditioning as network width grows. Inspired by the MUON framework, it proposes a geometry-adaptive descent algorithm that maintains training efficiency and stability even in very wide networks, establishing a scaling law that predicts improved performance with model size. Experimentally, the method validates successful training of PINNs with significantly wider architectures, supporting a proposed scaling law

### Strengths
The paper introduces a novel optimization strategy tailored to the PINNs’ optimization landscape, enabling stable training at unprecedented network scales. It demonstrates a significant practical advancement by successfully training much wider PINNs than previously possible, while providing theoretical insights that connect model size, conditioning, and optimizer behavior in a unified framework.

### Weaknesses
* The paper correctly identifies ill-conditioning as a core challenge when scaling PINNs. However, the root causes of ill-conditioning in PINNs are multifaceted and often stem from the physical properties of the PDE (e.g., stiffness from higher-order derivatives or strong boundary conditions), the loss function structure (notably scale mismatch between physics and data terms), and gradient issues such as vanishing or exploding gradients.
While the scaled MUON approach addresses condition number growth related to network width, it is unclear whether it effectively mitigates ill-conditioning caused by physically stiff PDEs. There is a risk that the method targets a narrower subset of ill-conditioning causes, potentially limiting its general applicability.

* PINNs commonly integrate multiple physical quantities with widely varying scales (e.g., a state variable u(x,t) vs. its second derivative ∂²u/∂x², which may differ by orders of magnitude). Norm-based scaling updates as proposed may inadvertently suppress or amplify gradients unevenly, harming learning stability especially for critical higher-order terms. This raises concerns about the generalizability of scaled MUON to complex, multi-physics PINN problems.

* The conservative update strategy for biases and last-layer weights, while potentially stabilizing, may reduce the expressive power of the network. Since the accuracy of the PDE residual often critically depends on the last layer's ability to represent fine solution details, this approach could create bottlenecks, especially in modeling subtle or high-frequency solution features.

* The need to compute or approximate operator norms (e.g., spectral norm or RMS norm) for each parameter group at every training step introduces significant computational overhead. When combined with the already expensive automatic differentiation for high-order derivatives in PINNs, this may substantially slow down training, impacting practicality on large-scale problems.

* Scaled MUON’s structured update rules reduce the flexibility and interpretability of hyperparameter tuning compared to common optimizers like Adam or L-BFGS. This is particularly critical in physics-informed contexts where careful loss weighting and sensitivity balancing are essential. The optimizer’s design may complicate such domain-specific adjustments.

* Experiments rely solely on the PirateNet architecture, which helps isolate width-scaling effects but limits broader claims. Validation on additional PINN backbones such as standard MLPs or recent architectures (e.g., PINNsFormer, FFNs) is necessary to demonstrate robustness and applicability beyond the chosen design.

* Theorem 2’s assumption of bounded second derivatives of the loss function is generally violated in PINNs, due to unbounded differential operators. This raises questions about the practical stability guarantees promised by the theory.

* While spectral norm control theoretically bounds gradient norms, it may restrict the network’s ability to learn weak or discontinuous solutions common in real PDEs. Existing optimizers like Adam already offer adaptive gradient control, and it remains unclear whether strict spectral norm constraints universally improve training, particularly for PDEs with singularities or weak solutions.

* Experimental and Evaluation Concerns: 


  - Narrow Definition of “Performance Improvement”: The scaling law focuses solely on loss reduction, without assessing physical solution accuracy metrics such as L² error, PDE residuals, or boundary/initial condition satisfaction. Demonstrating that loss improvements translate to better physical fidelity is critical.



  - Lack of Quantitative Stability Analysis: The claim of “stable feature learning” is not supported by concrete stability metrics (e.g., gradient norms, Hessian condition numbers). Comparative stability curves with MUON and baselines would clarify this benefit.



  - Potential Unfair Baseline Comparisons: Details on baseline optimizer tuning are insufficient. Common methods like Adam or SOAP benefit from learning rate schedules, decay, and warmup strategies; ensuring comparable tuning across all methods is essential for fair comparisons.



  -  Missing Quantitative Ill-conditioning Metrics: Key motivation is ill-conditioning growth with network width, but experiments lack direct measurement of condition numbers, gradient flow behavior, or spectral decay. Loss reduction alone is insufficient to confirm mitigation of ill-conditioning.



  - No Verification of Spectral Condition Satisfaction: Though the paper cites Yang et al. (2023) for spectral condition satisfaction, it provides no numerical or visual verification. Empirical confirmation of this claim would strengthen the argument.

### Questions
* Can the authors provide physical solution accuracy metrics (L² error, PDE residuals, BC/IC satisfaction) to complement loss-based performance measures?  

* Can stability be quantitatively demonstrated with gradient norms, Hessian condition numbers, or training stability curves?  

* How were baselines tuned? Were learning rates, decay, and warmup schedules matched fairly across methods?  

* Could the authors provide quantitative analysis of ill-conditioning, such as condition numbers, spectral decay, or gradient flow behavior? 
 
* Is there numerical or visual evidence that the spectral condition from Yang et al. (2023) is satisfied during training?  

*. Would Scaled MUON’s benefits hold across other PINN architectures beyond PirateNet (e.g., MLP, PINNsFormer)?  

* How does the increased computational overhead impact training time compared to baselines?  

* Could experiments investigate MUON’s effect on physical fidelity and gradient flow across a range of PDE types with varying stiffness and multi-physics characteristics?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a Scaled MUON optimizer to perform steepest descent under non-Euclidean norms, which has been proven to stabilize training and make the optimization landscape more scale-invariant. The proposed method successfully trained PINN networks with more than 1,000 neurons per layer, significantly exceeding the previous limit of 200-400 neurons. Theorems are provided to justify the choice of norms, showing that spectral-norm-constrained networks remain universal approximators and that the updates under Scaled MUON are more stable. Overall, this paper has excellent theoretical analysis, but is somewhat lacking in experimental aspects.

### Strengths
1. The motivation of the paper is clear.
2. Theorems 1 and 2 provide a theoretical foundation, explaining why the spectral norm is a better choice than the Frobenius norm for PINNs. 
3. The proposed framework is easy to follow.
4. The paper exhibits a well-organized logical structure, standardized tables, and clear writing.

### Weaknesses
1. While the comparison to baselines is strong, a more detailed ablation study within the Scaled MUON framework is needed. 
2. The experiments mainly focus on relatively low-dimensional problems (1D and 2D). A key challenge for PINNs is scaling to high-dimensional PDEs. It is unclear if the demonstrated scaling laws and the advantages of Scaled MUON hold in such settings. 
3. The author should compare it with more recent PINN methods. And how these methods perform within the framework of this paper.
4. The authors should provide more quantitative metrics to demonstrate the superior performance in the Tables.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
2
