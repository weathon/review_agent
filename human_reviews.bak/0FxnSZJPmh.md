# Physics-Informed Deep Inverse Operator Networks for Solving PDE Inverse Problems

- Decision: Accept (Poster)
- Scores: 6, 5, 6

## Abstract
Inverse problems involving partial differential equations (PDEs) can be seen as discovering a mapping from measurement data to unknown quantities, often framed within an operator learning approach. However, existing methods typically rely on large amounts of labeled training data, which is impractical for most real-world applications. Moreover, these supervised models may fail to capture the underlying physical principles accurately. To address these limitations, we propose a novel architecture called Physics-Informed Deep Inverse Operator Networks (PI-DIONs), which can learn the solution operator of PDE-based inverse problems without any labeled training data. We extend the stability estimates established in the inverse problem literature to the operator learning framework, thereby providing a robust theoretical foundation for our method. These estimates guarantee that the proposed model, trained on a finite sample and grid, generalizes effectively across the entire domain and function space. Extensive experiments are conducted to demonstrate that PI-DIONs can effectively and accurately learn the solution operators of the inverse problems without the need for labeled data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Physics-Informed Deep Inverse Operator Networks (PI-DIONs) for solving PDE-based inverse problems without the need for labeled data. The paper extends existing stability estimates from inverse problem literature to the operator learning framework, ensuring the robustness and generalizability of PI-DIONs across the entire function space and domain.

### Strengths
1. This paper provides a solid theoretical foundation for the proposed PI-DIONs.
2. The proposed method demonstrates practicality and efficiency in addressing PDE-based inverse problems without the need for labeled data.

### Weaknesses
1. The contribution lacks novelty. The architecture relies on relatively simple components, such as CNNs and MLPs for the branch and trunk networks. It doesn't introduce significant advancements beyond well-established methods.
2. The baselines used for comparison, such as DeepONet and FNO, are somewhat dated. The paper would benefit from comparisons with more recent and state-of-the-art methods to better demonstrate the model's competitiveness.
3. The experimental evaluation is limited in range. Conducting experiments on a broader range of benchmarks would strengthen the validation of the proposed method's effectiveness across diverse problems.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes an architecture called Physics-Informed Deep Inverse Operator Networks (PI-DIONs), which can learn the solution operator of PDE-based inverse problems without labeled training data. The architecture  of PI-DIONs is based on DeepONet, and trained with both the physics-infomred loss and data reconstruction loss. The stability estimates established in the inverse problem literature are extended to the operator learning framework. Experiments are conducted to demonstrate the effectiveness of PI-DIONs in learning the solution operators of the inverse problems without the need for labeled data.

### Strengths
1. The integration of physics-informed losses into an inverse problem framework based on operator learning is novel, and in principle PI-DIONs can solve the inverse problems (at least in scenarios mentioned in experiments) fast and without the need for labeled data.
2. Theoretical analysis of the stability estimates is provided.

### Weaknesses
1. Line 243, "where the term ∥f − f^\star∥L2(Ωm) in the righthand side", there is no such term there. Please clarify the equation in line 242 and include all terms on the right-hand side of the equation. 
2. It seems that the input to the reconstruction and inverse branch networks is fixed in shape, corresponding to the partial measurement with given geometry. The observed data in PINNs can have variable count and locations. Please discuss how PI-DIONs might be adapted to handle variable measurement geometries and if there are any limitations on the types of measurement setups it can handle. 
3. In the experiments, PI-DIONs are compared with purely data-driven DeepONet and FNO, which both did not take physics information into account. If possible, please include comparisons with PINNs in the experiments, since both your PI-DIONs and PINNs are physics-informed methods for inverse problems.
4. The simultaneous training of physics-informed losses for 1000 samples is a difficult task (similar to train 1000 PINNs simultaneously). I am curious about the training difficulties encountered. Please provide specific details on training time, hardware used, and any convergence challenges encountered. If possible, please also include an ablation study on the effect of sample size on PI-DIONs' performance since smaller sample size may lead to easier optimization.
5. The theoretical analysis on stability estimate is extended from existing key results that considered the single element case. 
6. Please provide a clear definition of u in line 152 and describe its relationship with partial measurement. In line 456, it is better to write "f(x,y) = 100x(1 − x)y(1 − y) ", so does line 450. 

Considering the above weaknesses, I give a score of 3 to the current version of this paper.

### Questions
1. DeepONet and FNO are used for forward problems traditionally, how did they deal with inverse problems in your experiments?
2. How is the labeled training target f mentioned in line 399 used?  The loss for target f is absent in line 152.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose Physics-Informed Deep Inverse Operator Networks (PI-DIONs), a novel architecture for solving PDE inverse problems without requiring labeled data. Theoretically, the authors extend stability estimates from traditional inverse problem theory to the operator learning setting, and prove universal approximation theorems for PI-DIONs. Empirically, the authors validate their proposed approach through experiments on reaction-diffusion equations, Helmholtz equations, and Darcy flow, achieving SOTA performance.

### Strengths
- The paper engages with an important problem in SciML, learning to solve inverse problems based on physics losses without additional training data.
- The theoretical results are quite interesting. The authors extend standard stability estimates for inverse problems to the operator learning setting. Promisingly, the theorems apply to the reaction-diffusion equation and the Helmholtz equation, two standard benchmarks in the literature.
- The proposed method is simple and presented clearly and generally.
- The empirical results are promising. On three standard benchmarks, the authors demonstrate SOTA performance of supervised learning and near-SOTA of unsupervised learning, compared to supervised DeepONet and FNO.

### Weaknesses
- The main weakness of the paper is that the empirical results, although promising, are relatively limited and could benefit from some clarification:
  - In Table 1, PI-DION in the supervised learning setting (with 1k training examples) is shown to outperform two different DeepONets and FNOs. However, it's a bit unclear from the paper why this is true, and additional clarification about this would be helpful. Is there a difference in the model architecture / training objective / optimizer between the DeepONets and PI-DION in the supervised setting?
  - See questions for more.

### Questions
- Questions about experimental results:
  - What are the number of parameters of each of the models in Table 1?
  - Could the authors provide a sensitivity analysis showing how performance changes as the relative weighting between physics and data losses is varied? This would provide valuable insight into the method's robustness.
  - How does PI-DION compare to other methods for solving inverse problems, e.g. Neural Inverse Operators [1]?
  - Any explanation about why the performance hit between supervised and unsupervised PI-DION is larger for Darcy Flow and Helmholtz equation than for reaction-diffusion?

- How limiting is the assumption that there exists stability estimates for the inverse problem?
- How well do the theoretical bounds from Theorems 2, 3 match the empirical results of Table 1 (reaction-diffusion and Helmholtz)?

1. Neural Inverse Operators for Solving PDE Inverse Problems

### Soundness
2

### Presentation
3

### Contribution
2
