# Understanding the Dynamics of Forgetting and Generalization in Continual Learning via the Neural Tangent Kernel

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 8

## Abstract
Continual learning (CL) enables models to acquire new tasks sequentially while retaining previously learned knowledge. 
However, most theoretical analyses focus on simplified, converged models or restrictive data distributions and therefore fail to capture how forgetting and generalization evolve during training in more general settings. 
Current theory faces two fundamental challenges: (i) analyses confined to the converged regime cannot characterize intermediate training dynamics; and (ii) establishing forgetting bounds requires two-sided bounds on the population risk for each task. 
To address these challenges, we analyze the training-time dynamics of forgetting and generalization in standard CL within the Neural Tangent Kernel (NTK) regime, showing that decreasing the loss’s Lipschitz constant and minimizing the cross-task kernel jointly reduce forgetting and improve generalization. 
Specifically, we (i) characterize intermediate training stages via kernel gradient flow and (ii) employ Rademacher complexity to derive both upper and lower bounds on population risk. 
Building on these insights, we propose \emph{OGD+}, which projects the current task’s gradient onto the orthogonal complement of the subspace spanned by gradients of the most recent task evaluated on all prior samples. 
We further introduce \emph{Orthogonal Penalized Gradient Descent} (OPGD), which augments OGD+ with gradient-norm penalization to jointly reduce forgetting and enhance generalization. 
Experiments on multiple benchmarks corroborate our theoretical predictions and demonstrate the effectiveness of OPGD, providing a principled pathway from theory to algorithm design in CL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper leverages the Neural Tangent Kernel (NTK) to derive upper bounds for both forgetting and generalization error at any step during training in vanilla CL tasks, while the bounds explicitly depending on the training steps. Two key conclusions are drawn: 1. Reducing the loss function’s Lipschitz constant (ρ) simultaneously decreases forgetting and generalization error. 2. Decreasing the magnitude of the cross-task kernel also mitigates forgetting and improves generalization. Based on these theoretical insights, the paper proposes two methods, OGD+ and OPGD, which project the current task’s gradient onto the orthogonal complement of the subspace spanned by
the gradients of recent tasks on all historical samples. Additionally, a gradient norm penalty (GAM) is introduced to jointly enforce “orthogonal constraints + Lipschitz reduction,” thereby suppressing forgetting while enhancing generalization. Experimental results demonstrate that the proposed methods significantly reduce errors. The writing is clear, the methodology sound, and the experimental outcomes show significant improvement

### Strengths
1. This paper models the error upper bound as an explicit function of the training step, allowing it to not only identify the main factors influencing the error but also discuss the effects of “training longer or shorter” on the final performance.

2. The main theorem clearly provides upper bounds for both forgetting and generalization errors, explicitly highlighting their dependence on training steps. By deriving a closed-form solution through kernel gradient flow and incorporating Rademacher complexity for two-sided control of the overall risk, the paper establishes a solid theoretical foundation. It further proposes a corresponding improvement method (OPGD) and proves its superiority, forming a coherent and logically complete framework.

3. The paper is organized around a clear framework: problem → challenges → theory → analysis → algorithm→ experiments → conclusion. The overall exposition is thorough, and the figures and tables effectively present the results in a clear and intuitive manner.

### Weaknesses
1. OPGD still requires storing a large amount of gradient data, leading to high computational and memory costs.  This may limit its applicability to large-scale models and tasks.

2. Beyond the current experimental datasets, it would be valuable to introduce more complex and realistic scenarios to further validate the robustness and generalization ability of the proposed algorithm.

3. Theoretical analysis occupies a large portion of the paper, while the main experimental results are relatively limited, making the demonstration of the method’s advantages less compelling.

### Questions
1. Is there a plan to evaluate them on more realistic tasks, such as online visual data streams or natural language sequences?

2. If the model deviates from the NTK assumption (e.g., CNNs), have different behaviors been observed in terms of forgetting and generalization mechanisms?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper analyzed continual learning in the Neural Tangent Kernel regime and derive intermediate-training bounds for both forgetting and generalization using kernel gradient flow and Rademacher complexity. They show that two quantities govern CL behavior during training: (i) the Lipschitz constant of the loss w.r.t. predictions—smaller is better for both forgetting and generalization, (ii) the cross-task kernel smaller norms reduce interference and forgetting. Building on these findings, they propose OGD+, which projects the current task’s gradient onto the orthogonal complement of a subspace formed by the previous task’s gradients evaluated on all earlier samples (eliminating cross-task kernels between any task pair), and OPGD, which augments OGD+ with gradient-norm penalization to lower the Lipschitz constant.

### Strengths
1. The paper is comprehensive, well written and well structured.

### Weaknesses
1. What means intermediate training?

2. The continual-training protocol is unclear-for example, how does the transition from task $k$ to task $k+1$ work? After training $t$ steps on task $k$, how is gradient flow for task $k+1$ initialized and defined relative to the state from task $k ?$

3. The assumption also requires a bounded loss function.

4. The paper did not specifically provide the forgetting and generalization error of PGN and PGN, only lemma 2 and lemma 3 provided, which is not enough; it should give the formal results like Theorem 1.

5. The comparison among related work is not sufficient. There are lots of theoretical CL works that consider the regularized setting, but the paper did not discuss.

6. The proposed methods are common in empirical CL studies, while the theoretical CL findings are not enough in my view.

### Questions
Above

### Soundness
2

### Presentation
3

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
The paper addresses the challenges of continual learning (CL), specifically focusing on the dynamics of forgetting and generalization. Continual learning aims to enable models to learn tasks sequentially without forgetting previously acquired knowledge. The authors analyze these dynamics within the Neural Tangent Kernel (NTK) regime, addressing two main challenges: characterizing intermediate training stages and establishing forgetting bounds using population risk bounds. They introduce two novel algorithms, OGD+ and Orthogonal Penalized Gradient Descent (OPGD), both of which project gradients orthogonally to reduce forgetting and enhance generalization. Their empirical studies on benchmarks like Permuted MNIST, Rotated MNIST, and Split CIFAR-100 validate the theoretical predictions and demonstrate the effectiveness of their approaches.

### Strengths
1. **Theoretical Contribution**: The paper extends the NTK-based analysis to intermediate training dynamics, which is crucial for understanding real-world continual learning scenarios. By providing both upper and lower bounds on population risk using Rademacher complexity, the authors offer rigorous theoretical insights.
    
2. **Algorithm Design**: OGD+ and OPGD are innovative in their approach to mitigate forgetting and improve generalization. The use of gradient orthogonality and gradient-norm penalization represents a well-founded strategy for continual learning models.
    
3. **Empirical Validation**: The experiments conducted on various benchmarks, such as Permuted MNIST and Split CIFAR-100, confirm the theoretical discussions, demonstrating the practical efficacy of OPGD and OGD+ compared to standard methods.
    
4. **Clear Presentation**: The paper clearly outlines its contributions, setting a solid baseline through comparative analysis with other methods in Table 1. This provides context for the improvements made by the proposed methods.

### Weaknesses
1. **Incremental Algorithmic Novelty**:  The algorithms are mainly based on OGD and gradient norm penalty, which seems to be a little bit incremental. 

2. **Limited Applicability**: The theoretical framework is developed under the NTK regime, which might not encapsulate the full behavior of practical deep networks with finite width and more complex architectures. This limitation is acknowledged in the paper, suggesting a need for exploring applicability to diversified settings.
    
3. **Specific Task Benchmarks**: The empirical tests are confined to a specific set of benchmarks, potentially overlooking performance variations across other task types. While benchmarks like Permuted MNIST and Split CIFAR-100 are standard, testing across a broader spectrum could provide more insights.

### Questions
1. **Computation Complexity**: Is the projection step computationally heavy for complex NN? How to address this problem? 

2. **Expansion Beyond NTK**: How would the authors suggest overcoming the limitations of the NTK-based analysis when dealing with finite-width networks or other types of architecture beyond classification tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the theoretical properties of CL in the NTK regime in the intermediate training stage, and give two insights on the role of Lipschitz and cross-task kernel on CL performance.
With these insights, they give two refined versions of the OGD algorithm.
These two algorithms are verified on standard CL datasets.

### Strengths
This is a good paper in both theoretical and practical aspects. 

On the theoretical side, intermediate-stage CL performance analysis is novel to the theoretical CL literature. Also, the Lipschitz observation is novel to the CL field. The cross-task kernel result is valid and kind of expected.

On the practical side, with the two theoretical messages, the refined algorithms refined a well-used CL algorithm on these aspects and improved its performance on standard basic CL datasets.

The logic is clear. The mathematical tools used are standard.

I am pushing this paper for acceptance.

### Weaknesses
I do not see major weaknesses in this paper.

### Questions
How do you think the two theoretical messages can benefit other CL algorithms?

### Soundness
3

### Presentation
3

### Contribution
3
