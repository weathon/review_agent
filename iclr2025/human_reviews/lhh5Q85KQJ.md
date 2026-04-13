## Human Reviewer 1

### Summary
This paper proposes a distributed minimax optimization algorithm, SubDisMO, assigning adaptive-sized submodels to the clients based on the resource budgets, thus enhancing the generalization of the global model by training adaptive submodels with perturbations. Additionally, the paper provides the theoretical analysis of convergence and generalization bound of SubDisMO, and experimental results demonstrate its superior effectiveness compared to state-of-the-art methods.

### Strengths
1. SubDisMO is the first distributed minimax optimization algorithm to account for resource budgets, enabling the training of resource-adaptive submodels with perturbations.
2. The paper provides the theoretical analysis of the convergence rate and generalization bound of SubDisMO.
3. It includes an in-depth analysis of different heterogeneity of client data, minimum covering number and the upper bound of the perturbation to evaluate the effectiveness of SubDisMO.

### Weaknesses
1. The formulation of SubDisMO as a distributed minimax optimization problem needs further clarification. Specifically, in Equation 2 on page 2, it is unclear whether, after altering the constraints of the minimax problem, Equation 2 still can be recognized as a minimax problem. This raises questions about whether SubDisMO functions as a distributed minimax optimization problem or more as a solution to minimax problems.
2. While SubDisMO claims that it’s a resource-aware distributed minimax optimization algorithm, it lacks analysis on the impact of varying resource budgets, particularly concerning the influence of mask distribution in SubDisMO.
3. Some figures in the paper need clearer presentations, for example, Figure 2 and Figure 4 need to be clear enough to see the difference between each method setting.

### Questions
1. SubDisMO designs a novel distributed minimax optimization problem, which introduces an inner maximization on perturbation $\epsilon_i$ for each client submodel. I have a question about this definition in Eq.2 on page 2: after changing the constraints of the minimax problem, can Eq.2 be recognized as a minimax problem? Is SubDisMO more like a solution for distributed minimax optimization problems?
2. In SubDisMO, the masking policy of the submodel is random, I have a question: if the masking policy is rolling, whether SubDisMO can still work in this setting? It means the performance of rolling mask with perturbation. Is the performance of this combination better than the current random mask with perturbation?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
5

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper introduces SubDisMO, a resource-aware distributed minimax optimization algorithm designed to enhance the generalization capability of global models in resource-limited environments. SubDisMO mitigates model sharpness by using perturbations to adaptively train submodels, thereby improving overall model performance. The authors provide a rigorous theoretical analysis, proving that SubDisMO achieves an asymptotically optimal convergence rate of $O(1/\sqrt{QTC^*})$ under non-convex conditions, along with a tighter generalization bound based on perturbation and parameter retention rates in each layer. Extensive experiments on CIFAR-10 and CIFAR-100 demonstrate that SubDisMO outperforms existing state-of-the-art methods in resource-constrained training scenarios, validating its effectiveness and superior generalization capabilities.

### Strengths
Originality: SubDisMO introduces a novel approach by integrating distributed optimization, model pruning, and adversarial training, addressing the challenge of training large-scale models in resource-constrained environments.

Quality: The research is of high quality, with rigorous theoretical analysis and extensive experiments that validate the algorithm's effectiveness against state-of-the-art methods on standard datasets.

Clarity: The paper is well-organized and clearly articulates the problem, methodology, and results, with figures and tables that effectively support the presentation.

Significance: SubDisMO's ability to train large models efficiently in distributed settings with limited resources is significant for edge computing and privacy-preserving ML, potentially impacting a broad range of applications.

### Weaknesses
## Complexity Analysis:
The paper could further improve by providing a more detailed analysis of the computational complexity of SubDisMO compared to other methods. While the focus is on communication efficiency, understanding the trade-offs in terms of computational resources could provide a more comprehensive view of the efficiency gains.

## Scalability Concerns:
Although the paper addresses the challenge of limited resources, there is room for a deeper discussion on the scalability of SubDisMO in extremely large-scale distributed systems with potentially thousands of clients. Addressing how the algorithm performs as the number of clients grows would be valuable.

### Questions
## Computational Complexity:
How does the computational complexity of SubDisMO compare to other state-of-the-art distributed learning algorithms?

## Scalability in Large-Scale Systems:
What are the expected performance and convergence characteristics of SubDisMO when the number of clients increases to several thousands? Are there any bottlenecks or challenges anticipated?

### Soundness
4

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
2

---

## Human Reviewer 3

### Summary
This paper studies distributed minimax optimization problem in the context of resource-limited clients. The paper introduces a generalized resource-aware distributed minimax optimization algorithm, which prunes the global large-scale model into adaptive-sized submodels with perturbations to accommodate varying resources during each communication round. The theoretically results demonstrate asymptotically optimal convergence and provide a generalization bound corresponding to the parameters. Experiments on CIFAR-10 and CIFAR-100 demonstrate the superior generalization and effectiveness of the algorithm.

### Strengths
This paper is well-written with clear objectives and contributions. It is the first time that a resource-limited distributed method was proposed for the minimax problem. The proposed algorithm adaptively generates submodels through local resources and trains with perturbations, which enhances the generality of the global model. The theoretical result provides a convergence bound that matches the sota result for full model settings and a generalization bound corresponding to the perturbation and parameter remaining rate in each layer.
Experiments verifies the results.

### Weaknesses
My major concern about this paper lies in the assumptions on the stepsize $\eta_l$ and $\eta_g$ in the theoretical analysis.
i). The conditions in Theorem 1 are somehow self-contradictory. According to line 3 - 6 in the big blanket, $\eta_g\leq\frac{\sqrt{C*}}{8TL\sqrt{N}}$ and $\eta_g\geq 128L$, it follows that L must be a very small number, i.e., $L\leq\frac{1}{32\sqrt{T}}$, and as T increases, L must become increasingly smaller. This is a very strong assumption. 
ii).The values of $\eta_g$ in Corollary 1 do not satisfy the condition $\eta_g \leq \frac{\sqrt{C*}}{8TL\sqrt{N}}$ in Theorem 1.

Besides, the numerical results for the full model version of SubDisMO should be added. Theoretically, the performance of this version should align with that of FedAvg and FedSAM, but these results are not included in Table 1. Adding this comparison would provide a more comprehensive evaluation.

### Questions
1.In Theorem 1, it appears that a smaller $\delta$ leads to a better bound, which is counterintuitive and contradicts the experimental performance. Could you further clarify the impact of $delta$?

2.Why do the performances of FedAvg and FedSAM in your experiments appear to be lower than those reported in the original paper? The original results show that both methods achieve over 80% accuracy on CIFAR-10, whereas your reported results are below 60%.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
4