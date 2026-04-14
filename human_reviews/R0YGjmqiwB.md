# Efficient Learning in Neural Networks without Gradient Backpropagation

- Decision: Reject
- Scores: 3, 5, 3, 3

## Abstract
The brain possesses highly efficient learning algorithms that have not been fully understood. The gradient backpropagation (BP) serves as a powerful tool for training artificial neural networks, but it diverges from the known anatomical and physiological constraints of the brain. Conversely, biologically plausible learning algorithms have efficiency limitations in training deep neural networks. To bridge this gap, we introduce a perturbation-based approach called low-rank cluster orthogonal (LOCO) weight modification. Theoretical analysis shows that LOCO provides an unbiased estimate of the BP gradient and achieves low variance in gradient estimation. Compared with some brain-inspired algorithms, LOCO keeps mathematical convergence guarantees and improves the efficiency. It can train the deepest spiking neural networks to date without gradient backpropagation, achieving state-of-the-art performance on several benchmark datasets and exhibiting the ability to overcome catastrophic forgetting. These findings suggest that biologically feasible learning methods can be substantially more efficient than previously believed. Furthermore, avoiding gradient backpropagation allows LOCO to achieve O(1) time complexity for weight updates. This opens a promising avenue for developing distributed computing systems that are more efficient than BP-based counterparts.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper presents an improved training algorithm based on node perturbation. The main idea is to project the gradients to be altered into a low-dimensional space that is orthogonal to the direction of the original gradient changes, thereby accelerating the convergence speed of node perturbation.

### Strengths
The theoretical aspects of this paper are clearly articulated, which is one of its strengths.

### Weaknesses
However, I have two main concerns:

1. **Biological Plausibility of Node Perturbation Compared to Backpropagation**: In the work of Lillicrap et al., it is suggested that backpropagation can actually be implemented using a local Hebbian-like rule. The conclusion regarding its biological implausibility arises from three main points:
   - a. Backpropagation demands synaptic symmetry in the forward and backward paths.
   - b. Error signals are signed and potentially extreme-valued.
   - c. Feedback in the brain alters neural activity.
   
   It would be beneficial to clarify how node perturbation or the proposed LOCO method addresses these three issues to demonstrate its greater biological plausibility.

2. **Comparison with Other Learning Algorithms**: My second concern stems from the lack of comparisons with other learning algorithms, which challenges the novelty of this work. The related work section is insufficient; there exists a substantial class of non-backpropagation algorithms, such as:
   - Neural Sampling in Hierarchical Exponential-family Energy-based Models
   - Predictive Coding in the Visual Cortex: A Functional Interpretation of Some Extra-Classical Receptive-Field Effects
   
   Additionally, the concept of orthogonal projection has already been explored in related works, such as:
   - Hebbian Learning Based Orthogonal Projection for Continual Learning of Spiking Neural Networks.
   
   Furthermore, node perturbation has been extensively studied in the context of artificial neural networks, including models like the forward-forward model, 
   - Scaling forward gradient with local losses
   - Gradients without backpropagation. 
I hope the authors can include comparisons with these learning algorithms, as both experimental and theoretical evaluations are lacking in this paper. This may obscure the novelty of the proposed work.

### Questions
see weaknesses

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
4

### Summary
This paper addresses the well-known challenges of implementing backpropagation in neural systems and proposes an alternative learning rule, LOCO, to train neural networks. LOCO extends node perturbation (NP) by introducing two additional constraints—low-rank and orthogonality in the weight modification space—which significantly improve learning efficiency. Through empirical results on tasks like XOR, MNIST, and phonetic transcription, alongside theoretical analysis, the authors demonstrate that LOCO outperforms traditional NP and others.

### Strengths
•	The pursuit of biologically plausible alternatives to backpropagation is a significant and intriguing problem for computational neuroscientists, though it holds less interest for the ML community, where backpropagation is already effective and versatile.

•	The authors support their findings by combining empirical results on well-known benchmarks with theoretical analysis, which strengthens the validity of their approach (although the theoretical assumptions would benefit from clearer presentation within formal statements).

### Weaknesses
•	If biological plausibility is the main motivator for this work, the learning rule should be computable using known biological processes. However, the projection matrix P lacks clear justification in terms of how it could be implemented biologically.

•	A significant concern of this submission is the absence of a discussion on the limitations.

•	A closely related work by Duncker and Driscoll et al. (NeurIPS 2020), which also uses a projection matrix P to promote orthogonal subspace learning (minimizing interference in continual learning), was not cited.

•	The presentation is mostly clear; however, the title is overly general and does not clearly distinguish this work from the extensive literature on learning without backpropagation.

•	The theoretical analyses are not framed within formal Theorem statements, making it unclear what specific conditions or assumptions (e.g., convexity) apply.

•	Minor typos: mixing up of \citep vs \citet in the second-to-last paragraph of the introduction, as well as "wights" on line 102 and "perofmrance" on line 399.

### Questions
•	Can you demonstrate how the projection matrix P could be computed using known biological signals?

•	Does your learning rule perform effectively only when the task can be learned via low-dimensional orthogonal manifolds? How does the rank of the task influence performance, and does the task rank need to be known in advance?

•	Could you discuss how your method relate to Duncker and Driscoll et al. (NeurIPS 2020)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces a new learning algorithm, low-rank cluster orthogonal (LOCO) weight modification. LOCO improves on node perturbation by projecting the layer inputs into task-dependent orthogonal low dimensional space, effectively reducing the parameter space searched by node perturbation. LOCO’s performance is tested on MNIST and Netalk datasets, including a catastrophic forgetting task, for both spiking neural networks and artificial neural networks and found to be better than vanilla node perturbation.

### Strengths
- Interesting algorithm that has connections to neuroscientific data and the orthogonal activity subspaces found in e.g. motor control. 
- Testing the algorithm’s performance across both SNNs and ANNs is a strength. 
- Algorithm certainly improves over node perturbation.

### Weaknesses
- Abstract, statement about traditional brain-inspired algorithms. Convergence guarantees to what? Also this is an incorrect statement, node perturbation itself has guarantees and e.g. see https://arxiv.org/pdf/2110.10815 for FA. 
- The proposed approach’s efficiency  is not tested (e.g. flops) and does not seem to be efficient as the projection matrix must be calculated at every update. 
- Similar to efficiency aspects, the proposed approach trades off one aspect of biological plausibility for another (the cluster orthogonal projection algorithm on the inputs).
- The method seems to rely on a simple task structure. Namely that the inputs can be projected to low dimensions (without losing too much task relevant information) and orthogonalised using kmeans clustering (relying on data prototypes). Neither of these seems valid assumptions for complicated tasks, and the submission does not test on appropriately hard tasks such as those in bartunov 2018 (Assessing the Scalability of Biologically-Motivated Deep Learning Algorithms and Architectures for ANNs.)
- Figures are small and difficult to read at standard zoom. In particular Fig 1. 
- Comparisons to BP are not always made

### Questions
Couple of typos: 
- 102 and 156, typo weights 
- 160 to obtained, 
- 319 we varies
- 399 typo performance

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
In this manuscript, the authors propose a novel biologically plausible learning algorithm called LOCO. This algorithm is based on the node perturbation (NP) method but incorporates an update projected onto a low-rank activity subspace. The authors explore the theoretical properties of LOCO and demonstrate numerically that it outperforms vanilla NP and several previously proposed algorithms when applied to multi-layered spiking neural networks on the MNIST and NETtalk datasets.

### Strengths
The numerical experiments are thoroughly and clearly explained.
Moreover, the results convincingly demonstrate the benefits of the proposed algorithm over conventional NP for deep neural networks with more than five hidden layers.

### Weaknesses
There are several mathematical inaccuracies in the presented results that affect the soundness of the conclusions:

- The abstract states that "LOCO provides an unbiased estimate of the BP gradient," which appears to be incorrect. In the limit of infinite perturbations, the LOCO update becomes $\Delta W^{LOCO}_l = \Delta W^{NP}_l P_l^T = \Delta W^{BP}_l P_l^T$. Thus, unless the projection matrix $P_l$ is full-rank, LOCO introduces a bias relative to the BP gradient.

- In Line 894, $\ell$ is referred to as the Hessian matrix. However, according to Malladi et al. (2024), $\ell$ represents the smoothness parameter (the Lipschitz constant) of the loss function. 

- Eq. 17: trace is missing from the Frobenius inner product.

- Eq. 18: While a matrix P satisfying this equation exists, P in this context is no longer a projection matrix. 

- A.3: Inequalities are missing from the equations throughout the proof, making the arguments difficult to follow.

- L965-966: The origin of $tr [\sum(\theta_t)]$ is unclear, and the equation appears to be incorrect. 

- L1221: Although $g$ follows a Gaussian distribution if the perturbation is Gaussian, the covariance is not necessarily the identity matrix.

- A 15: There seems to be confusion between subspaces in the activity space and the weight space. Even if the activity of pre-units resides in an r-dimensional space, the gradient g may not be constrained to an r-dimensional space.

- L1239-1241: Is it assumed that $\sigma^2 = || \mu_g ||^2$? If so, why should that be the case?

While the learning algorithm is motivated by biological plausibility, the method for obtaining the projection matrix $P_l$ lacks biological plausibility. 

It is well known that orthogonalization prevent catastrophic forgetting (French, CogSci 1991; Farajtabar et al., AISTATS 2020), hence the presented result on continual learning is not surprising.

### Questions
How did you control the learning rate when estimating learning efficiency? Additionally, algorithmic complexity may be less relevant in inherently parallel systems, such as the brain, neuromorphic chips, and, to a lesser extent, GPUs. How does this affect your conclusion?

Wouldn’t it be more effective to consider low-rank perturbations of the postsynaptic units rather than projecting the presynaptic units into a low-rank space? What is the motivation or benefit of projecting $x_{l-1} $ to a low-rank space instead of adding perturbations in a low-rank space?

### Soundness
1

### Presentation
3

### Contribution
2
