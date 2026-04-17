# Gradient Clipping Accelerates Saddle Avoidance in Distributed Optimization

- Decision: Reject
- Scores: 4, 2, 8, 2

## Abstract
A critical challenge in distributed nonconvex optimization is efficiently avoiding saddle points, which is vital for ensuring accurate and fast convergence to a desired equilibrium point. In this work, we demonstrate that gradient clipping, a technique widely used in machine learning to mitigate gradient explosion, can significantly accelerate the escape from saddle points and improve the speed of convergence to second-order stationary points in distributed optimization. More specifically, we propose an algorithm that exploits gradient clipping to achieve faster convergence in distributed nonconvex optimization. The result is significant in that gradient clipping is necessary and widely used anyway to avoid exploding gradients in deep learning, and hence, the extra benefit of faster saddle avoidance is achieved for free. In fact, we prove that our algorithm converges to a desired second-order stationary point faster than all existing saddle avoidance approaches for distributed optimization. Numerical experiments on benchmark datasets validate the effectiveness of the proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel distributed optimization algorithm that integrates gradient clipping with gradient tracking to accelerate saddle point avoidance in nonconvex settings. The authors demonstrate that gradient clipping, commonly used to mitigate gradient explosion in deep learning, can significantly enhance escape from saddle points and improve convergence to second-order stationary points. The algorithm employs a gradient-tracking framework with periodic noise injection and gradient clipping applied to local gradient estimates. Theoretical analysis proves consensual convergence to first-order stationary points and establishes an iteration complexity of O($\varepsilon^{-3}$) for reaching ε-second-order stationary points, outperforming existing distributed methods. Numerical experiments on benchmark datasets (e.g., Gisette, WallFlower) for logistic regression and robust PCA validate the algorithm's efficiency in escaping saddle points and achieving faster convergence compared to state-of-the-art approaches.

### Strengths
1. **Novel Integration:** The work is the first to successfully incorporate gradient clipping into gradient-tracking-based distributed optimization with provable convergence, addressing the challenge of nonlinearity introduced by clipping.
2. **Theoretical Rigor:** Comprehensive convergence analysis is provided, including guarantees for both first-order and second-order stationary points, with explicit complexity bounds (e.g., O($\varepsilon^{-3}$) iterations) that match centralized or semi-distributed state-of-the-art rates.
3. **Empirical Validation:** Extensive experiments on real-world datasets demonstrate the algorithm's superiority in saddle-point avoidance and convergence speed over existing methods, including scenarios with non-IID data.
4. **Practical Relevance:** Gradient clipping is already widely adopted in deep learning, so the additional benefit of accelerated saddle avoidance comes at no extra cost, enhancing its applicability.
5. **Comparative Advantage:** The algorithm achieves faster saddle evasion than all existing fully distributed counterparts, as shown in Table 1, and avoids the need for global information (e.g., counters in semi-distributed methods).

### Weaknesses
1. **Assumption Dependency:** The theoretical guarantees rely on strong assumptions, such as a symmetric doubly stochastic communication matrix and Lipschitz/Hessian-Lipschitz continuity of objective functions, which may not hold in all practical settings.
2. **Parameter Sensitivity:** Performance is sensitive to hyperparameters (e.g., stepsize $\alpha$, clipping threshold $c_0$, noise amplitude $\theta$), as evidenced by sensitivity analyses in experiments, requiring careful tuning for optimal results.
3. **Complexity of Analysis:** The theoretical derivations are highly technical and may be challenging to follow for practitioners, potentially limiting accessibility and implementation.
4. **Scope Limitations:** The focus on strict saddle points (excluding cases with zero minimal Hessian eigenvalues) may not cover all nonconvex problems, and the assumption of private local objectives could restrict use in scenarios requiring data sharing.
5. **Experimental Scope:** While experiments cover standard tasks, broader validation on large-scale or diverse datasets (e.g., in federated learning environments) is needed to generalize the findings.
6. **Lack of Generalization Performance Evaluation:** The experimental validation primarily focuses on training loss convergence without systematically assessing generalization capability. The absence of test set performance metrics (e.g., test accuracy for logistic regression, reconstruction error on unseen data for robust PCA) makes it difficult to evaluate overfitting risks and the algorithm's practical utility on unseen data.

### Questions
1. How does the algorithm perform under more realistic communication topologies (e.g., time-varying or directed graphs) that violate the symmetric doubly stochastic assumption?
2. Could the theoretical analysis be simplified to provide more intuitive insights into the role of gradient clipping in saddle avoidance?
3. What strategies are recommended for automatic tuning of hyperparameters (e.g., $c_0$, $\theta$) in practice, especially in non-stationary environments?
4. How does the algorithm scale with the number of agents and problem dimension, and are there any communication bottlenecks?
5. Have the authors considered comparing with recent adaptive gradient methods (e.g., Adam) in distributed settings to contextualize the benefits of gradient clipping?

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
2

### Summary
The paper presents a gradient method for distributed machine learning that incorporates gradient tracking with gradient clipping. Theoretically, it analyzes the gradient complexity required to find a second-order stationary point and demonstrates that the method achieves state-of-the-art (SoTA) complexity bounds in some settings. Experimental results are also provided to validate the effectiveness of the proposed approach.

### Strengths
The paper establishes a clear problem setting, and the complexity bounds achieved under settings are demonstrated to be reasonable.

### Weaknesses
I find the current contribution of this paper to be incremental, and therefore recommend revision. My primary concerns are detailed below.

1.  Lack of Dependence on Network Topology: The theoretical analysis does not explicitly characterize how the convergence rates depend on the spectral properties of the connected matrix W. For a decentralized learning paper, explicitly showing this dependence is crucial.

2. Novelty of Convergence Rates: The claimed faster convergence rates do not appear to constitute a significant advancement. The same rates have already been established in the centralized setting. Furthermore, the rate provided for the deterministic setting is comparatively slow.

3. Technical Contribution and Motivation: The extension of non-convex optimization analysis to the decentralized setting is not hard. Because one can treat it as a  linear-constrained problem. The paper would greatly benefit from a more detailed and compelling explanation of its core technical novelty and motivation.

4. Insufficient Citations: The manuscript fails to properly cite the original sources of the convergence rates it obtains. For instance: The epsilon^{-1.75}  rate for the deterministic setting was first achieved by [COLT, 2017, "Accelerated Gradient Descent Escapes Saddle Points Faster than Gradient Descent"]. In the stochastic setting, techniques like variance reduction (or gradient sliding) combined with clipping have been extensively studied in centralized works like [NeurIPS, 2018, "SPIDER: Near-Optimal Non-Convex Optimization..."].

### Questions
See the weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Distributed optimization is becoming increasingly more important in the era when machine learning models are becoming larger. Saddle points can significantly impede training efficiency, and escaping from saddle points can be even more complex in distributed training. This paper proposed a distributed optimization algorithm based on gradient clipping that is usually adopted to avoid gradient explosion. The authors proved the convergence of the proposed method and empirically demonstrated its effectiveness in tasks including non-convex regularized logistic regression, robust PCA, and neural network-based classification.

### Strengths
1. The paper is very clearly structured and well-written.

2. It is surprising to see that simple gradient clipping can help escape from saddle points, since gradient clipping is usually used at the other end of scenario where the gradient magnitude is too large. The effect is intuitive though, since it basically just modifies the effective stepsize. This usage is novel.

3. The theoretical complexity is better than existing saddle avoidance approaches.

4. Figure 2 clearly illustrates the benefits of gradient clipping.

### Weaknesses
1. Selecting appropriate values for stepsize $\alpha$, clipping threshold $c_0$, noise amplitude $\theta$, and noise-injection interval $\kappa_0$ may be difficult for various real-world tasks. Normally, these parameters should be set case by case.

2. The conducted experiments are not large-scale enough.

3. The algorithm only converges to a ball centered at a second-order stationary point. Since the noise is not injected at every iteration, it would be better if the authors could remove $\theta$ from the convergence rate bound.

### Questions
See above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper works on the problem of avoiding saddle points for distributed nonconvex optimization. They claim to show that gradient clipping can be used to find a $\epsilon-$second-order stationary point after $O(\epsilon^{-3})$ iterations. Some robust PCA experiments are performed.

### Strengths
The paper claims (Theorem 4.6) it finds an $\epsilon-$second-order stationary point after $O(\epsilon^{-3})$ iterations in a distributed (not semi-distributed) setting.

### Weaknesses
The main weakness of the paper is that it does not clarify its novelty compared to Xian and Huang (2023). The paper states that the work covers the semi-distributed setting, but it doesn't explain what is the difficulty in extending to the fully distributed setting. What is the main advantage offered here that was not possible for Xian and Huang (2023)? And to what degree do the authors claim the issue of "semi-distributed" is actually a concern? [See also questions.] 

Relatedly, the paper claims that "results for distributed nonconvex optimization are relatively sparse". However, the paper does not cite [1-3], all of which have results for this setting, and even have "decentralized" and "nonconvex"/"non-convex" directly in their titles. (This is not an exhaustive list, just meant to provide evidence, from a very basic literature search, that rebuts the claim.)

Based on these limitations, the result appears incremental in comparison to the (inadequately reviewed) prior literature.

[1] Haoran Sun, Songtao Lu, and Mingyi Hong. "Improving the sample and communication complexity for decentralized non-convex optimization: Joint gradient estimation and tracking." In International Conference on Machine Learning, pp. 9217-9228. PMLR, 2020.

[2] Taoxing Pan, Jun Liu, and Jie Wang. "D-SPIDER-SFO: A decentralized optimization algorithm with faster convergence rate for nonconvex problems." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 02, pp. 1619-1626. 2020.

[3] Ran Xin, Usman Khan, and Soummya Kar. "A hybrid variance-reduced method for decentralized stochastic non-convex optimization." In International Conference on Machine Learning, pp. 11459-11469. PMLR, 2021.

### Questions
The authors claim they achieve $\epsilon-$second-order stationary point after $O(\epsilon^{-3})$ iterations (not $\tilde{O}(\epsilon^{-3})$). Is this a typo? Or do the authors actually believe their claim holds without a $O(\mathrm{polylog}(1/\epsilon))$ factor? If the latter, can the authors explain how they manage to remove the $O(\mathrm{polylog}(1/\epsilon))$ term for achieving $\epsilon-$second-order stationary point?

Should it not be $\tilde{O}$ notation in Xian and Huang (2023) and Avdiukhin & Yaroslavtsev (2021) in "Semi-distributed" rows of Table 1?

Could you elaborate on the distinction between "distributed" and "semi-distributed"? Is there any other work that formalizes this distinction? If not, where in the current work is such a formal definition provided?

Precisely how does introducing gradient clipping significantly complicate convergence analysis? Assuming such complications exist, were they not already addressed in Xian and Huang (2023), which already gives a $\tilde{O}(\epsilon^{-3})$ iteration complexity to find an $O(\epsilon, \sqrt{\epsilon})$ stationary point?

Reference error: The work of Xian and Huang was part of the proceedings of NeurIPS 2023.

### Soundness
2

### Presentation
1

### Contribution
1
