# CONGO: Compressive Online Gradient Optimization

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
We address the challenge of zeroth-order online convex optimization where the objective function's gradient exhibits sparsity, indicating that only a small number of dimensions possess non-zero gradients. Our aim is to leverage this sparsity to obtain useful estimates of the objective function's gradient even when the only information available is a limited number of function samples. Our motivation stems from the optimization of large-scale queueing networks that process time-sensitive jobs. Here, a job must be processed by potentially many queues in sequence to produce an output, and the service time at any queue is a function of the resources allocated to that queue. Since resources are costly, the end-to-end latency for jobs must be balanced with the overall cost of the resources used. While the number of queues is substantial, the latency function primarily reacts to resource changes in only a few, rendering the gradient sparse. We tackle this problem by introducing the Compressive Online Gradient Optimization framework which allows compressive sensing methods previously applied to stochastic optimization to achieve regret bounds with an optimal dependence on the time horizon without the full problem dimension appearing in the bound. For specific algorithms, we reduce the samples required per gradient estimate to scale with the gradient's sparsity factor rather than its full dimensionality. Numerical simulations and real-world microservices benchmarks demonstrate CONGO's superiority over gradient descent approaches that do not account for sparsity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduced the compreesive online gradient optimization framework (CONGO) for solving sparse online convex optimization problems based on motivating examples in real world application. Three variants using simultaneous perturbation and compressive sensing are proposed based on different inspiration. The proposed algorithms were validated both theoretically and experimentally.

### Strengths
1. The idea of using compressing sensing to estimate gradients more efficiently is novel. From the theory provided, the total cost of function value evaluation is reduced which makes it suitable for problems of large dimension.

2. Based on the specific setting and the measurements used, three different variants are proposed, which makes the framework flexible.

3. The paper presents theory on the regret bound and validate the theory with experiments.

### Weaknesses
While the paper is generally interesting and the idea is novel, the reviewer wants to point out that the paper contains *formatting errors*, specifically, the heading "Under review as a conference paper at ICLR 2025" is missing at each page.

1. It seems to the reviewer that the proposed method needs careful tuning of the parameters, such as the step size and the sparsity, which may require additional information which makes the proposed algorithm less practical. 

2. CONGO-B seems to be less efficient and more unstable compare to the other variants, this is not explained in theory.

3. The proposed assumptions are a little restrictive in reality.

### Questions
1. How sensitive is the proposed algorithm to sparsity? What would happen if the sparsity measure $s$ is large, will the performance still be comparable to other existing methods?

2. Could adaptive strategies be used for the sparsity?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a framework designed for zeroth-order online convex optimization (OCO) in environments where gradients are often sparse. The core idea is to combine compressive sensing techniques with online optimization to take advantage of this sparsity, which allows for efficient high-dimensional optimization with fewer samples. 

The authors propose three variations—CONGO-B, CONGO-Z, and CONGO-E—that utilize different compressive sensing approaches and gradient estimation methods. They back up their approach with theoretical guarantees, showing sublinear regret bounds, and validate the framework through experiments on both synthetic and real-world tasks (like microservice autoscaling).

### Strengths
1. The use of compressive sensing within an OCO framework is a fresh and well-motivated idea. By focusing on sparse gradients, the authors address both sample efficiency and dimensionality reduction, which are critical in high-dimensional settings. 

2. The authors provide rigorous theoretical analysis, establishing regret bounds that demonstrate sublinear scaling with respect to the problem horizon, independent of the problem dimension.

3. The three algorithmic variants, CONGO-B, CONGO-Z, and CONGO-E, offer a nice balance of performance and complexity. For instance, CONGO-E uses Gaussian matrices and CoSaMP for enhanced performance, while CONGO-Z is more sample-efficient.

### Weaknesses
1. The theoretical analysis assumes exact gradient sparsity, which may not be realistic for all real-world problems.

2. While CONGO outperforms standard gradient descent with SPSA, it’s mostly compared against methods that don’t leverage sparsity. A more comprehensive comparison with advanced sparse optimization techniques or regularized gradient estimators would help here.

### Questions
1. The theoretical analysis assumes exact gradient sparsity, which may not hold in all practical scenarios. Could the authors discuss potential extensions of their theoretical analysis to approximately sparse gradients, or provide insights on how performance changes as the level of sparsity decreases? This could help clarify the framework's robustness and practical applicability.

2. While the regret bounds are dimension-independent, sample complexity grows logarithmically with dimension. In very high-dimensional settings, how does this sample complexity impact practical performance?

3. The framework is compared to standard SPSA-based methods, but how does it compare to other advanced sparse optimization or regularized gradient estimation techniques?

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
3

### Summary
This paper proposed a framework for online zeroth-order optimization leveraging the techniques and insights from compressed sensing. The authors provide several schemes for efficiently sampling the objective functions values and estimate the gradients, alongside with theoretical convergence proofs revealing the fast convergence rates. The numerical results demonstrate this approach's superior performance over state-of-the-art baselines.

### Strengths
This is a well-written paper in general, the idea of introducing compressed sensing for estimating the gradients is very inspiring. The numerical performance of the proposed scheme is excellent. The presentation of the paper is very clear and easy to read.

### Weaknesses
The novelty of the proposed scheme may be potentially limited (rebuttal against this point is welcomed as the reviewer is not familiar with zeroth-order optimization literature). The reviewer has seen similar approach been proposed in Wang et al, "Stochastic zeroth-order optimization in high dimensions" AISTATS'18, where they utilized a very similar idea but used LASSO (L_1) instead of CoSAMP (L_0). The numerical study did not considered this AISTATS'18 paper as a baseline, although being cited in the reference.

### Questions
Could the authors clarify the fundamental difference between the proposed method and AISTATS'18 paper mentioned above? Please highlight the advantages of the proposed method.

In the numerical experiments, did you observe any phase transition on the number of measurements to the convergence speed of the proposed algorithms?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies zeroth-order online convex optimization, where the gradient of the objective function is assumed to be sparse. The proposed algorithms, CONGO, combine the (projected) gradient descent algorithm for online convex optimization, with a gradient estimation procedure using compressive sensing technique. The regret is proven to be O(\sqrt(T)) and does not depend on the dimension of the problem, and the per-iteration sampling complexity scales with the sparsity level of the gradient. Experiments confirm the effectiveness of the algorithms.

### Strengths
The paper is overall well written, and presents the setup, results, and proof clearly. The algorithms proposed appear efficient in terms of the regret and the sampling complexity.

### Weaknesses
-- One might argue that the results are not too surprising: the regret follows from the regret of online gradient descent, while the sampling complexity follows from the compressive sensing results. 

-- In CONGO-B, line 827 – 829, gradient recovery requires solving an LP, which can be computationally inefficient, especially in high-dimensional setting. In addition, compressive sensing usually requires knowledge of the sparsity level before setting the number of samples. If such knowledge is lacking or inaccurate, compressive sensing might fail completely [1]. 

[1] Amelunxen, Dennis et al. “Living on the edge: phase transitions in convex programs with random data.” Information and Inference: A Journal of the IMA 3 (2013): 224-294.

### Questions
The regrets in Theorem 2 and 3 holds in expectation, but if understanding is correct, they also hold with high probability?

### Soundness
3

### Presentation
3

### Contribution
3
