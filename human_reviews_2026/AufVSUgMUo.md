# Task-free Adaptive Meta Black-box Optimization

- Decision: Accept (Oral)
- Scores: 2, 6, 6, 8

## Abstract
Handcrafted optimizers become prohibitively inefficient for complex black-box optimization (BBO) tasks. MetaBBO addresses this challenge by meta-learning to automatically configure optimizers for low-level BBO tasks, thereby eliminating heuristic dependencies. However, existing methods typically require extensive handcrafted training tasks to learn meta-strategies that generalize to target tasks, which poses a critical limitation for realistic applications with unknown task distributions. To overcome the issue, we propose the Adaptive meta Black-box Optimization Model (ABOM), which performs online parameter adaptation using solely optimization data from the target task, obviating the need for predefined task distributions. Unlike conventional metaBBO frameworks that decouple meta-training and optimization phases, ABOM introduces a closed-loop adaptive parameter learning mechanism, where parameterized evolutionary operators continuously self-update by leveraging generated populations during optimization. This paradigm shift enables zero-shot optimization: ABOM achieves competitive performance on synthetic BBO benchmarks and realistic unmanned aerial vehicle path planning problems without any handcrafted training tasks. Visualization studies reveal that parameterized evolutionary operators exhibit statistically significant search patterns, including natural selection and genetic recombination.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes ABOM, a black-box optimization algorithm which adaptively adjust parameters based on experience from function evaluations. ABOM parameterizes the selection, crossover and mutation module in evolutionary algorithm via MLP with attention mechanism, and train these parameters by approximating to the elite selection procedure. The authors provide the global convergence guarantee of ABOM. The experimental results over both synthetic and UAV benchmarks demonstrate that ABOM matches or exceeds commonly used BBO baselines.

### Strengths
1. The proposed MLP parameterized algorithm can benefit from GPU parallel computation.

2. ABOM demonstrates superior empirical optimization performances on both synthetic and UAV benchmarks.

### Weaknesses
1. The theoretical and empirical algorithm analysis does not explain the reason of using attention-based MLP to parameterize the evolutionary modules. From the algorithm design side, the "optimal" policy corresponds to a elite-selection style EA algorithm. In the theoretical analysis, the main results are derived by setting all parameters except the last layer's bias to 0. The global convergence guarantee is not finite-time, which seems to hold simply because of the introduced randomness from dropout. I think other EA based on elite selection might also fit to this infinite-time convergence guarantee. 

2. The proposed ABOM also introduces additional hyperparameters such as dropout rate and hidden dimensions, which is sensitive according to figure 5 and non-trivial to tune like other BBO algorithms such as CMA-ES.

3. The target problem of the proposed method should be single-task black-box optimization problem, where algorithm sample strategies and parameters are usually adjusted based on the sampled data (e.g., MLE-based parameter fitting in Bayesian optimization, and step-size adjustment in CMA-ES). Therefore, I think meta BBO problem and methods are not much related to the main research topic of this paper.

4. I think adding more recent EA-based or NN-based BBO baselines such as [1, 2] and more challenging high-dimensional benchmarks used in these works can better assess the algorithm performance.

5. (minor) The presentation can be improved to be clearer. (1) The citation format should be \citep or \citet instead of \cite to compatible with ICLR templates. (2) The population size and elite number both use notation $N$.

[1] Zhang, Zhengfei, Yunyue Wei, and Yanan Sui. "An invariant information geometric method for high-dimensional online optimization." 6th Annual Learning for Dynamics & Control Conference. PMLR, 2024.

[2] Yun, Taeyoung, et al. "Posterior Inference with Diffusion Models for High-dimensional Black-box Optimization." Forty-second International Conference on Machine Learning.

### Questions
1. According to Algorithm 1, is there only one gradient update in each sampling iteration?

2. How does ABOM handle the boundary issue, since MLP parameterized policy may generate samples outside the input bound.

### Soundness
2

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
This paper proposes an end-to-end optimization model, ABOM, which employs attention mechanisms to perfrom parent selection, crossover and mutation operators on the population to generate offspring. A loss function based on the distance between offspring and the elite archive is introduced to guide the model to improve the offspring. Experimental results validate ABOM's advantage over traditional BBO and MetaBBO methods.

### Strengths
1. Unlike existing MetaBBO methods based on deep neural networks, ABOM does not require pre-training, saving substantial time and computational resources. Additionally, the model takes the population itself as input and produces offspring in an end-to-end manner, reducing the human effort required to design optimization states and model actions. Although this end-to-end approach is also adopted in other MetaBBO methods such as RNN-OI and GLHF, ABOM's loss function avoids requiring the gradient of the target problem, which appears novel.

2. The end-to-end neural computation enables efficient batched execution on GPUs, as validated in experiments, but it also imposes higher hardware requirements, especially for high-dimensional problems.

3. ABOM demonstrates superior performance compared to BBO and MetaBBO baselines on both BBOB and UAV path planning tasks, confirming its effectiveness.

### Weaknesses
1. The loss function is calculated as the distance between offspring and the elite archive. Given that the search range and problem dimensionality can be large (e.g., [-100, 100] range and 500 dimensions for BBOB in this paper), the scale of the loss and gradients could be large, potentially causing unstable training. Furthermore, the loss function encourages the model to generate offspring that completely surpass the parent population, representing a greedy strategy that may reduce exploration.

2. For each problem instance, model parameters are initialized and updated from scratch. While this enables task-free adaptation, it disregards experience and knowledge gained from optimizing previous problems. Leveraging learned knowledge to enhance current optimization is a key advantage of MetaBBO.

### Questions
1. In Figure 4, the selection often focuses on very few individuals (sometimes only one). Does this imply the model is performing a greedy local search around a limited set of points?

2. Are the population sizes and hidden dimensions sensitive to the problem dimension? Figure 5 (a) & (b) indicate that different functions have different optimal settings (e.g., f4 prefers a larger population while f24 prefers a smaller one). Is it feasible to adapt population sizes and hidden dimensions dynamically to improve performance?

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
This paper introduces ABOM, By eliminating the dependency on handcrafted training tasks, ABOM performs adaptive parameter learning using only optimization data from the target task itself. This enables zero-shot optimization, where ABOM adapts evolutionary operators during optimization to improve performance dynamically. The paper evaluates ABOM on both synthetic benchmarks (BBOB) and a real-world unmanned aerial vehicle (UAV) path planning problem, demonstrating competitive performance across multiple scenarios. The results are promising, showing that ABOM outperforms existing baseline methods.

### Strengths
1. Sound Method Design: The parameterization of evolutionary operators (selection, crossover, mutation) into differentiable, attention-driven modules is well-designed. The online update of optimizer parameters is achieved by using population data generated during the target problem's runtime, with the objective of minimizing the distance between offspring and the elite archive  as the loss function.
2. Theoretical Guarantee (to a degree): The paper provides a convergence proof under idealized assumptions.
3. The experimental evaluation is thorough, and the results are positive.

### Weaknesses
1. The paper needs to explain the differences between the proposed method and other, such as GLHF and B2OPT, in detail, especially regarding the parameterization of evolutionary operators. If prior methods were adopted, proper citations are required.
2. The loss function $\min _\theta||\hat{P}^{(t)}-E^{(t)}||_2$ encourages the offspring population $\hat{P}^{(t)}$ to be close to the current elite archive $E^{(t)}$, essentially using the current local optima to guide optimizer updates. This is a form of "bootstrapping," which introduces search bias and may theoretically lead to getting trapped in local optima.
3. While the authors attempted a theoretical proof of global convergence, it is presented in an overly absolute manner. The resolution of the aforementioned bias relies on persistent exploration via Dropout. However, this theory has a problem: its global convergence is inherently "unbounded," meaning there's no guarantee of achieving convergence within a bounded number of iterations. To illustrate, even a completely random algorithm, when combined with a selection operator, can be proven to converge globally. Therefore, the significance of this convergence proof is limited.
4. Furthermore, if the global optimum lies on the boundary of the search space, or if constraint handling cuts the feasible region into discontinuous parts, the convergence proof will not hold. In summary, attempting to prove convergence is commendable, but the conclusions should not be presented too absolutely or rashly. Further refinement and discussion are needed.
5. This paper effectively performs On-Instance Training. Although there's a meta-behavior of "learning optimization strategies," it lacks cross-task generalization, which deviates from the traditional definition of meta-learning. This needs clarification in the introduction.
6. The limitations of On-Instance Training need to be discussed. For example, a lack of adaptation to cold-start scenarios with limited evaluation counts, and the inability to transfer knowledge between tasks (each task requires independent optimization, unable to leverage similar problems for acceleration).
7. A very detailed parameter sensitivity analysis is needed for On-Instance Training. It's recommended to include the learning rate and dropout rate, as they determine the exploration & exploitation tradeoff and are critical to the proposed method.
8. The experimental section lacks in-depth analysis of results. While the performance of ABOM is compared to several baselines, there is limited interpretation of the underlying reasons for ABOM's success.

### Questions
Refer to the above weakness part.

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
4

### Summary
This paper introduces a novel learning-assisted approach for optimization. The authors propose ABOM framework to address the time-consuming training in existing meta-black-box optimization approach. ABOM inherits the concept of  learnable evolutionary operators from previous GLHF framework, where the iterative reproduction & selection operations in an evolutionary algorithm are abstracted as s group of neural networks, and hence bridge efficient gradient decent with learning for optimization. The most important component in ABOM is its online parameter adaption mechanism, where the up-to-date elite solutions are recorded along the optimziation progress, and at each step, the solution population output by the neural network-based evolutionary operators is self-supervised by minimizing the distance with elite solutions at hand. Through this way, ABOM could adapts itself to novel optimization tasks during the online evolution process, without tailored pre-training, The authors provide an intuitive theoretical proof on the convergence of ABOM and validated ABOM's performance on both synthetic and realistic testbeds. The results show that online adaption could deliver comparable performance to existing MetaBBO approaches, while mitigating the training burden.

### Strengths
I appreciate the novelty of this paper. Since existing MetaBBO approaches require a pre-defined problem distribution and corresponding pre-training to make the meta-level policy generalizable, the idea in this paper (online adaption through self-supervision) enlights more efficient MetaBBOs.

### Weaknesses
1. Theoretical perspective: I have say that though the authors provide a intuitive proof on what they have claimed (Collary 1, 2, Theorem 3.1), a strong assumption (may not happen in real optimization problem) makes me suspect the rationale behind the proof. This assumption is: "the elite solution is ϵ-suboptimal". I wonder how to guarantee such assumption since we face randomized (stocastic) optimization here. This question becomes more obvious when we consider Eq. (10), what if the elite information prematures? What is exact contribution of ABOM to promote the evolution stepping out such premature? Is the crucial component the Dropout in mutation and crossover neural network? Is this indicating that ABOM is actually a random search with adaptive local search?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
