# Efficient Training of Neural Stochastic Differential Equations by Matching Finite Dimensional Distributions

- Decision: Accept (Poster)
- Scores: 5, 8, 8, 3

## Abstract
Neural Stochastic Differential Equations (Neural SDEs) have emerged as powerful mesh-free generative models for continuous stochastic processes, with critical applications in fields such as finance, physics, and biology. Previous state-of-the-art methods have relied on adversarial training, such as GANs, or on minimizing distance measures between processes using signature kernels. However, GANs suffer from issues like instability, mode collapse, and the need for specialized training techniques, while signature kernel-based methods require solving linear PDEs and backpropagating gradients through the solver, whose computational complexity scales quadratically with the discretization steps. In this paper, we identify a novel class of strictly proper scoring rules for comparing continuous Markov processes. This theoretical finding naturally leads to a novel approach called Finite Dimensional Matching (FDM) for training Neural SDEs. Our method leverages the Markov property of SDEs to provide a computationally efficient training objective. This scoring rule allows us to bypass the computational overhead associated with signature kernels and reduces the training complexity from $O(D^2)$ to $O(D)$ per epoch, where $D$ represents the number of discretization steps of the process. We demonstrate that FDM achieves superior performance, consistently outperforming existing methods in terms of both computational efficiency and generative quality.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a novel approach to training neural stochastic differential equations by introducing Finite Dimensional Matching, a new scoring rule designed for continuous Markov processes. The proposed method leverages the Markov property of stochastic processes to reduce the computational complexity of training neural SDEs, with the goal of enhancing both efficiency and generative quality. Theoretical contributions establish that the new scoring rule provides a strictly proper method for comparing two-time joint distributions. Experimental results demonstrate improved performance in training efficiency and generative quality when evaluated against competing methods.

### Strengths
The paper tackles a crucial challenge in training neural SDEs by introducing a new scoring rule that optimizes efficiency for continuous Markov processes. The proposed FDM method is backed by mathematical proofs, providing a strictly proper scoring rule that extends from finite-dimensional distributions to continuous Markov processes. This theoretical contribution is valuable to the literature on neural SDE training methods. Experiments show that FDM offers computational efficiency gains, reducing training complexity from quadratic to linear in the number of discretization steps. The approach outperforms prior methods in computational efficiency, as shown in multiple experimental benchmarks.

### Weaknesses
The paper’s main theorem relies on strong assumptions regarding the Markovian properties and continuity of the processes involved. These assumptions may limit the applicability of the FDM algorithm in more complex, non-Markovian stochastic processes or those with jumps, which are common in real-world scenarios. Explicitly discussing these limitations and potential ways to relax these assumptions would make the contributions more transparent.

Also, the writing template is for ICLR 2024, not ICLR 2025.

### Questions
Please see weaknesses.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a scoring rule for continuous time stochastic processes that is directly derived from a scoring rule on a generic space. They show that this rule is proper (i.e. injective from the space of paths to the space of laws), which is a non-trivial contribution. Experiments show that this method outperforms existing concurrent methods based on signature kernels and SDE-GANs. I stress that the experiments are carried out on a vast array of datasets.

### Strengths
* The paper is well written and structured, making it easy to read and to follow. The introduction is sound and features a thorough literature review.
 
* The main technical contribution is Theorem 2, which allows to convert any scoring rule on a generic space. I believe this contribution to be non trivial and novel, although simple to prove.  

* The experiments are carried out on a vast array of time series datasets and show overall superior performance of the proposed approach. The authors compare themselves to all relevant baselines to the best of my knowledge. Large generative models such as diffusion models are not included ; however, I do believe that they do not belong to the same class of models and do not require a comparable computational budget.

### Weaknesses
* While the contribution made in Theorem 2 is elegant and novel, one could object that it is slightly insufficient --- this is not my point of view, but another reviewer might disagree and I am willing to discuss this point. In order to strengthen their theoretical part, I encourage the authors to consider for instance the sample complexity of their kernel i.e. how fast does the empirical divergence they define through a kernel on $\mathcal{E}$ converge through the expected divergence ? Do sample complexities in $\mathcal{E}$ carry over to the space of paths ? See Gretton (2012) for an example. 

* A blind spot of the paper is in my opinion the choice of the kernel. The authors do not seem to consider other kernels for $\mathbb{R}^d$ valued processes, which could yield an interesting extension. This might especially be interesting since some kernels are sometimes used because of their specific properties (invariances, ...). I would suggest that the authors consider this point, at least in the Appendix. 

* This approach nicely extends to kernels defined on any space - this could be graphs, images, etc. and could allow to generate time series in these spaces. This would provide, in my sens, a extremely valuable extension to the paper. 

* The experimental section is hard to read and a tad unstructured. I encourage the authors to use less tables, add more comments and broaden the analysis of their experiments. Also, a notable restriction of this part is that experiments are only carried out on $2$ dimensional time series. I strongly encourage the authors to extend their experiments to high dimensional datasets. Also, there are no confidence intervals in the tables. 

* Regarding this last point, I believe that a valuable extension could be to consider random feature approximations to the kernels for high dimensional generation, which is still a major hurdle in the field. Similarly, the authors could consider sliced kernels on $\mathcal{E}$ when the dimension is high.  

* Concerning experiments, an interesting task to consider could be the augmentation of a time-series dataset, and the analysis of the gain in performance for any model trained on this dataset. 

* Concerning experiments, I believe that it would be highly valuable to extend applications beyond finance. Generating time series is a major hurdle in many domains with great social impact such as neuroscience, healthcare, biology, climatology, economics ... 

* A valuable extension of this work would be the investigate the use of the devised score for other purposes than training generative models, such as two-sample tests for instance.

### Questions
* Please include confidence intervals in your tables: variability of your results is a very important aspect.

* Could you please include vizualizations of the full generated time series, rather than only plotting the time marginals ?

* Could you add experiments on at least one high dimensional and one non euclidian real-world dataset ?

I would considered increasing my score if a significant number of concerns on the experiments are addressed. Similarly, I could consider lowering my score during the discussion phase if other reviewers relativise the strength of the theoretical contribution --- which again seems sufficient to me.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a very simple yet elegant way of comparing the similarity of the laws of two homogeneous Markov processes. The idea follows from the fact that the distribution of the process is completely determined by the transition kernel. Hence the ability to compare the transition distribution should give enable a comparison between the laws. 

The authors then use this finding as a scoring rule to learn to simulate from a neural SDE, given the observations from a ground truth process. Then, they test their scoring rule and their FDD matching procedure in numerical experiments, showing that their method outperforms existing law matching techniques.

### Strengths
The method is very simple and elegant and the implication of this result, **if true**, could be impactful to the neural SDE and generative modeling community.

**Update: I was wrong about the validity of the proof. The result is sound. **

### Weaknesses
I think there is a significant issue in the proof that could invalidate your main theorem 2. In the middle part of page 14, you have the integral over $t_1,t_2$ of the $S(P_{t_1,t_2}, P_{t_1,t_2}')$ scoring rules being equal, but then you conclude that $$S(P_{t_1,t_2},P_{t_1,t_2}') = S(P_{t_1,t_2},P_{t_1,t_2})$$ a.e.. This is not true or I am missing something? 

**Update: I was wrong about the validity of the proof. The result is sound. **

My intuition is that this is not an easy fix: You want to match the distribution over all $t_1,t_2$, so, in some sense, you want the expectation equality to hold for all test measures $\nu$ instead of just one particular $\nu$. I would be very happy to raise the score and rewrite my review if I am wrong. However, it does seem that proving a result like yours is possible, given that the generator or the resolvent will determine the law of the Markov process (see Either and Kurtz). 

I appreciate the rigor in defining the math notations and results. But the writing and explanations in the paper need improvements. For example, what do you mean by "Update the model parameters $\theta$ through backpropogation to maximize $\hat S$" (inside the algorithm)? Also, there is no explanation of what the data is. What are the "Average KS test scores"? Are you repeating the experiment across multiple batches and producing the percentage of rejection "chance of rejecting the null hypothesis (%) at 5%-significance level on marginal"? 

Other minor issues include: 
1. I think you need $\mathcal{E}$ to be Polish. 
2. The radial basis function (RBF) kernel is not defined. 
3. The $\pi$ notation seems a little distracting, why not just use $(x_t,x_s)$ for $\pi_{t,s}(x)$. 
4. It would be nice to emphasize that the Markov processes you define are homogeneous Markov processes.

### Questions
The main concern I had was in regard to the proof. Please help me to understand or fix the issue. 

If this and the writing issues I raised above can be addressed, I will happily give you at least a 6. However, since I feel that your main result is wrong, I have to give a low score for now.

**Update: I was wrong about the validity of the proof. The result is sound. **

If this (or a variant of the) scoring rule for Markov processes is indeed correct, I think the authors could improve the paper by exploring the sensitivity properties of this scoring rule; for example, what kernel to use and how the score behaves when P and Q are close? Is there a simple formula to compute the gradient?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes a new method called Finite Dimensional Matching (FDM) for training Neural SDEs by identifying a class of strictly proper scoring rules for comparing continuous Markov processes. Using this scoring rule, they claim to reduce the training complexity from quadratic in discretization timesteps to linear.

### Strengths
This paper addresses an important problem: the high computational complexity (quadratic in time steps) associated with training Neural SDEs using scoring rules. The authors propose a reduced complexity method, aiming for linear complexity to enhance performance. They also provide a theoretically grounded approach in designing a new scoring rule. However, despite this strong motivation, the results are somewhat unconvincing due to certain weaknesses noted below.

### Weaknesses
This paper has several notable weaknesses. While the authors propose a reduced complexity approach for training Neural SDEs, they do not adequately explain key concepts, making the paper difficult to follow, especially for readers without a strong background in this area. For example, despite experience with SDEs in score-based generative models and deep learning theory, I found the explanations lacking in detail and context and some results are not convincing.

The authors do not provide sufficient preliminary material on scoring rules or background on how these rules are used to measure divergence between two Markov processes. A concrete example of the scoring rule $s(P, z)$ with an RBF kernel, presented early on, would have improved clarity.

Additionally, the complexity reduction claim is unconvincing. For example, at the beginning, the authors claim they reduce the complexity to linear using $D$ to denote the time steps, however, this notation $D$ is never used again in the rest of the paper. Instead, in Section 4.2 Algorithm, they compare two stochastic processes and use $B$ to denote the total number of time steps. However, the nested summations in the top equation on page 5 suggest quadratic rather than linear complexity, i.e., $B^2$.

Finally, the paper uses the **ICLR 2024 format rather than the ICLR 2025** format.

### Questions
In Theorem 2, does the result hold for any scoring rule $s$, or does $s$ also need to be strictly proper? Could you clarify if there are specific conditions on $s$ required for Theorem 2 to apply?

### Soundness
2

### Presentation
1

### Contribution
2
