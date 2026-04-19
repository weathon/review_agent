# Differentiable Solver Search for fast diffusion sampling

- Decision: Reject
- Scores: 5, 6, 8

## Abstract
Diffusion-based models have demonstrated remarkable generation quality but at the cost of numerous function evaluations. Recently, advanced ODE-based solvers have been developed to mitigate the substantial computational demands of reverse-diffusion solving under limited sampling steps. However, these solvers, heavily inspired by Adams-like multistep methods, rely solely on t-related Lagrange interpolation. We show that t-related Lagrange interpolation is suboptimal and reveals a compact search space comprised of timestep and solver coefficients. Building on our analysis, we propose a novel differentiable solver search algorithm to identify the optimal solver. Equipped with the searched solver, our rectified flow models, SiT-XL/2 and FlowDCN-XL/2, achieve FID scores of 2.40 and 2.35, respectively, on ImageNet-$256\times256$ with only 10 steps. Meanwhile, our DDPM model, DiT-XL/2, reaches a FID score of 2.33 with only 10 steps. Notably, our searched solver outperforms traditional solvers by a significant margin. Moreover, our searched solver demonstrates its generality across various model architectures, resolutions, and model sizes.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper proposes a method for accelerating reverse diffusion. State-of-the-art models solely rely on the time variable to interpolate and reverse diffuse. The proposed approach builds on the Taylor expansion on top of which the Adams-Bashforth is built around x and not only t in order to improve the search performance. Authors elaborate on the theoretical grounding of their approach and show results on a few benchmarks.

### Strengths
The idea of relying on x in addition to t to expand the search space seems very natural.

### Weaknesses
The paper's writing is an obstacle for the reader to access the work. The number of typos is too large for me to report them here. There are numerous sudden jumps in the text which miss any logical connectors. Also too many of these to start reporting them. 

The analysis in Eq. (7) --> (24) is interesting but it is hard to follow as it is written in a semi-narrative style. It may help to rephrase it as a theorem (state the final result) and the analysis would be the proof of the result.

### Questions
What is the computational complexity of the proposed approach and how does it compare to existing methods?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies diffusion algorithms for generating images. The authors propose a novel differentiable solver search
algorithm to build better diffusion solvers. Specifically, the authors demonstrate that the upper bound of discretization error in reverse-diffusion ODE is related to both timesteps and solver coefficients and define a compact solver search space. Then, a differentiable solver search algorithm can be designed to make better diffusion models. The authors conduct experiments compared with current state-of-the-art methods. They show that the proposed DiT-XL achieves 2.33 FID under ten steps, beating current best methods by a large margin.

### Strengths
1. The authors propose a novel differentiable solver search algorithm to build better diffusion solvers. Specifically, the authors demonstrate that the upper bound of discretization error in reverse-diffusion ODE relates to both timesteps and solver coefficients and defines a compact solver search space. 

2. The experimental results seem great compared with current state-of-the-art methods.

### Weaknesses
FYI: Since I am not working in this area, my reviews may be biased (or even wrong) in a large probability. In general, I found the experimental results to be excellent, and the proposed method seems simple and elegant. I will lean to accept but keep open during the discussion period.

1. The concern of the error bound analysis in Section 4.3: First of all, there are some typos; these $x$ and $\hat{x}$ should be bold. I lost in Equ. (22), should be $||$ be $\| \|_2^2$. The bound provided in Equ. (24) is meaningless to me. It could be helpful to discuss this further. I feel that the authors want to make their method theoretically sound, but it goes in the opposite direction... Even if the authors claim the method is optimal, the algorithm derivation is largely empirical. (Can you justify why the method is optimal? From my understanding, the method should at least match an existing lower bound for the problem.) So, the authors may prefer to keep it as it is.

2. What is $\eta$ in Section 4.3?

3. Section 5 provides algorithms 1 and 2, the proposed differentiable method for solving ODE. This kind of configuration reminds me of some typical extrapolation methods for solving ODE. For example, Richardson's extrapolation for solving ODE forms a kind of table; the method will converge to ODE in a very efficient way. If possible, please discuss this.

### Questions
See the weakness.

### Soundness
3

### Presentation
4

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
This paper addresses the inefficiencies in diffusion models for image generation, which require numerous denoising steps during inference. The authors present several key contributions:

1. The authors demonstrate that the choice of interpolation function in the reverse-diffusion ODE can be reduced to mere coefficients, which simplifies the error minimization process related to discretization.

2. The authors propose a novel algorithm that identifies optimal solver parameters within a compact search space defined by timesteps and solver coefficients, enhancing the performance of pre-trained diffusion models.

3. Utilizing their algorithm, they achieve state-of-the-art (relative to a selection of methods) results on ImageNet from 5 to 10 sampling steps.

### Strengths
# Content

1. The paper critically revisits Adams-like multistep methods and highlights their limitations specifically in the context of diffusion models. 

2. The derivation of error bounds and the use of Cauchy-Schwarz inequalities to establish relationships between error, solver coefficients, and timestep choices demonstrate a rigorous mathematical approach.

3. By proposing a universal interpolation function $\mathcal{P}$ without an explicit form and focusing on coefficients rather than fixed interpolation methods, the paper opens new avenues for flexibility in solver design. This could lead to more adaptable and potentially more accurate methods in sampling the reverse diffusion process.

4. The introduction of a differentiable solver search algorithm provides a novel way to optimize timesteps and coefficients. This approach could leverage pre-trained models, possibly leading to improved performance in practical applications.

5. The paper's focus on error bounds related to pre-trained velocity models is valuable, as it acknowledges the imperfections in real-world applications and provides a framework for quantifying these errors.

### Weaknesses
# Presentation (Minor)

I marked the Presentation as poor. The reason for this is that, to my liking, the equations are not properly embedded into the text and there are too many prominent typos.

Please improve your usage of punctuation in and surrounding equations. Furthermore, 29 enumerated equations in the main paper, of which many are not referenced, can be considered excessive. Detailed derivations could be moved to the appendix, shifting the focus to the core functions of your method and leaving more space for figures 4 & 5 (e.g. allowing for larger text within the figures), and algorithms 1 & 2. This could drastically improve the presentation of your work.

To further improve the presentation of your work, please also check for typos, like in the title of section 3, Eular vs. Euler, etc..

# Content (Major)

1. The emphasis on optimizing solver coefficients based on small data (50K in the experiment section) raises concerns about overfitting. While the expectation of coefficients is meant to enhance generalization, the process must be carefully managed to ensure robust performance across varied datasets.

2. The Paper does not feature any other metrics than FID. 

3. While the paper suggests state-of-the-art performance, its experiments and comparisons appear selective. It is important to compare it to other methods that could potentially outperform your method as well. Otherwise, the reader has no perspective regarding the limitations of your approach.

4. The Paper also does not discuss limitations w.r.t. how well the solver algorithm scales to smaller or larger amounts of samples. Furthermore, all evaluation was stopped at 5 solver steps.

### Questions
In general, I am willing to raise my score if my questions and concerns are addressed with compelling evidence.

Concerning the aforementioned weaknesses, I pose the following questions:

1. The paper features FID as its only metric. Can you incorporate more metrics, such as e.g. Improved Precision & Recall, as well as Inception-Score?

2. How long does it take for Algorithm 2 to complete in theory? O-Noation w.r.t. network evaluations, samples and solver steps should be featured in your paper.

3. You used 50K samples for Algorithm 2 in your experiments section, can you add an ablation study for the cardinality of the samples used to solve your coefficient search? (e.g. 10K, 50K, 100K & 1M samples)

4. You stopped your evaluation at 5 Steps, how much do scores deteriorate for 1 to 4 steps, can you add an additional ablation study for less than 5 solver steps?

5. While your evaluation in Tables 1 & 2 suggests your method outperforms competing methods, how does your work compare to Distillation Methods, such as Consistency-Distillation Training, which yields methods that require less than 5 solver steps? Such comparisons should be featured to put the performance of your method into perspective relative to the state-of-the-art for efficient solving techniques of the reverse process.

6. How do you explain the 10-step solver outperforming 50 Euler steps in Figure 5 (c), what scores would your method reach for 50 steps? I kindly ask you to evaluate more than one metric (see 1.).

7. How well does your method work across different variance schedules? Can variance schedules be identified, where your method works better or worse? Does your method perform better on diffusion processes where the forward process is driftless (e.g. VE) or forward processes that do not omit the drift function (e.g. VP)? 

8. Can you add an evaluation of your text-to-image experiments that is based on metrics rather than visual impressions?

Overall I kindly ask you to rework your paper's presentation and add a more rigorous evaluation with more metrics than FID, measuring the diversity- and fidelity of samples.

### Soundness
2

### Presentation
2

### Contribution
3
