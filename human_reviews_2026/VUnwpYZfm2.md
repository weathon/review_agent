# Exploring the Boundary of Diffusion-based Methods for Solving Constrained Optimization

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Diffusion models have achieved remarkable success in generative tasks such as image and video synthesis, and in control domains like robotics, owing to their strong generalization capabilities and proficiency in fitting complex multimodal distributions. However, their full potential in solving Continuous Constrained Optimization problems remains largely underexplored. Our work commences by investigating a two-dimensional constrained quadratic optimization problem as an illustrative example to explore the inherent challenges and issues when applying diffusion models to such optimization tasks and providing theoretical analyses for these observations. To address the identified gaps and harness diffusion models for Continuous Constrained Optimization, we build upon this analysis to propose a novel diffusion-based framework for optimization problems called DiOpt. This framework operates in two distinct phases: an initial warm-start phase, implemented via supervised learning, followed by a bootstrapping phase. This dual-phase architecture is designed to iteratively refine solutions, thereby improving the objective function while rigorously satisfying problem constraints. Finally, multiple candidate solutions are sampled, and the optimal one is selected through a screening process. We present extensive experiments detailing the training dynamics of DiOpt, its performance across a diverse set of Continuous Constrained Optimization problems, and an analysis of the impact of DiOpt's various hyperparameters.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper explores constrained optimization in generative diffusion models. The main problem that the paper tackles is the fact that a diffusion model may end up generating samples outside of the feasible region due to the inherent stochasticity of the generative process. This is indeed an important problem that warrants a rigorous scientific investigation and the paper is timely. To address the problem, the paper proposes DiOpt, which introduces a weighted bootstrapping method to self-supervise the model to generate samples within the feasible region. The idea, as I understand, is to generate samples ("candidate points") during training and weigh them according to constraint violation (feasible points get positive weights and infeasible points negative weights); The diffusion model loss term (squared error of predicted noise) is then weighed accordingly and used for training the diffusion model.

### Strengths
The paper addresses an important problem for the community and has the potential to help us overcome some of the current limitations we are facing with diffusion models. The proposed weighting scheme is somewhat new, and the method is validated against other methods on several benchmarks.

### Weaknesses
I guess my biggest problem is with the scientific contribution. First and foremost, even though the paper claims to address the hard-constrained generation problem in various places, such as Table 1 and the intro section, in my opinion, this is still a soft-constrained generation problem. If I understood correctly, the model is simply "discouraged" from generating infeasible samples via the bootstrapped weight values during the training stage. I don't find anything that actually hard-constrains the generated samples to be within the feasible region, and their experiment data also shows that (e.g., the first column of Table 2).

However, even aside from the discussion of whether it is a hard-constrained method or not, I don't know if the paper contains a sufficient amount of new scientific knowledge or methodological breakthroughs. The main idea of the DiOpt method is to weigh the loss function via bootstrapped samples so that the generation of feasible samples is promoted, whereas infeasible samples are suppressed. However, considering the high scientific standard of ICLR, I don't know if this is something I would call an "ICLR-worthy" idea. Also, the mathematical analysis in Section 4 on "why ... diffusion models encounter infeasibility issues in constrained optimization," which is argued as a contribution of this paper, is rather underwhelming: It defines feasibility in **linear** programming mostly, and there are a couple of sentences showing an asymptotic bound on the probability of generated sample existing in the feasible region unfortunately without any proofs or rigorous dicussions. So again, I don't know if this is an ICLR-worthy contribution to the scientific community.

Furthermore, overall, the presentation can be improved significantly. First and foremost, the scientific contribution of this work is unclear. Both the intro and the abstract could benefit from jumping straight to the exact problem that the authors want to address, how they propose to solve it, and what contribution to the scientific community they are presenting. Also, the bullets in the intro, which are supposed to summarize their contributions, don't really do justice. That's because these bullets mostly talk about "what" they do, instead of "why" it is important. In its current form, the contribution seems to be very narrowly positioned to the development of this very specific recipe that employs bootstrapping to discourage the model from creating infeasible samples.

The paper could also benefit from more high-level, intuitive explanations of the core scientific idea. Although my research isn't characterized as optimization research, I teach a course on that topic and would claim that I am fairly familiar with the technical content. Despite that, it required me several readings to grasp what the core scientific idea was.

### Questions
- Is DiOpt limited only to linear, convex optimization problems, or can it generalize to more complex non-convex optimization problems as well?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper apply diffusion model on learning to optimize problem to solve constrained optimization problem. Compared to standard supervised diffusion model, which produces infeasible in high dimensional problems, their proposed DiOPt method can hold the constrains better thanks to the combination of supervised and self supervised learning. In self supervised learning stage, a weighted loss is applied to bootstrap feasible samples. The design yields better feasibility and optimality trade-off compared to baselines.

### Strengths
- the bootstrapping self-supervised training can effectively bias the sample towards the feasible region and is compatible with different diffusion formulations
    
- the proposed method is empirically validated on diverse constrained optimization problems with detailed ablation

### Weaknesses
- the method introduces extra hyperparameters including weight, reset frequency and lookup updates. It would be beneficial if author and provide heuristics on how to choose those hyperparameters and how sensitive the algorithm are to those parameters.
    
- the author only report evaluation time in the main text. It would be helpful to also report training time for different method to see the overhead introduced by two-stage training.

### Questions
- can DiOPT handle equality constraints?
    
- how sensitive is DiOPT to parameters like rs and Kt in the paper?
    
- in standard constrained optimization, Lagrangian multiplier is employed to control the penalty level based on violation. is it possible to use the same practice to make the weight update for constrains less rely on heuristics?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces DiOpt, a training framework enhancing the capability of diffusion models for solving optimization problems. The method introduces sample weighting functions to bias the training towards samples which have lower constraint violation and regret. The evaluation is conducted across several interesting problems, most notably the ACOPF setting, which is a challenging and highly nonconvex real-world application.

### Strengths
- **Empirical Analysis:** The evaluation includes several different optimization problems, the majority of which require adherence to nonconvex constraints sets. In particular, I appreciate the results on the ACOPF settings, which are considered to be an important problem; I believe these experiments add real-world significance to the results.

- **Methodological Simplicity:** The method proposed is fairly intuitive, and the analysis motivating its adoption sets this up well. As the results are strong, the simplicity of the approach should be considered a strength of the work.

- **Exposition:** The paper is easy to follow, and positions itself well within the broader literature.

### Weaknesses
- **Novelty:** I have some concerns regarding the novelty of the method, considering the similarity to [1]. While the overlap methodologically leads me to view this work as more of an application paper, it is not currently presented this way. I believe this work could differentiate itself better by leaning further into an exploration the specific weighting function used for this domain. From a theoretical perspective, the analysis surrounding the weighting scheme is fairly limited, and, in its current form, this appears to be closer to a heuristic than a grounded rule. Some of the ablations in the appendix do help strengthen the authors' case, and it would be useful if these could be alluded to better in the main paper.

- **Clarifications of Experiments:** The exposition here seems to be the weakest, in part because little space is dedicated to it. Many key details are buried in the appendix. For example, in the appendix it seems that the QP experiment is convex; if this is the case, why does DC3 report such high violations -- I'd expect these to approach zero, based on the description. Can the authors speak to this? 

- **Additional Baselines:** It would be interesting to see how the performance compares to methods that integrate hard constraints. Of course, formal guarantees cannot be provided on the nonconvex settings considered, but it would be of interest to provide results on other methods from Table 1.

- **Out-of-Distribution Testing:** Has any analysis been given to OOD settings? I assume that in all settings the model has been trained on the same distribution (e.g., same constraints and objective) as it is tested on. Could conditioning be used to encourage generalization?

---

[1] Ding, Shutong, et al. "Diffusion-based reinforcement learning via q-weighted variational policy optimization." Advances in Neural Information Processing Systems 37 (2024): 53945-53968.

### Questions
See questions in weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors discuss solving constrained optimization problems with a generative diffusion model. They show that training the model on a dataset of solutions is not adequate, with the sampled solutions regularly being outside the feasible space. Thus, they propose an alternate training scheme, DiOpt, that mixes "supervised" training with solutions and "unsupervised" training with random points, weighted by their objective value, to steer the diffusion model towards generating samples in the feasible region. The experiments show that the proposed method can be used as a learned solver on a set of constrained optimization tasks.

### Strengths
- The idea of training a diffusion model in an "unsupervised" way, i.e., without a ground truth dataset of samples (solutions), is very interesting and has not been explored before to the best of my knowledge. The authors effectively show that with appropriate weighting, they can train the diffusion model to generate samples in the desired region, which in this case, corresponds to the feasible space of solutions to the optimization problem. This could be a significant contribution both to the learned optimization and the overall diffusion crowd.

- DiOpt can effectively produce solutions to the constrained optimization problem, whereas a naive approach of training the diffusion on solutions only fails. Additionally, when compared to two previous baselines, DiOpt consistently achieves results closest to an optimization solver, while only requiring a fraction of the time.

### Weaknesses
- The baselines to which the proposed method is compared are not well established in the main text, making it difficult to interpret the results of Table 2. The main result of the paper in Table 2 requires the reader to know how the two baselines (DC3, MBD) work to get a clear picture of the advantages of the proposed algorithm. It seems that DC3 trains a network to perform the optimization, whereas MBD runs some kind of solver within its algorithm (and seems not to work at all). Thus, apart from establishing that DiOpt achieves better results, the experiment does not really provide any information related to what the advantages or critical components of DiOpt are. 

- The authors, throughout the paper, compare the proposed DiOpt to the naive approach of training the diffusion model in a supervised way, but do not include results of the "naive diffusion" approach in the main table. Some of those results are found in the appendix ablations. Table 2 should include an additional row with the naive approach to establish a baseline of what the problems of training the diffusion without DiOpt are.

### Questions
- You mention six related learned constrained optimization methods from the literature, but end up only comparing to two. Are the other methods not applicable to the settings you have applied DiOpt to, and if not, could you apply DiOpt to their settings?

- What is the training time of DC3 in Table 2, and what is the training time of DiOpt? Additionally, MBD seems to be an inference-only method, meaning that it does not require any training at all. Is the comparison fair in that case?

- Is the issue of the naive diffusion approach the amount of training data? If you had a large enough dataset, would you observe the same issue as shown in Figure 2?

- In Algorithm 1, should line 333 be `if n mod 2 == 0 then`? Does the reset of Equation (11) happen for both infeasible and feasible ($\omega = 0$) points?

### Soundness
3

### Presentation
3

### Contribution
4
