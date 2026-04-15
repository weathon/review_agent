# An Invariant Information Geometric Method for High-dimensional Online Optimization

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 1, 5, 1

## Abstract
Sample efficiency lies at the heart of many optimization problems, especially for black-box settings where costly evaluations and zeroth-order feedback occur. Typical methods such as Bayesian optimization and evolutionary strategy, which stem from an online formulation that optimizes mostly through the current batch, suffer from either high computational cost or low efficiency. To strengthen sample efficiency under reasonable computational cost, one promising way is to achieve invariant under smooth bijective transformations of model parameters. In this paper, we build the first invariant practical optimizer framework InvIGO based on information geometric optimization. It can incorporate historical information without violating the invariant. We further exemplify InvIGO with historical information on multi-dimensional Gaussian, which gives an invariant and scalable optimizer SynCMA that fully incorporates historical information with no external learning rate to tune. The theoretical behavior and advantages of our algorithm over other Gaussian-based optimizers are further analyzed to demonstrate its theoretical superiority. We then benchmark SynCMA against other leading optimizers, including the competitive optimizer in Bayesian optimization, on synthetic functions, Mujoco locomotion tasks and rover planning task. In all scenarios, SynCMA demonstrates great competence, if not dominance, over other optimizers in sample efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.

### Summary
This paper propose a new online learning method based on evolutionary strategy. The proposed method is geometric invariant under transformations.

### Strengths
The whole paper is well-presented and easy to follow.

### Weaknesses
see questions.

### Questions
1. The theoretical guarantee of the proposed method seems weak. The theory should be formulated as Theorems based clear assumptions. Current statements in the theory section are vague. 

2. Limitations and further directions should be included in the last section?

3. Theorem 1 looks like just some properties. 

4. Theorem 2/3 looks very vague. 

5. Why use such a long section (section 3) just as an example? Why only focusing studying the more general case?

6. Overall, though I am not familiar with the studied topic, I think combining geometric invariant with ES seems not very significant, since the fisher matrix already has such property. 

7. Do use historic data a standard way in online learning? Does other method also use such technique?



minors:
problem 12 -> problem (12)

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes an information geometric optimization (IGO) method that is invariant under all smooth bijective transformations of the model parameters.

### Strengths
The supplement contains running code (even it the zip file is full of superfluous stuff like MacOS and python caches).

### Weaknesses
Abstract: "Typical methods such as Bayesian optimization and evolutionary strategy, which stem from an online formulation that optimizes mostly through the current batch, suffer from either high computational cost or low efficiency."
I cannot accept this statement. The theoretical analysis of evolution strategies may still be under-developed. Yet, we can clearly say that their sample efficiency is close to optimal, in the sense that it matches a general lower bound on the number of samples that holds for all zeroth-order comparison based methods. I can only conclude that this paper is built on a wrong premise.

In response to the rebuttal: here is a link to a paper with a general lower runtime bound that applies to evolution strategies:
https://inria.hal.science/inria-00112820/file/lblong.pdf
It is very different from the analysis of one-max and other discrete problems.

IGO is built all around invariance principles. Therefore, calling the framework of the paper InvIGO not all but sensible. Furthermore, by design, all IGO algorithms have the desired invariance property in first order approximation. Update steps in high dimensions are necessarily small, hence higher-order differences of steps are tiny. This means that there is simply no margin for improvement over existing IGO methods. Against this background, it remains particularly unclear why the proposed method should be particularly suitable for high-dimensional problems.

I perceive section 2 as entirely chaotic. Equation (3) comes out of the blue. Following equation (5) a bit later, the paper talks about a loss function. However, we are dealing with general objectives, not with statistical data and loss functions. Assumption 1 and Definition 1 seem to be completely disconnected from the previous discussion.

The authors list "it has no external learning rate" as an advantage of their method over CMA-ES. I checked the code and saw lots of magic constants, some of which seem to be taken directly from CMA-ES. Also, I really don't know what that statement is supposed to mean, since the notion of an "external learning rate" does not exist in CMA-ES.

Experimental results: Table 1 is not acceptable. It shows a snapshot for one point in time. I want to see ECDF plots showing the time evolution of performance as they are standard in empirical research in the field (please refer to the COCO/BBOB framework for details). Figure 1 is even worse, since it fails to display function values on a logarithmic scale. Again, this violates all standard in the field, which it is unacceptable. Worse, I have to take this as a clear hint at weak understanding of evolution strategies and their linear convergence guarantees. The term "Near global optimum performance", which is crucial for reading the results, is not defined.

I tried the software provided as a supplement on the well-known Rosenbrock benchmark function in 10d:
  python exp.py --optimizer SynCMA --func rosenbrock --dim 10 --eval_num 100000 --rep 1
The result is that the proposed method fails to improve the initial solution! I tried CMA-ES as an alternative. There, solution quality reaches 8.03345e-06 after about 36000 evaluations and then jumps up to 8.49299e+27! What should I say? Please use the reference implementation of CMA-ES provided by its inventor, which NEVER does this. I can only conclude that the experiments are buggy and cannot be trusted.

### Questions
I don't have any questions that need clarification.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a novel variant of information geometric optimization (IGO) for black-box optimization. This paper addresses some weaknesses of the existing IGO framework: full invariance to the change of parameterization of the sampling distribution (e.g., the algorithmic behavior (slightly) changes whether the covariance matrix is parameterized by its elements or the elements of its Cholesky factor), and the historical information during the search. These limitations are tackled by casting the original objective function of the IGO framework to the minimization of the KL divergence, then introducing the additional KL regularization terms. The resulting algorithm has been instantiated with the Gaussian distribution, resulting in the proposed SynCMA. The proposed approach has been compared to several CMA-ES variants on 10 author-selected synthetic problems and two very simple control tasks.

### Strengths
As claimed in the paper,  a novel variant of IGO that is invariant to the choice of the parameterization of the sampling distribution and incorporates the historical information is proposed. In particular, the way of incorporating the historical information is interesting.

### Weaknesses
The author seem using the term ‘online optimization’ to mean black-box optimization. The term, online optimization, refers to the optimization problem having incomplete knowledge of the future. The use of line optimization in this paper is rather strange and confusing. 

Though theoretically nice to have full invariance to the parameterization of the sampling distribution, its value in practice is limited. First of all, the natural gradient itself is conceptually invariant to the parameterization. In practice, it is not fully invariant as we take a finite learning rate update. However, the difference in the update between different parameterization is relatively small. The goodness of the invariance to the parameterization has not been evaluated in this paper. Moreover, already in the original IGO paper, namely IGO-ML, a variant of the IGO algorithm that is fully invariant to the change of parameterization is proposed. It hasn’t been mentioned in the paper and no comparison has been made. 

Historical information is incorporated into the IGO framework in existing works (*). As far as I can see, their approach, incorporating the historical information as virtual solutions, is invariant to the choice of the parameterization of the sampling distribution as solutions are independent of the parameterization. It has not been mentioned in the paper.

(*) Youhei Akimoto, Anne Auger, and Nikolaus Hansen. 2014. Comparison-based natural gradient optimization in high dimension. In Proceedings of the 2014 Annual Conference on Genetic and Evolutionary Computation (GECCO '14). Association for Computing Machinery, New York, NY, USA, 373–380. https://doi.org/10.1145/2576768.2598258

The authors repeat an incorrect explanation of the original CMA-ES. Firstly on P7, it is stated that the covariance matrix is parameterized as \sigma \Sigma, where \sigma is a scalar and \Sigma is a PDS matrix. This is wrong. In the CMA-ES, the covariance matrix is parameterized by \sigma^2 \Sigma. The same mistake has been made in the description of the experimental setting on P15. Therefore, it is doubtful that the experimental setup is correct and fair.

Moreover, the author says that \sigma is the external learning rate that needs to be tuned. However, it is also completely wrong as \sigma is adapted during the optimization. Indeed, the adaptation of \sigma is the most important component of ES. 

The authors tested the compared algorithms on unimodal and multimodal functions. Especially on multimodal functions, the initial sampling distribution, i.e., the initial mean vector and the initial covariance matrix of the Gaussian distribution, is very important and affects its resulting performance. However, as far as I can see, there is no explanation for it, and because of the above mentioned incorrect explanation of \sigma, it is rather unclear whether a fair comparison has been done. As all the test problems are shifted to have the global optimum at the origin, wrong settings would easily lead to unfair (or meaningless) comparison. 

The hyperparameter setting does not seem to be fair. Though it is not explicitly written in the paper, I suppose that the authors uses the default hyper-parameters for the CMA-ES variants as they are not meant to be tuned for each problem instance. On the other hand, because the comparison are done on a single problem instances for each problem, I suspect that the parameters are tuned for these problem instances. If so, it is unfair. To avoid such a (possibly non-intended) unfair hyper-parameter setting, I strongly recommend the authors to evaluate the proposed approach on COCO framework, where 15 problem instances are generated and used for each of 24 functions.

### Questions
Why don’t you just use COCO framework (aka BBOB testbed) to compare algorithms?

Which implementation of the existing approaches have been used for the experiments?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors introduces an optimization framework relying on information
geometry, in particular natural gradient. The proposed framework allows
for real reparameterization invariance, usable in practice. Moreover, it
provides a learning rate-free setup, with some theoretical guaranties.

### Strengths
- Trying to tackle the practictal difficulties of IGO.
- Some code is provided, but I did not tried to run it.

I am not familiar enough with the optimization literature to comment on
the experimental part.

### Weaknesses
Although the text by itself is gramarly and syntactically correct,
overall the paper is difficult to follow. The transition are rather
abrupt and not a lot of explanation are given at each step.

The use of two names, InviGO and SyncCMA, is confusing as it mades
difficult to seen the boundaries between the two methods.

The justification on the absence of a learning rate is not sufficient,
or not clear enough. There are also a lot of hyper-parameters in the
algorithm.

Except in the title, there is no mention of the words "high-dimensional"

The chosen name, InvIGO (I guess the "inv" states for invariant ?), is
surprising since that it's the fundation of IGO to rely on invariance.

The complexity analysis is rather artificial and useless.

### Questions
I understand that InviGO is more generic than just Gaussian
distributions, but I have the feeling that the separation with SyncCMA
is quite artificial, don't you think it should be clearer to derivate
the full optimization algorithm in a straighforward way ?

At the end of Section 3, paragraph Limitation and future direction, what
is the suggested strategies to adapt the constants during the course of
the iterations ?

Could you clarify the statement in the title about the high dimension ?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor
