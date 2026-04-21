# PIED: Physics-Informed Experimental Design for Inverse Problems

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 8

## Abstract
In many science and engineering settings, system dynamics are characterized by governing partial differential equations (PDEs), and a major challenge is to solve inverse problems (IPs) where unknown PDE parameters are inferred based on observational data gathered under limited budget. 
Due to the high costs of setting up and running experiments, experimental design (ED) is often done with the help of PDE simulations to optimize for the most informative design parameters (e.g., sensor placements) to solve such IPs, prior to actual data collection. This process of optimizing design parameters is especially critical when the budget and other practical constraints make it infeasible to adjust the design parameters between trials during the experiments.
However, existing experimental design (ED) methods tend to require sequential and frequent design parameter adjustments between trials. Furthermore, they also have significant computational bottlenecks due to the need for complex numerical simulations for PDEs, and do not exploit the advantages provided by physics informed neural networks (PINNs) in solving IPs for PDE-governed systems, such as its meshless solutions, differentiability, and amortized training. 
This work presents Physics-Informed Experimental Design (PIED), the first ED framework that makes use of PINNs in a fully differentiable architecture to perform continuous optimization of design parameters for IPs for one-shot deployments. 
PIED overcomes existing methods' computational bottlenecks through parallelized computation and meta-learning of PINN parameter initialization, and proposes novel methods to effectively take into account PINN training dynamics in optimizing the ED parameters. 
Through experiments based on noisy simulated data and even real world experimental data, we empirically show that given limited observation budget, PIED significantly outperforms existing ED methods in solving IPs, including for challenging settings where the inverse parameters are unknown functions rather than just finite-dimensional.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper solves the one-shot experiment design using PINNs in both forward and inverse problems. It overcomes computational bottlenecks by parallelism and meta-learning of initialization. The experiments on both synthetic and real-life datasets show the performance improvements.

### Strengths
1. this paper joins many advanced techniques together to solve the problem.
2. most of the representation is clear, with extensive experiments and details.

### Weaknesses
1. The overall algorithm flow is unclear in the main text. Figure 1b is too general and missing almost all of the details of the technique.
2. Does not compare with existing NN-based experiment design methods.
3. Almost all techniques are adopted from existing literature.

### Questions
1. What's the relationship between the 'PINN for ED' and the 'adaptive-sampling PINN for inverse problem'?
2. Can a trained PINN generalize to a new ED problem with new $\beta$? Can the model (including the meta-learning part) generalize to a new distribution of $\beta$？

### Soundness
3

### Presentation
3

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
The authors developed a physics-informed experimental design approach for the inverse problems to find the optimal and most informative design parameters. The paper is well-organized.

### Strengths
The theoretical analysis of PINN is thorough.

### Weaknesses
The effect of some experiments is not significant.

### Questions
1. In Fig. (c) and Fig. (d), the training loss and test error at final stage are similar for both methods. How can this be used to highlight the advantages of the proposed method.
2. How to define the value of μ in Eqs. (17) and (18). It is a very important parameter to determine the difficulty for solving the NS equations.
3. How to select the noise variance for different experiments. The noise variance may have a great influence for the prediction results. Can the method still be available when the noise variance is very large.

### Soundness
3

### Presentation
3

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
This paper suggests PIED, a method for optimal experimental design for PDE inverse problems via Physics-informed neural networks (PINN). The PIED framework consists of three steps: 1. A PINN to learn a forward simulator that maps parameters $\beta$ to solutions of the PDE $u_\beta$. 2. An observation selector that returns noisy predicted observations $\tilde Y= u_\beta(X) + \epsilon$ at a fixed number of sensor placements $X$. 3. A PINN inverse solver that predicts the ground truth parameter $\hat \beta(X, \tilde Y)$ from $X$ and $\tilde Y$. 4. The optimal sensor placements $X$ are found by minimising the MSE between $\beta$ and $\hat \beta(X, \tilde Y)$ which requires back-propagation through the inverse problem solver. The paper claims the following contributions and innovations: 
- The weight initialisations of all PINN-based components are shared and from a pretrained model which stabilises and accelerates training.
- The forward simulator for various choices of $\beta$ are learned in parallel
- The mean square error between $\hat\beta(X, \tilde Y)$ and $\beta$ is approximated by learning the inverse solver with a smaller number of training steps (FIST criterion for ED)
- The mean square error between $\hat\beta(X, \tilde Y)$ and $\beta$ is approximated through a linearisation of the PINN training dynamics (MoTE criterion for ED)

The framework is tested on optimal experimental design for sensor placement on Eikonal, Wave, and Navier-Stokes equations as well as groundwater and cell growth dynamics, and benchmarked against Bayesian experimental design, random sensor placements, and sensor placements on a grid.

### Strengths
- Overall, the paper is well written and seems methodologically sound.
- The paper addresses a challenging experimental design problem with little existing approaches.
- There is sufficient experimental results to support to proposed approach.
- The paper combines an interesting set of ideas: 
	- Meta learning and shared weights for more efficient PINN training. 
	- Approximate training dynamics through of the inverse solver to make back-propagation through the inverse solver feasible.

### Weaknesses
- The approach is mostly motivated by the comparison with classical simulators, but other types of neural surrogate models for the forward simulator are not mentioned. This is probably because most such approaches like Fouier Neural Operators don't investigate the ED downstream task. Nevertheless, this made me wonder how much this approach actually relies on Physics-informed neural networks. For example, the processing in parallel threads would be typical for all approaches that target ED with a neural network.
- The description of the algorithm/framework is a little inconsistent. The PIED framework in Figure 1 suggests that the PIED framework is trained end-to-end, while Algorithm 3 in the appendix clarifies that training proceeds in three phases that are not interleaved: Optimising for initial weights,  training to define ED criteria, optimisation of ED criterion. I wish that this was clearer from the main text.

### Questions
- I would suggest to describe the PIED framework as described in algorithm 3 more clearly in the main text of the paper
- After the experimental design is complete and sensor placements $X$ are obtained: Are the obtained sensor placements supposed to be universally informative and optimal for various downstream tasks of the practitioner, also those that involve classical simulators, or is the intention to always perform inference of the parameter $\beta$ with the trained inverse PINN (which may be inaccurate compared to a slower classical solver). 
- Related to the previous question: Is it possible that the PINN memorises information about the PDE system at arbitrary evaluation points through the shared initialisation $\theta_{SI}$, such that the sensor locations $X$ are not the most informative to make predictions about the system, but only the most informative when we additionally have access to a PINN with parameters $\theta_{SI}$. In other words, could it be that the optimal experimental design is different when one wants to use classical simulators + MCMC to solve the inverse problem?

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
2

### Summary
The paper proposes a novel algorithm for solving the inverse problem in PDEs - that is, estimating the parameters of a PDE governing the dynamic characteristics of a system.  In particular, the paper assumes that (a) obtaining (x,y) observations of the system is expensive, so we must select our observation points carefully; and (b) observations require an initial setup that we cannot (practically) repeat, so we must specify our test points up-front and not dynamically as in e.g. Bayesian Optimization.

To tackle this problem the paper suggests physics-informed experimental design (PIED) that uses two sets of PINNs to select appropriate test points for a given system of PDEs (with a-priori unknown parameters).  The general approach uses a set of PINNs operating as forward simulators to generate functions satisfying the PDEs, sampling these for a set of points X, then using PINNs as inverse solvers to estimate the PDE parameters from the observations.  The efficacy of the points X is measured as the difference between the "real" PDE parameters (used in the forward simulators) and the corresponding estimated estimated PDE parameters.

### Strengths
The algorithm is certainly interesting.  The PIED framework certainly looks practical.  The motivation behind and justification of each step is presented, and the experimental results look good.

### Weaknesses
One point that needs to be address in the paper is that of computational cost.  After all, one of the motivations of using PINNs in parallel is computational efficiency, so it would be good to have a comparison in terms of same.

### Questions
See previous.

### Soundness
3

### Presentation
3

### Contribution
3
