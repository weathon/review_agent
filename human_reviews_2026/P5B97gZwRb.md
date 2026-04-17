# Hyperparameter Trajectory Inference with Conditional Lagrangian Optimal Transport

- Decision: Accept (Oral)
- Scores: 2, 6, 4

## Abstract
Neural networks (NNs) often have critical behavioural trade-offs that are set at design time with hyperparameters—such as reward weights in reinforcement learning or quantile targets in regression. 
Post-deployment, however, user preferences can evolve, making initial settings undesirable, necessitating potentially expensive retraining. 
To circumvent this, we introduce the task of Hyperparameter Trajectory Inference (HTI): to learn, from observed data, how a NN's conditional output distribution changes with its hyperparameters, and construct a surrogate model that approximates the NN at unobserved hyperparameter settings. 
HTI requires extending existing trajectory inference approaches to incorporate conditions, exacerbating the challenge of ensuring inferred paths are feasible.
We propose an approach based on conditional Lagrangian optimal transport, jointly learning the Lagrangian function governing hyperparameter-induced dynamics along with the associated optimal transport maps and geodesics between observed marginals, which form the surrogate model.
We incorporate inductive biases based on the manifold hypothesis and least-action principles into the learned Lagrangian, improving surrogate model feasibility.
We empirically demonstrate that our approach reconstructs NN outputs across various hyperparameter spectra better than other alternatives.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a technique for performing hyper-parameter trajectory inference (HTI), a problem where one introduces a model $\hat{p}(y|x,\lambda) \approx p\_{\theta\_{\lambda}}(y|x)$, which eases the adjustment of $\lambda$ at inference time. To that end, the authors propose using Lagrangian Optimal Transport (Villani, 2009; Pooladian et al., 2024) to build their model $\hat{p}$.

### Strengths
There are 2 main strong points with this submission.

First, the authors do a nice link in their experiments section, with their motivations in the introduction. The reasoning for why one needs hyper-parameter inference are also clear and well motivated.

Second, there is a good variety of important tasks for which the proposed method applies. In all tasks the authors show an improvement over other methods, sometimes with a significant margin.

### Weaknesses
Overall, I think this paper has major organization issues:

1) Section 4 is way too short to merit being a section on its own (2 paragraphs), and I think the authors define the problem they are treating way too late in the paper. In my view, this should be done as early as possible (e.g., in the beginning preliminaries section).

2) With `1)` in mind, the preliminaries section seem disconnected with the problem statement. It could be nice to highlight how conditional OT/lagrangian OT relate to the HTI task.

3) The paper seems to have many interesctions with the work of (Pooladian et al., 2024). For instance, they use the same amortization procedure and metric parametrisation. With that in mind, it would be nice to have a comparison with their method, highlighting the improvements over previous art.

4) About section 5 in general, the fact that the main algorithm is only shown in the appendix harms readability. Furthermore, the authors don't present the final objective function/learning problem being optimized.

### Questions
The hyper-parameter trajectory notion seems a bit contradictory to me. For instance, usually hyper-parameters are fixed during training, so they don't really evolve. From what I got from the text, the authors are trying to model the mapping $\lambda \mapsto p\_{\theta\_{\lambda}}$ (see the intro) but in my view this lacks a temporal structure, i.e., $\lambda(t), t \in [0 , 1]$ to warrant the nomenclature "trajectory inference". Could the authors elaborate on this idea?

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
This paper studies the problem of hyperparameter optimization for training neural networks. To do this, the authors develop a conditional Lagrangian optimal transport framework that allows them to learn a Lagrangian for problems of interest. The authors demonstrate that the learned Lagrangian can accurately model hyperparameter landscapes across examples, demonstrating the utility of their method.

### Strengths
- The paper does a good job of setting up its methodology, with clear explanations of each piece. 
- The problem formulation developed here seems new and is well motivated.
- The use of learning a conditional Lagrangian Optimal Transport model is novel and elegant.
- The authors give a range of experiments that demonstrate the ability of this method to learn hyperparameter trajectories.

### Weaknesses
- It is hard to see what the real use of this method will be, and the message of this gets somewhat lost in the experiments.  While this is a cool idea, what are the applications in which this will be most relevant, especially when one doesn't know a priori how chaotic the trajectories will be?
- There are no baselines that this work is able to compare against.
- While the results indicate that the method can do a good job of modeling trajectories with proper tuning, I am missing what insights we gain from these trajectories. Rather than just chasing performance, can the authors comment on interesting findings in what is actually learned?

### Questions
- Rather than just empirical evidence across examples, can the authors give a theoretically motivated way to choose which method to use? In other words, how do we guide our choice of conditional Lagrangian? Is there an automatic way to do this?
- Does the method do a good job of extrapolating outside of the range of learned hyperparameters? Having a clear discussion of interpolation versus extrapolation in trajectory inference like this is useful.
- I cannot get a sense of how scalable this method is. How expensive is this to run in really large scale ML settings?
- Can the authors combine their work with Bayesian optimization formulations to yield a way to choose optimal hyperparameters rather than just modeling across a range of hyperparameters?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper defines Hyperparameter Trajectory Inference (HTI): given a neural model whose behaviour changes with a continuous hyperparameter, learn a surrogate that interpolates the model’s output distributions between a few trained anchor values. To do this, the authors extend neural Lagrangian optimal transport to the conditional setting: they learn, from sparse observations, a conditional Lagrangian and amortised CLOT maps/geodesics so that inferred paths stay low-action and remain in dense regions. They validate this on a 2D toy task, on two RL surrogates (cancer therapy, Reacher), and on quantile regression, where their full model (learned metric + density potential) is consistently better than ablations and the CFM/direct surrogates.

### Strengths
- HTI is a useful abstraction: many hyperparameters in practice (reward weights, discount factors, quantile levels, robustness coefficients) define families of policies or predictors, but current practice trains only a few points on that curve. Framing this as conditional trajectory inference and tying it to OT provides a principled way to discuss “hyperparameter-induced dynamics.”
- The authors do not simply make their approach learn pairwise conditional OT maps. It learns the cost itself via a conditional Lagrangian, with two inductive biases, and the Lagrangian is the device that lets them force the inferred paths to stay plausible when data are sparse.
- The experiments section seems to confirm that the approach enables learning from a few, sparse observations...

### Weaknesses
- ... but all (few) real evaluation tasks are fairly low-dimensional on the output side, forecasts with short horizons. The method is sold as applicable to “complex and higher-dimensional geometries,” but the experiments do not necessarily convey this statement.
- The RL scenarios considered are in fact well-behaved; the parameter $\lambda$ yields a linear combination of "objectives" (e.g., main reward+penalty/cost). In such cases, it is widely known that the resulting trade-offs from tuning $\lambda$ are well-behaved (see [1]).
- The introduction of core mathematical notions (conditional OT, the c-transform, and the Lagrangian formulation) is difficult to follow for readers not already fluent in OT theory and Lagrangian. For instance, the paper directly presents the dual form with the c-transform (eqs. 2-3) and the action integral (eq. 5) without first motivating them intuitively or connecting them to the simpler Kantorovich dual with explicit constraints. As a result, the rationale for moving from standard OT to a Lagrangian formulation, and why this path-based cost is required for CTI/HTI, remains implicit.
- The same issue carries over when explaining why the method can generalize from sparse observations: the text hints that the least-action and density biases provide this capability, but the link is scattered across sections 3 to 5 instead of being clearly stated when these concepts are first introduced. Overall, the exposition of the theoretical background is technically correct but conceptually opaque, making the narrative harder to follow than necessary.
- As acknowledged by the authors, a main limitation of the approach is that it only scales to a unique hyperparameter. In practice, many hyperparameters must be tuned. This fact is acknowledged only at the end of the paper; the limitation should be explicitly stated earlier in the text.
- Since one of the key selling points is the capability to learn from sparse observations, the evaluation would benefit from evaluating the quality of the predictions with respect to the sparsity of the observations (add or remove anchors and analyze the impact).

[1] Roxana Radulescu, Patrick Mannion, Diederik M. Roijers, Ann Nowé: Multi-objective multi-agent decision making: a utility-based analysis and survey. Auton. Agents Multi Agent Syst. 34(1): 10 (2020)

### Questions
- Could you explain why exactly you need to introduce the $c$-transform instead of the standard dual formulation? 
- Concerning the move from classical COT to Lagrangian COT, could you articulate more clearly why a path-based cost is needed for CTI/HTI and whether using only the kinetic term with a learned metric (no potential) would already be enough for sparsity?
- How does Equation 12 help in evaluating Equation 11 faster? As you train $T\_{\theta}\_{T, k}$ via $\mathcal{L}\_{map}$, do you explicitly need to minimize Equation 11?
- Would you expect to obtain so clean results on RL experiments if the reward scalarization function were non-linear?

### Soundness
3

### Presentation
2

### Contribution
3
