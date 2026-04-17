# Projected Coupled Diffusion for Test-Time Constrained Joint Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
Modifications to test-time sampling have emerged as an important extension to diffusion algorithms, with the goal of biasing the generative process to achieve a given objective without having to retrain the entire diffusion model. However, generating jointly correlated samples from multiple pre-trained diffusion models while simultaneously enforcing task-specific constraints without costly retraining has remained challenging. To this end, we propose Projected Coupled Diffusion (PCD), a novel test-time framework for constrained joint generation. PCD introduces a coupled guidance term into the generative dynamics to encourage coordination between diffusion models and incorporates a projection step at each diffusion step to enforce hard constraints. Empirically, we demonstrate the effectiveness of PCD in application scenarios of image-pair generation, object manipulation, and multi-robot motion planning. Our results show improved coupling effects and guaranteed constraint satisfaction without incurring excessive computational costs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a method to jointly sample two independent variables coupled using a joint cost function. Practically, given the independent denoising score functions of each variable, the goal is to couple their denoising process directly at test time to sample them jointly. The paper introduces some variants: (1) standard LMC/DDPM based: perform one step-denoising, use classifier guidance using the gradient of the cost function and then perform projection to map the noisy latents to the safe-set. (2) use tweedie clean estimates to calculate the gradient of the cost function. Overall the paper claims that the proposed method generalizes the formulation of classifier-guidance, projected diffusion, compositional diffusion and joint diffusion.

I am familiar with a concurrent work in this line of research: https://arxiv.org/abs/2509.08775
While my concerns are based on the concurrent work, my judgment of this paper will be independent of it.

### Strengths
1. The paper's unified formulation of classifier-guidance, projected diffusion, compositional diffusion and joint diffusion is very impactful and timely. The authors have provided exhaustive results to analyze the performance of their algorithm on toy-domain, a planar robotics task and an image generation task as well.

2. PCD can impose hard constraints in addition to joint diffusion while being computationally efficient.

### Weaknesses
1. Data fidelity vs projection: Since the cost function and projection operation hold for the clean distribution, data fidelity is of primary importance here. The more realistic the clean estimates are, the better evaluation and guidance can be done. However, as the authors acknowledge as well, the projection operation hurts data fidelity, as also observed by the concurrent work, when applied to low-quality clean estimates (particularly at higher noise levels). 

2. Differentiability of cost function: This is a considerably strong assumption (which is also used by many prior works like MPD, DPCC). For example, the signed distance based indicator function in SHD might not be differentiable everywhere. This is a case, especially for collision checking objectives for real robot executions. I agree that effective engineering design can mitigate this, but this limits the scalability of the approach.

3. convexity of constraints: Since two experiments in the paper deal with navigation and manipulation, it is worth noting that non-convex safe sets are pretty common in these two settings, most commonly arising from obstacle avoidance constraints. For example, in the highway task if the trajectories are trained without the rectangle in between and forced to avoid it at test time, the resulting safe-set becomes non-convex. This again limits the scalability.

### Questions
1. How feasible is designing a projection operator for every task? How sensitive is the overall quality of samples to projection hyperparameters?

2. How is the cost function in general defined for noisy latents? It seems from the algorithms that the method always uses the clean estimates for projection and using the same for cost function also empirically results in the best performance.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Projected Coupled Diffusion (PCD), a novel test-time framework for generating jointly correlated samples from multiple pre-trained diffusion models while simultaneously enforcing hard, task-specific constraints. The core problem is that generating from a joint distribution $p(x, y)$ is difficult, especially when $x$ and $y$ must be correlated in a specific way and satisfy some given hard constraints. PCD addresses this by modifying the reverse diffusion sampling process. At each step, the update for each variable is guided by three components:

1. The score from its own pre-trained model (e.g., $s_X^\theta(x_t, t)$).

2. A gradient from a coupling cost function $c(x, y)$ that encourages the desired correlation between variables.

3. A projection operator $\Pi_{\mathcal{K}}$ that forces the updated sample back into the feasible set of hard constraints.

This approach requires no retraining and unifies compositional generation with hard-constraint enforcement at test time. The authors demonstrate PCD's effectiveness across three distinct domains: multi-robot motion planning, diverse robot manipulation, and constrained image-pair generation.

### Strengths
1. This paper unifies two important aspects in diffusion sampling: coupled generation and constrained generation through a clear and effective way. The proposed PCD provides a general, test-time-only framework to address this. PCD operates over multiple pre-trained models and costs (analytic or learned), requiring no retraining of the base diffusions, which enables the method applicable to a wide variety of settings, for example, where paired data and costs are scarce or proprietary.

2. This paper is well written, easy-to-follow, and conducts extensive studies across various domains, including multi-robot planning, Push-T trajectory pairs, and paired face generation.

### Weaknesses
1. PCD relies on projection and gradient-based updates to enforce constraints. How does it handle test-time constraints that are non-differentiable, for instance, a logic-based rule where a sample is accepted only if it passes some non-differentiable verification? A concrete example would help.

2. The performance of the proposed method might require the estimated Tweedie to be of high-quality. Otherwise, the further guidance term might likely be inaccurate or even compromise the overall sampling process.

3. This paper introduces gradient-based guidance and per-step projections, which can increase wall-clock latency compared with non-gradient based baselines. In appendix C.1, the authors also mention "PCD is approximately 4 ∼ 7× slower than vanilla diffusion mainly due to the per-step projection operation." Are there any potential ways to enable faster sampling with PCD?

4. Could the authors provide a curve of performance v.s. the number of sampling steps to illustrate how will the quality of the estimated Tweedie term affect performance? An intuition is that with more sampling steps, the Tweedie estimate will be more accurate, which could facilitate better guidance.


5. I would encourage the authors to include discussion on the limitations of PCD along with failure analysis with qualitative examples.

### Questions
See the Weakness above.

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
2

### Summary
The submission proposes a test-time framework to sample from multiple pre-trained diffusion models while (1) promoting some notion of joint "correlation" across variables and (2) enforcing constraints on the individual outputs of each model. PCD augments the usual Langevin / DDPM updates with (1) a user-specified  coupling cost between variables and (2) projections at every diffusion step to guarantee individual constraint satisfaction. Their framework  several existing methods (e.g., classifier guidance, projected diffusion, and some forms of compositional diffusion) as special cases. Empirical demonstrations cover three domains: multi-robot navigation (collision avoidance coupling cost, velocity constraints), robot manipulation on PushT (diverse, non-intersecting trajectories  as a coupling cost with velocity constraints), and ``paired'' face generation (age-contrast coupling with gender/attribute constraints).

### Strengths
The paper is well structure and clearly written. The problem of composing several trained diffusion models at test time under constraints is indeed relevant, and prior work has aimed to tackle similar problems when sampling a single variable (e.g. a single image). The main novelty, in my understanding, is the ability to compose diffusion models potentially defined over different variables. The approach is widely applicable and requires minimal modifications of standard sampling algorithms. The use case in muti-robot systems is well motivated, since constraints arise naturally in physical systems and training joint distributions can be computationally costly as the number of agents grows. The toy-example in images -- although artificial -- showcases the applicability of the framework in a totally different setting.

### Weaknesses
I think that the notion of "correlated" variables, which is emphasized throughout the paper, could be better defined/explained/motivated. Since the framework is flexible to accommodate arbitrary coupling costs, correlations are in my view an understatement with respect to the practical utility and applicability of the proposed method and underlying problem it is tackling.

In PushT trajectory dissimilarity appears to me as a contrived objective/task, there might be manipulation examples (e.g. bimanual over multiple objects) that lend themselves more naturally/directly to the framework.

### Questions
Can you scale the approach beyond two variables/agents? 

Can you numerically verify that the two coupling limits match known methods (classifier guidance, projection), at least in a toy experiment?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper studies the problem of combining pretrained diffusion models while enforcing task-specific constraints. The authors present a projected coupled diffusion framework for constrained joint generation. This framework introduces two generative dynamics with coupled  guidance terms, and use projection to impose hard constraints. Several experiments are used to show the effectiveness.

### Strengths
- This paper studies the problem of generating samples from pretrained diffusion models while satisfying task-specific constraints. This problem is important because pretrained diffusion models are often available, but sample constraints are typically not enforced during training.

- The authors formulate a new problem of generating correlated samples under hard constraints.

- They propose a generation method based on coupled dynamics, combining a coupled cost with projection onto hard constraints. This approach generalizes projected diffusion to coupled dynamics.

- The authors further show that several existing methods can be viewed as special cases of the proposed coupled dynamics framework.

- Several experiments show benefits from both projected diffusion and cost guidance.

### Weaknesses
- The projected coupled dynamics are intuitively designed by combining cost and projection. However, the conditions under which this method converges have not been studied, and to where. 

- The effect of coupled costs has not been analyzed, and costs may not always be differentiable.

- The projection step is not discussed in detail. It can be infeasible when the constraints are non-convex. It is also not clear how to do projection for latent diffusion models. 

- All special cases correspond to degenerate forms of the projected coupled dynamics. New application scenarios for the projected coupled dynamics have not been explored.

- Experiments demonstrate the effectiveness of combining cost and projection, which is expected since it benefits from both projected diffusion and reward guidance. However, it remains unclear to what extent the experiments reveal the advantages of the coupled dynamics.

### Questions
See comments in Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
