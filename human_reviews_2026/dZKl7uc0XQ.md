# Mirror Flow Matching with Heavy-Tailed Priors for Generative Modeling on Convex Domains

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
We study generative modeling on convex domains using flow matching and mirror maps, and identify two fundamental challenges. First, standard log-barrier mirror maps induce heavy-tailed dual distributions, leading to ill-posed dynamics. Second, coupling with Gaussian priors performs poorly when matching heavy-tailed targets. To address these issues, we propose Mirror Flow Matching based on a \emph{regularized mirror map} that controls dual tail behavior and guarantees finite moments, together with coupling to a Student-$t$ prior that aligns with heavy-tailed targets and stabilizes training. We provide theoretical guarantees, including spatial Lipschitzness and temporal regularity of the velocity field, Wasserstein convergence rates for flow matching with Student-$t$ priors and primal-space guarantees for constrained generation, under $\varepsilon$-accurate learned velocity fields. Empirically, our method outperforms baselines in synthetic convex-domain simulations and achieves competitive sample quality on real-world constrained generative tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work proposes mirror flow matching, which addresses previous limitations of mirror diffusion models (Liu et al., 2023a) and presents a comprehensive theoretical framework for this approach. The key contributions include: (1) a modified mirror map designed to guarantee Lipschitz conditions, and (2) the use of a Student's t-distribution heavy-tailed prior for training the flow matching model in the transformed dual space. The empirical study demonstrates improvements compared to baseline methods, such as reflect flow matching and gauge flow matching.

### Strengths
1. The theoretical analysis is comprehensive and solid, including spatial and temporal regularity results and final distribution error bounds under discretization. These theoretical contributions may be of independent interest beyond this work.

### Weaknesses
1. The mirror map is limited to simple sets with closed-form expressions. Despite the comprehensive theoretical analysis of this framework, its practical applicability remains limited due to these restrictions.
2. The additional benefits in the empirical study of the proposed approach compared to Gaussian priors are minor. Moreover, the experiments lack a comparison to the traditional mirror diffusion models (in Tables 1 and 2), which serve as the main baseline work discussed throughout the paper.
3. The expression of the inverse mirror map is unclear. The authors should provide a table with explicit forms of both the forward and inverse mirror maps for different convex sets to improve clarity and reproducibility.
4. It is unclear how the Lipschitz conditions in Equation (1) are satisfied given Proposition 2.2, which only provides a bound on the expectation of the Lipschitz constant, while Equation (1) requires pointwise Lipschitz conditions.

### Questions
1. In the original mirror diffusion model work, the authors consider specific linear constraints with orthonormality conditions, which admit a closed-form inverse. However, such requirements are not explicitly stated in this work. Can the authors specify the forward and inverse mirror maps used in this work and clarify what conditions are necessary for closed-form inverses?
2. What is the role of the heavy-tailed prior distribution in Proposition 4.1? How does it contribute to the theoretical guarantees or empirical performance?
3. Why does RFM not achieve a 100% feasibility rate in Table 1? 
4. In Proposition 3.1, what is the prior distribution for the primal and dual spaces, respectively? 


Overall, this work is theoretically solid. I will adjust my rating upon clarification of the applicability and details of the mirror map stated above.

### Soundness
3

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
3

### Summary
This work aims to improve the mirror flow matching algorithm — which is used to generate samples from a distribution supported on a convex domain — by proposing a new mirror map and using the Student-t distribution as the prior distribution.

They show on artificial data that the log-barrier transformation, which is commonly used as a mirror map, can induce heavy tails when pushing the target distribution into the dual space. To replace it, they introduce a new mirror map that ensures the existence of the moment of order p of the pushforward, by this new map, of probability measures satisfying the “boundary-measure condition.” This mirror map also has the property of being strongly convex, which allows transferring convergence guarantees from the dual space to the primal space.

They also illustrate, with a well-chosen 1D example, that for target distributions with heavy tails, the Gaussian prior is unsuited for the flow matching framework, and they propose using a Student-t prior instead. Still with the Student-t prior, they obtain Lipschitzness guarantees both in time and space for the true velocity field in the general flow matching framework when the target has a polynomial tail bound. This allows them to derive an upper bound on the Wasserstein distance between the true data distribution and the intermediate distributions generated by the learned velocity field of the general flow matching algorithm. Using the strong convexity property of the mirror map, they transfer this discretization bound from the dual space to the primal space.

They tested this new mirror flow matching approach on simulated data and on the FHQv2 dataset by generating watermarked images.

### Strengths
The paper is well written and a pleasure to read. The main new ideas concerning mirror flow matching — the mirror map and the new Student-t prior — are well motivated, and the arguments given to introduce each of them are fully valid.

In addition, the theoretical results on the Lipschitzness guarantees, both in time and in space, of the velocity field when the prior is a Student-t distribution (Proposition 4.1) are new and relevant for the general flow matching framework, as they require lighter constraints on the target distribution than prior works. The same holds for the discretization upper bound in Theorem 3, which is also relevant for the general flow matching framework.

### Weaknesses
While the theoretical sections are very rigorous, the experimental section is not and should be rewritten. The choice of baselines is not justified, and the baselines differ between Experiments 5.1 and 5.2. The choices of the hyperparameters $\kappa$ and $\nu$ are also not justified, and their values change between the two experiments. The architecture of the neural network used to estimate flow matching in the dual space is not mentioned. The comparison with other baselines is insufficiently detailed (e.g., the number of samples used to compute the MMD and KL divergence metrics in Table 1 is not specified). Finally, the notion of Feasibility (Table 1) is defined in the paper.

### Questions
- Proposition 2.2 assumes that the newly defined mirror map does not induce heavy tails and guarantees the existence of moments of the pushforward probabilities that initially (before being pushed forward by the mirror map) satisfy the boundary-measure condition. It would be interesting to see, in practice, for examples where this assumption is violated, whether the proposed mirror map induces lighter tails than the log-barrier map. You already verified this for the experiment described in Appendix A — do you observe the same result (i.e., that your proposed mirror map induces lighter tails than the log-barrier map) for the target distributions described in the experiments from Sections 5.1 and 5.2?

- Line 202: you claim, “This example highlights a key principle: when the target distribution is heavier-tailed than the prior, the conditional distribution is likely to have a mode that is dominant near $x_t$
 for some values of $t$” Do you know of other examples or references that could help generalize this principle beyond the pathological case presented in Example 2?

- Can you justify your choice of baselines for the experiments detailed in Sections 5.1 and 5.2?

- How did you set the hyperparameters $\kappa$ and $\nu$ in both experiments? What happens for other values of these hyperparameters? How sensitive is the algorithm to them? 

- What is the architecture of the neural network used to parameterize the vector field trained in the dual space?

- Can you provide more information about the results reported in Table 1 (e.g., the number of samples used to compute the MMD and KL divergence metrics)? How is Feasibility defined?

- Finally, can you comment further on the fact that your results are not as good as those of the MDM baseline for the watermarked image generation task? Do you think this might be due to suboptimal choices of the hyperparameters $\kappa$ and $\nu$?

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
4

### Summary
The paper “Mirror Flow Matching with Heavy-Tailed Priors for Generative Modeling on Convex Domains”proposes a framework for generative modeling under convex constraints using flow matching enhanced by mirror maps and Student-t priors. The authors identify that traditional log-barrier mirror maps lead to heavy-tailed dual distributions and ill-posed dynamics, while Gaussian priors fail to align with such heavy-tailed targets. To address these issues, they introduce a regularized mirror map ensuring finite moments and strong convexity, coupled with Student-t priors that stabilize training and maintain well-behaved velocity fields. Theoretical contributions include proofs of spatial Lipschitzness, temporal regularity, and Wasserstein convergence bounds for flow matching with heavy-tailed priors, extended to the constrained (primal) domain. Empirically, the method achieves superior or competitive performance compared to reflection- and gauge-based baselines on constrained generative tasks—such as sampling within polytopes, L2  balls, and watermarked image generation—while guaranteeing feasibility and improving robustness on complex convex domains.

### Strengths
- The paper is rigorous and very well written. 

- It makes relevant methodological contributions: particularly on how to perform flow matching with a mirror map framework, on how to choose the mirror map, and on how to set the initial distribution in order to handle target distributions with heavy tails, which is important as the distributions in the dual space can be heavy-tailed.

- The paper contains experiments in which the target distributions are over polytopes, and watermarked image generation experiments, although the images are low-quality.

### Weaknesses
- The applications of the work are not very compelling and may not justify the complexity of the framework. 

- In the experimental section, the authors compare their performance to Mirror Diffusion Models, which is a previous work on using mirror maps for diffusion models. However, there is not comparison to Mirror Diffusion Models in the paper. Since Flow Matching and Diffusion Models are very related, it would be useful to clarify the differences between Mirror Flow Matching and Mirror Diffusion Models.

- More generally, the paper is lacking a related work section that places it in the context of previous works that leverage mirror maps for diffusion/flows.

- The paper can be considered incremental in that it is not the first diffusion/flows paper to rely on mirror maps, but it does clarify some design choices to make them work better.

### Questions
- How should the number of degrees of freedom of the Student t prior be set?

- What if the distribution has mass at the boundary of the constraint set? The mirror transformation would send such points to infinity, so how can we model them?

### Soundness
4

### Presentation
4

### Contribution
3
