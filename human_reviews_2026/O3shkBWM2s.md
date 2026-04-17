# Robust Optimization for Mitigating Reward Hacking with Correlated Proxies

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Designing robust reinforcement learning (RL) agents in the presence of imperfect reward signals remains a core challenge. In practice, agents are often trained with proxy rewards that only approximate the true objective, leaving them vulnerable to reward hacking, where high proxy returns arise from unintended or exploitative behaviors. Recent work formalizes this issue using 
r-correlation between proxy and true rewards, but existing methods like occupancy-regularized policy optimization (ORPO) optimize against a fixed proxy and do not provide strong guarantees against broader classes of correlated proxies. In this work, we formulate reward hacking as a robust policy optimization problem over the space of all 
r-correlated proxy rewards. We derive a tractable max-min formulation, where the agent maximizes performance under the worst-case proxy consistent with the correlation constraint. We further show that when the reward is a linear function of known features, our approach can be adapted to incorporate this prior knowledge, yielding both improved policies and interpretable worst-case rewards. Experiments across several environments show that our algorithms consistently outperform ORPO in worst-case returns, and offer improved robustness and stability across different levels of proxy–true reward correlation. These results show that our approach provides both robustness and transparency in settings where reward design is inherently uncertain.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to design reinforcement learning (RL) agents that remain robust to imperfect or proxy reward signals. The authors formulate reward hacking as a robust policy optimization problem over all proxy rewards that are $r$-correlated with the true reward, deriving a tractable max–min formulation. When rewards are linear in known features, the method incorporates this structure to yield more interpretable and robust policies. Experiments across several environments show improved worst-case performance compared to existing approaches.

### Strengths
The paper is well written and clearly presented.

It addresses an important problem in reinforcement learning (RL) where the reward function is imperfect. There have also been recent papers on similar settings in language models, such as:
- [1] https://arxiv.org/abs/2504.03784
- [2] https://arxiv.org/abs/2405.11204
Considering the growing popularity of LLMs, it would be interesting to discuss the potential applications of this approach.

The proposed approach appears principled, and the theoretical results effectively support the empirical findings.

### Weaknesses
- Equations (9) and (13) do not include the importance sampling ratio, whereas Section 3.3 does. I think this is because the chi-square divergence involves this ratio—if so, the authors could clarify this more explicitly. 

- The authors claim to have developed efficient algorithms, but the computational complexity of the proposed method is not discussed. Including empirical comparisons of runtime and memory usage against benchmarks would make the paper more complete. 

- How are the feature mappings selected? Are the theoretical/empirical results sensitive to the choice or number of mappings?

- The experiments could be strengthened by comparing with recent related works, such as:
  - InfoRM: Mitigating Reward Hacking in RLHF via Information-Theoretic Reward Modeling
  - RRM: Robust Reward Model Training Mitigates Reward Hacking
  - Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies robustness to reward misspecification by assuming access to a proxy reward $R_{\text{proxy}}$ and a reference policy $\pi_{\text{ref}}$.
It defines an uncertainty set of true rewards whose Pearson correlation with the proxy reward under $\pi_{\text{ref}}$ equals $r$, and trains a policy to maximize worst-case return over that set.
A change-of-measure via the Radon--Nikodym derivative (density ratio) was used, which results in a regularized optimization objective the authors note "closely resembles" occupancy-regularized policy optimization (ORPO) objective, differing by a regularization scaling and an extra penalty term inside the square root.
Convergence guarantee of the proposed method is provided, and empirical evidence on four "real-world inspired" reward hacking environments is given.

### Strengths
The paper gives a clean derivation of a max-min robust RL objective under a correlation-constrained reward set and reduces it to a tractable form via duality and change of measure.
The resulting penalty can be read as an "orthogonalized" $\chi^2$ term, penalizing the component of occupancy shift not aligned with proxy improvement, which is a neat geometric refinement relative to ORPO's regularization.
The implementation details for ratio estimation and the comparison to ORPO will be useful to practitioners.

### Weaknesses
## Reward-space vs. behavior-space

First, the core uncertainty model is reward-space, not behavior-space.
Correlation under $\mu_{\pi_{\text{ref}}}$ ignores the well-known fact that many behaviorally equivalent rewards (e.g., potential-based shapings) can have arbitrary correlation with the proxy, while tiny changes in reward can flip the optimal policy near decision boundaries.
As a result, the method may penalize or ignore the wrong directions in practice; a behavior-centric formulation (e.g., safe improvement stated in terms of behaviorial metrics) would better match the problem that matters.

The paper repeatedly motivates "guarding against reward hacking," yet provides no behavioral safe-improvement guarantee and even asserts that the correction term "enforces robustness to potential reward hacking," which is not supported by a theorem.
Over-pessimistic and over-optimistic rewards can both lead to suboptimal behaviors, and the agent may just learn a different reward hacking behavior.

## Reference policy

Second, the entire construction is anchored to a reference policy.
How's it chosen? Does it need to be a good policy?
The correlation constraints are imposed under $\mu_{\pi_{\text{ref}}}$, and are vacuous off its support; the adversary may set arbitrarily bad rewards in unvisited regions.
Consequently, the guarantee weakens precisely where a newly learned policy might go.
The paper assumes $\pi_{\text{ref}}$ is given and does not offer a principled selection rule, beyond practical choices.

## Correlation hyperparameter

Third, $r$ is a hyperparameter that must be known or tuned. The authors state there is "no principled method" to pick it and resort to grid search over $r \in \\{0.1,\dots,0.9\\}$, which is unintuitive for a Pearson correlation meant to represent true-proxy alignment.
Results also depend on the training/evaluation $r$ mismatch.

## Succesor representation

Finally, the linear-reward assumption $R(s,a)=\theta^\top\phi(s,a)$ is long-established (**successor representations/features/measures**); the section omits direct attribution even though the subsequent optimization uses discounted feature expectations exactly as in that literature.

### Questions
In its current form, the paper offers a tidy derivation and a small geometric refinement of ORPO, but it does not resolve the central mismatch between reward-space uncertainty and behavior-space objectives, and it relies on an unintuitive correlation hyperparameter and a reference-policy anchor.
I encourage the authors to
- clarify the relationship between reward function correlation under a reference policy and behavior difference;
- clarify the implications of off-support vacuity and provide coverage-aware constraints; and
- justify or estimate $r$ more principledly.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Building off Laidlaw et al (2025)'s definition of "r-correlated" proxy reward functions, the authors introduce a robust optimization framework aimed at reducing reward hacking. Their method focuses on maximizing the minimum reward achievable across all possible r-correlated reward functions. This leads to a loss function that, while similar to that of Laidlaw et al, is distinct in its exact regularization term. Across several environments, they show that their approach achieves higher worst-case reward, compared to ORPO from Laidlaw et al, while typically maintaining similar expected reward.

### Strengths
The paper has several strengths:
- Intuitive conceptual contribution of optimizing worst-case reward over all reward functions that are r-correlated to the given proxy reward function. This seems like the natural progression from Laidlaw et al (2025).
- Theoretically sound framework. They apply several tools from robust optimization to derive a tractable loss function, which though ultimately similar to Laidlaw et al (2025), differs in the exact regularization term, and they show this leads to improved worst-case rewards in their experiments
- Improvement for linear reward setting. They are able to derive a refinement for the case where the reward function is known to be linear.
- Better occupancy measure estimates lead to also improving ORPO. Both this work and Laidlaw et al (2025) use a discriminator network to measure the occupancy measure divergence between a given policy and the reference policy. The authors note that in the original ORPO implementation, the discriminator network is not fully optimized, and thus, the resultant policies still end up visiting states that are low frequency under the original policy. By training the discriminator for longer, they also obtain improved results compared to the original ORPO implementation.
- Extensive experiments and ablations. The authors test their method on five environments that were designed for studying reward hacking, and include several additional results in the appendices (e.g. results with unnormalized rewards, results on robustness to "r", etc)

### Weaknesses
One paragraph that I thought was a bit strange was:
> "We adopted a similar approach used by ORPO (Laidlaw et al., 2025). For each environment, we first performed a grid search over several different values of r, and for each fixed r, we trained the policy using our algorithm. We then selected the rvalue that leads to the policy with the best expected worst-case return ... Notice that ORPO selects the optimal r that yields the best expected return under the true reward, which is infeasible in practice when the true reward is unknown during training. In contrast, our approach for choosing r is more practical."

It's true that the authors' approach is computable without the true reward unlike the process used in ORPO, but it's not much more practical, as without knowing the "correct" value of r, it's not clear which measure of the worst-case reward is most meaningful.  Therefore, I would hesitate to describe this approach as more "practical" than ORPO. Instead, it might be more helpful for the authors to directly reference the two practical methods for selecting r discussed in Appendix I.

By the way, the authors say they include results for all r that they considered in H.5, but then H5 only includes results for the Traffic and Tomato environments. Also strangely, they consider r \in {0.3, 0.5, 0.9} for Traffic and r \in {0.1, 0.4, 0.7, 0.9} for Tomato. Why do they not use the same grid for all environments? Also, it is somewhat strange that the grid for Traffic is not uniform either. Can the authors please include results for all environments?

The authors' process for selecting "r" also highlights one aspect of their framework that I found counter-intuitive. The set R_corr(r) as defined in Eq 4 includes all reward functions that are *exactly* r-correlated with the proxy. Apriori, I would have expected them to define this set as including all reward functions that are *at least* r-correlated with the proxy. Then, increasing r would monotonically increase the maximal worst-case reward over the functions in R_corr(r). As I understand it, with the current defn, this kind of monotonicity does not necessarily hold? And that is also why in the selection process for r that they use in the experiments (noted above), the highest r is not always picked (though, noise in the occupancy measure estimation and policy training process may also affect this empirically)? It seems odd that, under this framework, selecting r = 0.3 makes the policy robust to reward functions with exactly 0.3 correlation, but not to those with higher correlation, such as 0.6. Could the authors clarify this design choice?

### Questions
- Why is the same grid for r not used in all environments?
- Why did the authors not define R_corr as all reward functions that are *at least* r-correlated to the proxy, rather than *exactly* r-correlated to the proxy?
- How would one obtain the "reference policy" in practice?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the issue of reward hacking in reinforcement learning, where an agent may optimize for an imperfect proxy of the reward function which could lead the agent to diverge from the intended true objective. To tackle this, the authors take a robust approach to optimize for the feature weights of the true underlying reward from an uncertainty set if feature weights defined by $\chi^2$-divergence. This allows them to model this as a max-min optimization problem over the proxy rewards which are constrained via correlation with the true rewards of the system, which the authors then extend to linear rewards. Ultimately, they prove convergence of their algorithm and sample complexity bounds on the occupancy estimation.

### Strengths
- The paper is relevant and highly motivated by practical implementation. It builds upon the state of the art to improve upon interpretability and derive a tractable objective for the non-convex robust optimization problem. The paper is very clear and progresses fluidly.
- A closed-form solution is proposed and derived for the worst-case reward feature vector of the adversary by transforming this vector into a whitened version of itself, $\tilde{\phi}$, such that the $Q$-function becomes the identity matrix.
- From what I could determine, the proposed method is backed up by a strong theoretical analysis. Subsequently, the authors provided much empirical validation of this theory to verify their claims as well as sufficient detail to reproduce these experiments.

### Weaknesses
- In practice the correlation between a proxy and rewards, $r$, is unknown. The authors briefly mention this in the appendix and use a grid search to find this. However, as the author's mention, there is not a principled method for selecting the optimal $r$ and thus it may not scale.
- An assumption is made that the true rewards lie within the defined uncertainty set. In practice, this may not always be the case. The proposed reference policy may not provide sufficient coverage of the feature space which could lead to the problem.

### Questions
How could this extend to other uncertainty sets?

Minor things:
- $\gamma$ should be defined prior to equation 1 where you define the objects in the MDP tuple.
- It would be nice to see one of the algorithms appear in the main text as well as the convergence bounds stated formally in section 3.3 to help highlight the contribution of this work.
- Missing space between "latent" and "(true)" on line 1026.

### Soundness
3

### Presentation
4

### Contribution
3
