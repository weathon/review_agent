# Generative Bayesian Optimization: Generative Models as Acquisition Functions

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
We present a general strategy for turning generative models into candidate solution samplers for batch Bayesian optimization (BO). The use of generative models for BO enables large batch scaling as generative sampling, optimization of non-continuous design spaces, and high-dimensional and combinatorial design. Inspired by the success of direct preference optimization (DPO), we show that one can train a generative model with noisy, simple utility values directly computed from observations to then form proposal distributions whose densities are proportional to the expected utility, i.e., BO's acquisition function values. Furthermore, this approach is generalizable beyond preference-based feedback to general types of reward signals and loss functions. This perspective avoids the construction of surrogate (regression or classification) models, common in  previous methods that have used generative models for black-box optimization. Theoretically, we show that the generative models within the BO process follow a sequence of  distributions which asymptotically approximate an optimal target under certain conditions. We also evaluate the performance through experiments on challenging optimization problems involving large batches in high dimensions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the challenge of how to approach Batch based Bayesian optimisation. The challenge in Batch BO is how to query a set of evaluation locations as we want them to be diverse. The traditional approach turns this into a sequential decision problem. The idea in this paper is to learn a separate model that parametrises the support of the acquisition function such that this can be used to sample candidate locations. The paper presents several different approaches on how such a model can be defined both using the acquisition function perspective and one of preference losses.

### Strengths
The paper is generally well written and honest in what it proposes not overstating what is proposed, thank you for this. The work builds significantly on the work proposed in [1] but proposes new and novel mechanisms to train the model and the use-case here is directly BO which was only suggested in the Appendix of [1]. The preference based training is to my knowledge novel in this modelling scenario. The paper tackles a very important problem in any sequential decision making system where multiple proposals can be exploited at the same time.

[1] Stein berg, D. M., Oliveira, R., Ong, C. S., & Bonilla, E. V. (2024). Variational Search Distributions. CoRR, (), .

### Weaknesses
While I appreciate the extensive background writing it somewhat also pollutes the story a bit. For example, I am not actually sure why BO using latent representations is part of the Batch-BO introduction. While it is an important class of models it is in somewhat not used to "paint a picture" of what you are proposing. I think the work would come into a better context if some of the background was shortened and details where only introduced if they helped the story of the paper. This might be completely personal opinion but I find the text a bit scattered up to Section 3 and some of this could be moved to after the proposed method. Again just a personal opinion that has had no impact on my score or review.

One thing that I didn't fully understand is what is the specific q that you are using and what the p_0 is for the experiments that requires this. Maybe I've missed it in the paper but it would make the results easier to interpret.

### Questions
Could you expand on what the specific model for q is and detail what p_0 is in the experimental settings that requires this?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a framework called generative Bayesian optimization (BO), which transforms generative models into candidate samplers for batch BO. By leveraging ideas from direct preference optimization (DPO) in RLHF, the approach trains generative models directly using noisy utility values derived from observations, making the proposal distribution's density proportional to the acquisition function. This eliminates the need for intermediate surrogate models, reducing computational cost and approximation errors compared to prior methods like VSD or BORE. The framework supports two training strategies: a DPO-like loss for preference-based feedback and divergence minimization with utility-weighted samples for general rewards. It is particularly suited for large-batch scaling, high-dimensional spaces, non-continuous designs, and combinatorial problems. Theoretically, the authors show that the sequence of generative models asymptotically concentrates at the global optima under certain conditions. Empirical evaluations on challenging high-dimensional optimization tasks demonstrate improved performance for large batches.

### Strengths
1. The key innovation is treating generative models as direct acquisition functions via DPO-inspired training, bypassing the two-stage process (surrogate then generative) in existing methods. This simplifies the pipeline, makes it more general to utility types, and enables efficient sampling for large batches in complex spaces. The extension to batch BO addresses a practical need in parallel evaluations, like simulations or experimental design.
2. The paper provides convergence analysis, showing asymptotic concentration of proposal distributions around optima. Assumptions seem reasonable (e.g., bounded domain, sub-Gaussian noise, smooth objectives), building on standard BO theory (e.g., Bull, 2011 for EI; Srinivas et al., 2010 for regret). 
3. Scalable to high dimensions and combinatorial domains, where traditional GP-based BO struggles. Compatibility with pre-trained generative models allows fine-tuning for optimization, with low sampling cost. The framework generalizes beyond preferences to arbitrary utilities, broadening applicability to areas like hyperparameter tuning, materials, and robotics.
4. Equations are well-explained, and the motivation from real-world batch scenarios is compelling.

### Weaknesses
1. While claiming reduced cost by avoiding surrogates, training generative models (especially large ones like LLMs) on utilities could still be expensive in practice, particularly for online BO with frequent retraining. No discussion of training overhead vs. baselines.
2. Focuses on batch BO, but less emphasis on sequential cases or comparison to non-generative batch method. Robustness to noisy utilities or misspecified priors isn't addressed. For combinatorial spaces, discretization or embedding strategies aren't detailed.

### Questions
Please see the Weaknesses.

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
This paper presents the direct use of generative models as samplers for design selection in batched Bayesian optimization. Unlike prior two-stage approaches that first fit a surrogate or regressor and then train a generator, the authors propose a single-stage method that learns a generative model whose density is proportional to the expected utility. Two approaches are introduced. The first is preference-based learning, analogous to Direct Preference Optimization (DPO), using a contrastive objective built from pairwise utilities, with a robust variant that accounts for noisy labels and sign flips. The second is a KL-divergence route that trains the generator to match the utility-weighted target distribution via a balanced forward KL, implemented with importance-weighted sampling to yield an unbiased objective, along with an alternative KL derived from Bregman divergences to avoid allocating high probability mass to low-utility regions. The paper also provides theoretical analysis showing convergence to the utility-based target distribution and concentration of the generative BO proposals at the true optimum.

### Strengths
1. The core idea of a single-model training approach for the generative sampler is well motivated and clearly presented.

2. The method attempts to generalize across expected-utility acquisition function variants.

3. Scalability is linked to batched BO, making the approach practically relevant if successful.

4. The paper provides theoretical analyses that support the claims, with careful attention to both convergence and optimality.

5. The empirical evaluation compares the proposed methods against relevant baselines.

### Weaknesses
1. The empirical analysis is well motivated but does not make a fully convincing case for the proposed algorithm.

2. Related to the above, the results do not show consistent superiority over all baselines on the more complex problems. While the authors do note cases of underperformance with provided justifications, it would help to comment on how the approach could be improved to close those gaps.

3. The plots are quite difficult to read.

3. Minor: while the paper claims generalization to utility-based acquisitions, it does not clearly discuss extensions to other acquisition families such as GP-UCB or Thompson Sampling.

### Questions
1. I am willing to raise my score if the authors can address my main concerns about the empirical results, specifically, by demonstrating stronger robustness and clearer superiority over baselines where possible. 

2. While the framework targets utility-based acquisitions, can you comment on extensions to other acquisition families such as Thompson Sampling and GP-UCB? Perhaps such extensions are not straightforward or not possible? An explanation would be great.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Generative Bayesian Optimization (GenBO), a framework where a generative model is used directly as the acquisition function for Bayesian Optimization (BO).
Traditional generative BO methods often use a two-stage process: first, training a surrogate model (like a Gaussian Process), and second, training a generative model based on the surrogate's predictions. This can lead to compounding errors.
GenBO, inspired by Direct Preference Optimization (DPO), bypasses the intermediate surrogate model. It trains the generative model directly on simple, observed utility values (such as Expected Improvement or Probability of Improvement). The resulting model's probability density becomes proportional to the expected utility.
Therefore, sampling from this trained generative model is equivalent to sampling candidates from the acquisition function, simplifying the pipeline, reducing approximation errors, and enabling scalability to large batches and complex, high-dimensional spaces like protein design.

### Strengths
The author provided the theroetical analysis of their proposed method.

### Weaknesses
**Illegible Figures**

The figures are poorly rendered and extremely difficult to interpret. The hues selected for the proposed method and the various baselines are too similar, making the different plots (e.g., performance curves) indistinguishable. Compounding this issue, the legend is rendered at a font size that is far too small to be legible. As a result, it is impossible for the reviewer to accurately interpret the experimental results or validate the paper's claims.

**Insufficient Experimentation and Analysis**

The empirical validation is severely lacking. The paper presents minimal experiments with what appears to be a complete absence of analysis. The claims are not adequately supported by empirical evidence.

**Trivial Methodology and Limited Contribution**

The core idea seems to be training an LLM on (data, utility function value) pairs via DPO to generate better data points. This appears to be a trivial application of DPO to the Bayesian Optimization (BO) setting. If one were tasked to "use DPO for BO," this is the most straightforward idea one would likely conceive. The conceptual novelty and overall contribution seem too limited.

**Clarity on Training Data**

The amount of data used for training the LLM is not specified. BO inherently assumes an expensive-to-evaluate setting, which typically yields sparse data. It is highly questionable whether this data regime is sufficient to fine-tune an LLM effectively.

**Baseline selection**

The field of generative model-based BO is well-established, often referred to as LBO (Latent Space Bayesian Optimization)[1~7]. The paper fails to position itself relative to this body of work. What are the specific advantages of this DPO-based approach compared to existing LBO methods? The lack of comparison makes it impossible to assess the method's practical or theoretical benefits.

**Reference**

[1] Tripp, Austin, Erik Daxberger, and José Miguel Hernández-Lobato. "Sample-efficient optimization in the latent space of deep generative models via weighted retraining." Advances in Neural Information Processing Systems 33 (2020): 11259-11272.

[2] Maus, Natalie, et al. "Local latent space bayesian optimization over structured inputs." Advances in neural information processing systems 35 (2022): 34505-34518.

[3] Lee, Seunghun, et al. "Advancing bayesian optimization via learning correlated latent space." Advances in Neural Information Processing Systems 36 (2023): 48906-48917.

[4] Chu, Jaewon, et al. "Inversion-based latent bayesian optimization." Advances in Neural Information Processing Systems 37 (2024): 68258-68286.

[5] Moss, Henry B., Sebastian W. Ober, and Tom Diethe. "Return of the latent space COWBOYS: Re-thinking the use of VAEs for Bayesian optimisation of structured spaces." arXiv preprint arXiv:2507.03910 (2025).

[6] Grosnit, Antoine, et al. "High-dimensional Bayesian optimisation with variational autoencoders and deep metric learning." arXiv preprint arXiv:2106.03609 (2021).

[7] Lee, Seunghun, et al. "Latent bayesian optimization via autoregressive normalizing flows." arXiv preprint arXiv:2504.14889 (2025).

### Questions
See the weakness section above.

### Soundness
2

### Presentation
2

### Contribution
1
