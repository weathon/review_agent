# Generalised Linear Models in Bayesian RL with Learnable Basis Functions

- Decision: Reject
- Scores: 6, 6, 4

## Abstract
Bayesian Reinforcement Learning (BRL) provides a framework for generalisation of Reinforcement Learning (RL) problems from its use of Bayesian task parameters in the transition and reward models. However, classical BRL methods assume known forms of transition and reward models, reducing their applicability in real-world problems. As a result, recent deep BRL methods have started to incorporate model learning, though the use of neural networks directly on the joint data and task parameters requires optimising the Evidence Lower Bound (ELBO). ELBOs are difficult to optimise and may result in indistinctive task parameters, hence compromised BRL policies. To this end, we introduce a novel deep BRL method, $\textbf{G}$eneralised $\textbf{Li}$near Models in deep $\textbf{B}$ayesian $\textbf{RL}$ with Learnable Basis Functions ($\textbf{GLiBRL}$), that enables efficient and accurate learning of transition and reward models, with fully tractable marginal likelihood and Bayesian inference on task parameters and model noises. On challenging MetaWorld ML10 and ML45 benchmarks, GLiBRL improves the success rate of one of the state-of-the-art deep BRL methods, VariBAD, by up to $2.7\times$. Comparing against representative or recent deep BRL / Meta-RL methods, such as MAML, RL$^2$, SDVT, TrMRL and ECET, GLiBRL also demonstrates its low-variance and decent performance consistently.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work proposes a novel Bayesian RL method which avoids the difficulty of optimization of the ELBO. This is achieved by using generalized linear models. The work begins by thoroughly motivating why BRL methods commonly require optimizing the ELBO before highlighting how the linear assumption enables them to avoid this issue at all. The resulting method provides a tractable, permutation-invariant posterior which allows for closed form marginal likelyhood updating.
The empirical evaluation highlights that the proposed GLiBRL approach has the capability of outperforming BRL/meta-RL baselines in the challenging metaworld setting.

Overall it seems like an interesting contribution to ICLR and I am leaning towards acceptance.

### Strengths
The work is very well motivated and is a good fit for the ICRL community. I believe that it could serve as the basis for engaging discussions in the subset of the RL community. The approach seems very novel and avoids a significant hurdle in existing BRL methods. Thus, this work likely will spawn future works that explore other approaches to avoid having to deal with ELBO when working with Bayesian RL.

I appreciate the depth of section 2.1 as it nicely sets up the following section in which the GLiBRL method is introduced.

The empirical evaluation seems meaningful though complementing the experiments with a simpler setting that requires less compute might help the community to quicker evaluate their own approaches against GLiBRL.

I also want to highlight that I am very appreciative of code being available already during the reviewing period.

### Weaknesses
Some implementation details and design decisions should be discussed in a bit more detail. For example, in line 359 the dimensionality of $\theta_T$ and $\theta_R$ are set but it is not clear how relevant this design decision is. Why did the reward get a much larger dimensionality than the transitions? Could that setting be an explanation for the results in Figure 2 being less decisive for the rewards than the transitions?

The results could be a bit stronger if other meta-RL methods would have been considered that work better on metaworld. E.g. the work by Melo (https://proceedings.mlr.press/v162/melo22a.html) or Shala et al. (https://openreview.net/forum?id=UENQuayzr1).
In Line 418 it is stated that GLiBRL outperforms the "state-of-the-art", though I'm doubtful that this is the case.

I'm happy to increase my score if my questions get clarified and the explanation for design decisions makes it clearer how difficult it is to setup GLiBRL.

### Questions
* For how many seeds are you reporting the results?
* How large is the overhead of sampling for a Wishart distribution with GLiBRL?
* How did you tune the hyperparameters of GLiBRL/How sensitive is GLiBRL with respect to hyperparameter choices?
* Why is Table 1 included? What does the max value tell us that is not shown from the final and average performances?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes GLiBRL, a Bayesian reinforcement learning framework that models dynamics and rewards as generalised linear functions of learnable basis features, with small neural networks producing features $C_T$ and $C_R$ and linear heads equipped with conjugate priors. This design yields closed form posteriors for model parameters and a tractable marginal log likelihood objective that replaces ELBO based training, aiming to avoid posterior collapse while preserving sample efficient adaptation.

### Strengths
1. The central idea of combining learnable feature maps with linear in features Bayesian heads is conceptually clean and technically sound, giving exact posteriors $p(\theta_T,\theta_R\mid \mathcal C)$ and a closed form objective $\log p_{\phi}(\mathcal C)$ that avoids the optimisation pathologies of ELBO methods. This brings welcome clarity to Bayesian RL by separating representation learning from conjugate inference.
2. The paper positions the approach against VariBAD and related meta RL models and provides evidence on standard ML10 and ML45 benchmarks that the method improves success rates and stabilises training.
3. The empirical section includes ablations that connect the full method to a Normal Normal special case and to variants without noise inference, helping isolate which components matter most, and the reported results are accompanied by enough hyperparameter detail to encourage reproducibility.

### Weaknesses
1. The linearity in learned feature space introduces an expressivity trade off; when true dynamics or rewards demand strongly nonlinear or interaction heavy structure, the burden shifts entirely to the feature networks. The main text does not deeply probe failure modes or provide capacity sweeps that link basis dimension to performance and stability.
2. The baseline set, while representative, omits some strong recent meta RL or Bayesian model based baselines and does not include PEARL like variants that more directly test the claim that conjugate training avoids ELBO issues in comparable settings.
3. Computational efficiency claims are qualitative. There is limited reporting of wall clock time, memory, and per episode update cost as a function of basis width and dataset size, and no scaling plots that demonstrate the practical advantage of closed form updates as feature dimension grows.
4. The probabilistic assumptions are restrictive, including Gaussian noise and independence between transition and reward parameters. The paper does not examine robustness to heteroscedastic or non Gaussian noise, nor does it discuss how the approach would adapt when the matrix normal row covariance is not identity, which could matter for real world dynamics.

### Questions
1. How sensitive are results to the capacities of the basis networks and to the feature dimensionalities $D_T$ and $D_R$; can you report systematic sweeps and identify thresholds where linear heads become the bottleneck?
2. Can you provide wall clock, memory, and per episode posterior update timings on ML10 and ML45, and a scaling study that varies basis width to validate that closed form updates remain advantageous as feature dimensionality grows?
3. How would the method handle non identity row covariances in a matrix normal likelihood, or heteroscedastic and heavy tailed noise; is there a reparameterisation or approximate conjugacy that preserves a tractable training objective, and what is the empirical impact on performance?
4. Since a key motivation is avoiding ELBO collapse, can you add diagnostics beyond expected KL, such as mutual information between latents and data or predictive log likelihood trajectories, and include a PEARL style comparator tuned on MetaWorld V3 to clarify whether GLiBRL’s advantage stems from conjugacy, feature learning, or both.

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
This paper proposes the use of a linear model (of learned basis functions) for producing a parametrized distribution describing a family of tasks, for use in a Bayesian RL framework. The linearized final layer model enables efficient inference, and results in better performance than an existing method (VariBAD) and equivalent or slightly better performance than a meta-learning approach (MAML).

### Strengths
- Learning a distribution over a family of tasks that varies in a relatively low-dimensional way is a powerful approach to transfer, and has been only lightly explored. This is a reasonably practical approach to the problem.
 - The domains on which the method was applied are challening.
 - The approach seems mathematically solid, though I did not go through the math in detail.

### Weaknesses
First, the authors have not done a proper literature review. Their Background seems to be indicate that the field jumped straight from generic classical Bayesian RL (which the title implies this paper is, but which it is not, and which is correctly attributed to Duff) to meta-learning. In fact the setting the paper is in is either a latent-context MDP (due to Brunskill and Li, 2013 or so) or a hidden-parameter MDP (due to Doshi-Velez, 2016 or so), depending on whether the latent parameter generating the family of MDPs is discrete or continuous. The authors being unaware of these lines of work lead them to miss important related literature, for example the Bayesian hidden-parameter MDP model due to Killian (2017), which is probably the state-of-the-art model in this context, and very close to the model proposed here. An experimental comparison to that work, as well as follow-on work by Yao and Killian, is mandatory. For example, the authors mention sampling using the learned models as future work, but Killian already did that, eight years ago.

The experiments are also somewhat limited, being five samples over two domains, one of which does not show much improvement over an (at this point) very well established but quite generic meta-learning method.

### Questions
In your background section, do I understand correctly that by "context" you mean a batch of sample transitions from a single task? That is not what it means in a contextual MDP, or a latent context MDP, so some clarity here would be helpful. Perhaps the samples are a sort of non-parametric representation of the latent context.

### Soundness
2

### Presentation
3

### Contribution
2
