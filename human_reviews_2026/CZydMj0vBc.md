# Learning Jump-Diffusion Dynamics from Irregularly-Sampled Data

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Accurately modeling time-continuous stochastic processes from irregular observations remains a significant challenge. In this paper, we leverage ideas from generative modeling of image data to push the boundary of time series generation. 
For this, we find new generators of SDEs and jump processes  for conditional interpolation which match the marginal distributions of the time series of interest. 
Specifically, we can handle discontinuities of the underlying processes by parameterizing the jump kernel densities by scaled
Gaussians that allow for
closed form formulas and hence rapid evaluation of the corresponding
Kullback-Leibler divergence in the loss. 
Unlike most other approaches, we explicitly account for both irregular and non-aligned sampling times in constructing the generators. We also clarify several theoretical aspects that lead to a more robust formulation of the model.  We underline our theoretical results by numerical experiments involving combinations of jumps and SDE dynamics  that illustrate the benefits of the proposed framework

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
A paper proposing a diffusion model for irregularly sampled time series. The key innovation is the proposal of jump-diffusion transition kernels to model the bridge probability in a way that is independent of the sampling time of the data. The paper is primarily theoretical and has a limited evaluation on synthetic data, but it does use one semi-realistic data set consisting of a moving square.

### Strengths
A mathematically rigorous paper trying to extend generative diffusion modelling to a non-trivial scenario of irregularly sampled time series. The quality of the work is sound and the mathematical details are given in depth.

### Weaknesses
- The paper is way too dense to expect someone who is not in the same niche to be able to grasp much, particularly given the very short time given to reviewers (who also have another four papers and, occasionally, a day job). I appreciate that the topic is highly technical, but the notation and style of presentation would have been significantly improved by some streamlining.
- The main technical contribution appears to be a regularisation in the bridge probability, is that so? It is somewhat difficult to assess exactly what was already done in the very recent papers by Holderrieth and Zheng, and what was innovation in this paper.
- Sometimes there is some sloppiness in the use of the word "process", for example Alg. 1 is a procedure to sample trajectories (approximately), which in most literature is distinct from the process (infinite dimensional object).
- The memory aspect is another place where greater clarity would be needed, it's even unclear what it's meant (my impression was that the statistics of the project were obtained by some moving average over the trajectory but  as the topic is important its relegation to a short paragraph was definitely too little) 
- The empirical section was very schematic, which might be fine for a strongly theoretical paper but perhaps something that would showcase better the advantages of including jumps would have helped, as the performance is often close to the diffusion method.

### Questions
- It seems to me that the trajectories might be irregularly sampled, but all trajectories must be sampled at the same times, is that so?
- Could you clarify a bit more how memory is introduced?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors apply a generator learning technique to the task of irregular time series generation. The authors derive the neccessary loss for such an application, and show performance in synthetic settings.

### Strengths
1. The work is technically solid, the derivations are well written and clearly presented. 
2. The writing and presentation of the paper is clear, it is easy for readers to understand.

### Weaknesses
Minor: 
1. Slight notation inconsistency, e.g $\Delta$ and $\nabla^2$ are both used for hessian in different places. 
2. The paper assumes an identity diffusion coefficient, limiting the modeling flexibility. Although this is relatively standard in this type of research, the diffusion coefficient can be easily modeled by a neural network.

Major:
1. The work lacks novelty. This is perhaps the most concerning weakness. The work seems to be taking Holderrieth 2025's idea and apply it to the irregular time series modeling setting in TFM (Zhang, 2024). 
2. Related to the previous points, it will be more convincing that the work is empirically solid by repeating some numerical experiment in TFM. Most of the experiments now only show that it improves in synthetic scenarios. 
3. The first two points combined lead to a relatively weak motivation to apply generator learning (with jump processes) in such a specific field of application.

### Questions
1. Can the author explain why having no singularities during the observed points of irregularly sampled data is beneficial? Is it solely to construct a full probabilistic view, or are there some particular benefits to having $P_0, P_1$ not being the observed points, i.e does it improve robustness, help with uncertainty quantification, etc?
2. Can the author explain why having a jump process during the generator learning is necessary? Aside from the benefit during synthetic data modeling, does it actually help (or perhaps at least does not deteriorate) in modeling full continuous time series?
3. The parameter $\eta^2 = 0.3$ seems small compared to the synthetic data's scale, leaving a rather smooth path to model; doesn't this limit the usage of SDE? What happens if we leave $\eta^2 = 1$?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose a method that estimates jump diffusions from observational data. By using existing known generators of certain classes of jump diffusion processes, the authors propose transforming those such that their time marginals match the observational data. This is similar to the techniques used in stochastic interpolants which maps one marginal distribution with known properties to another. The authors leverage an Ito-Levy type decomposition to represent the continuous and pure jump parts. The key component is to parameterize the jump component as a Gaussian distribution which allows the use of KL divergence for training. The authors then describe how to estimate the related generator given observations of irregularly sampled time series. Finally, the authors consider a few numerical examples on the proposed method.

### Strengths
The authors propose an elegant formulation that matches a given base generator to a target generator that supports jump processes.

The authors find initial distributions that are amenable to sampling from. 

The authors find a workaround to compute the jump measure more effectively than in existing works.

### Weaknesses
The central contribution is largely a combination of the work of Zhang et al and Holderrieth et al to consider generator matching with jumps. 

The numerical results appear to be inconclusive as to the efficacy of the method, which is not something to be concerned about, but it would be good to see which scenarios the method does perform well in as a comparison. This leads to a question on where the method should be employed what set of tasks. This was not evident in the main text. 

The numerical experiments do not illustrate the importance of the jump component, which should be studied.

### Questions
Can the authors detail the differences and the techniques needed to bridge Zhang el al to Holderrieth et al? 

I’m struggling a bit to understand what the correct use-case of this method would be. The authors motivate with financial data or possibly limit order book data which is irregularly sampled, but there are a series of methods that could work in such scenarios. Can the authors comment on this a bit more, why and where the particular method would work well?

Are there more obvious jump related data that the authors could consider? This would go a long way in motivating the use case of the method.  

Is it limiting to consider functions $r$ that are parameterized by a Gaussian?

### Soundness
3

### Presentation
2

### Contribution
2
