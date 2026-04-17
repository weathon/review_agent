# Supporting High-Stakes Decision Making Through Interactive Preference Elicitation in the Latent Space

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
High-stakes, infrequent consumer decisions, such as housing selection, challenge conventional recommender systems due to sparse interaction, heterogeneous multi-criteria objectives, and high-dimensional features. 
This work presents an interactive preference elicitation framework utilizing preferential Bayesian optimization (PBO) to learn the unknown utility function of a user from pairwise comparisons that are integrated in real-time. To increase efficiency in a complex feature space, we learn the preference model in the latent space of an autoencoder (AE). Additionally, to mitigate a cold start, we obtain a personalized probabilistic prior through an automated user interview with a large language model (LLM).
We evaluate the developed method on rental real estate datasets from two major European cities. The results show that executing PBO in the AE latent space improves final pairwise ranking accuracy by 12\%. For LLM-based preference prior generation, we find that direct, LLM-driven weight specification is outperformed by a static prior, while probabilistically weighted priors using LLMs achieve 25\% better pairwise accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose an interactive preference elicitation (PE) framework based on preferential Bayesian optimization (PBO).

It learns a latent utility function for the user from pairwise comparisons queried to that user.

To address the cold start issue and obtain informative probabilistic priors of feature weights for the utility function, they use an LLM-guided user interview, instead of a predefined static weight vector.

They also use an autoencoder (AE) to obtain a latent representation of lower dimension that can be used improve the sample efficiency of exploration.

They evaluate their method on rental market datasets from two European cities (Madrid, Spain and Munich, Germany).

### Strengths
This paper is well-motivated and addresses a real-world problem. As the authors state in the conclusion, it has immediate applications for online real estate platforms, where it could reduce user fatigue by minimizing the number of property comparisons needed to identify suitable options.

The presentation of the paper is mostly clean and easy to follow. Notation is properly introduced, and the order in which concepts are introduced is good.

The experimental results are detailed, including a high number of runs, graphs with confidence intervals, and a table with numerical metrics.

### Weaknesses
Experiments are limited to rental markets in two cities: Madrid, Spain and Munich, Germany.

It would be nice to see experiments beyond the rental market setting, to get a better assessment of how well the method performs in other domains.

The LLM that the authors used, Gemini 2.5 Flash-Lite, is closed-source.

The Munich dataset that the authors used is not publicly available due to licensing issues.

Minor corrections:

Page 3: Replace r << d with r \ll d.

Page 6: Replace "as activation function" with "as the activation function".

Suggestions:

Page 7: Since there's space, include the formula for NDCG@k.

### Questions
Page 4: "we do not explicitly denote data normalization" What does this mean?

Have you experimented with other LLMs, besides Gemini 2.5 Flash-Lite? Why not experiment with an open-source LLM? It would be good to run the experiments with other LLMs, to see how much the choice impacts performance.

How did you select the personas?

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
2

### Summary
The paper targets high-stakes, sparse-interaction decisions such as housing selection, where conventional recommender systems fail due to limited feedback and high-dimensional features. To address this, it proposes an interactive preference elicitation framework that combines Preferential Bayesian Optimization (PBO) with two key components: LLM-based probabilistic priors, which interpret natural-language interviews to initialize user utility weights and mitigate cold start; and an Autoencoder-based latent representation, which reduces dimensionality for efficient exploration. The system learns a latent utility model from pairwise user comparisons using a Gaussian-process surrogate and qEUBO acquisition for adaptive querying. Evaluations on real-estate datasets from Madrid and Munich show that latent-space PBO improves ranking accuracy and that LLM-guided priors significantly enhance sample efficiency and cold-start performance.

### Strengths
The paper tackles an important and realistic problem of learning preferences in high-stakes, sparse-feedback decision settings. It proposes a coherent framework that integrates LLM-based prior elicitation, Autoencoder latent-space learning, and Preferential Bayesian Optimization into a unified probabilistic approach. The method is well-founded, combining Gaussian Process modeling with qEUBO-based active querying for efficient preference learning. Experiments on real-estate datasets show clear improvements in ranking accuracy and cold-start performance, demonstrating both methodological novelty and strong practical relevance.

### Weaknesses
## Weaknesses
A central modeling limitation lies in the treatment of noise within the preference likelihood. The paper explicitly assumes that both the user’s preference inconsistency and the autoencoder (AE) reconstruction error can be modeled as *jointly Gaussian and additive*. In Appendix A.1 (lines 593 page 11), the authors state that the decoder output can be written as $ \hat{x} = h_\theta(g_\theta(x)) = x + \epsilon $ with $ \epsilon \sim \mathcal{N}(0, \Sigma_\epsilon) $, and that the user’s utility varies locally linearly, $ u(\hat{x}) \approx u(x) + \nabla u(x)^\top \epsilon $. Combining these yields a single Gaussian error term $ \eta $ with variance $ \sigma^2 = \sigma_{\text{pref}}^2 + \sigma_{\text{recon}}^2 $ in the Probit likelihood, expressed as  
$ P(z \succ z') = \Phi\!\left(\frac{u(x) - u(x')}{\sigma}\right) $,  
where $ \sigma $ captures both preference noise and AE uncertainty (page 5).  
While elegant, this simplification conflates heterogeneous uncertainty sources—human inconsistency, model reconstruction bias, and latent-space distortions—into one scalar variance term. It presumes homoscedastic, isotropic noise and local linearity of the utility function, which are rarely valid in complex, nonlinear real-estate feature spaces. In practice, AE errors vary significantly across regions of the feature manifold, and user responses exhibit contextual, multimodal variability. This global Gaussian assumption therefore risks *underestimating epistemic uncertainty*, leading to overconfident posterior estimates under the Laplace approximation and potentially premature exploitation in qEUBO selection.  

Beyond these statistical concerns, the model’s abstraction of user feedback neglects potential heteroscedastic and structured noise patterns. Real users exhibit region-dependent reliability—for example, being consistent on low-price listings but noisy on high-price ones—yet the framework treats all feedback as equally noisy.

### Questions
See Weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes an interactive preference elicitation framework for complex, infrequent decisions like housing purchase choice. It combines LLMs (to generate personalized priors from natural language) and an autoencoder (to reduce feature dimensionality) within a PBO that learns user utility from pairwise comparisons in real time, evaluated on rental datasets from two European cities.

### Strengths
The paper is well-written with clear structures. The problem itself is well motivated.

### Weaknesses
The intuition for the selection of each structure for each purpose could be strengthened. Please refer to the questions.

### Questions
- In line 64-67, what is the purpose/intuition of still using LLM for the continuous feature space? Even after reading Section 3.2.1, my question about this does not go away.
- In line 82, you may want to elaborate/define the meaning of ``interactive preference elicitation'' which has appeared in the abstract & title but nowhere else before line 82 in the intro.
- What is the purpose of the warm-start, and any more empirical results to show the importance of this?
- In the experiment, what is the intuition for the vanilla (statistical) having an opposite trend (increase first and then decrease over the number of queries) with other benchmarks? 

I might raise my rating after seeing the rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3
