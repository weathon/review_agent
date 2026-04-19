# ScaLES: Scalable Latent Exploration Score for Pre-Trained Generative Networks

- Decision: Reject
- Scores: 8, 6, 5

## Abstract
We develop Latent Exploration Score (LES) to mitigate over-exploration in Latent Space Optimization (LSO), a popular method for solving black-box discrete optimization problems. LSO utilizes continuous optimization within the latent space of a Variational Autoencoder (VAE) and is known to be susceptible to over-exploration, which manifests in unrealistic solutions that reduce its practicality. LES leverages the trained decoder’s approximation of the data distribution, and can be employed with any VAE decoder–including pretrained ones–without additional training, architectural changes or access to the training data. Our evaluation across five LSO benchmark tasks and twenty-two VAE models demonstrates that LES always enhances the quality of the solutions while maintaining high objective values, leading to improvements over existing solutions in most cases. We believe that new avenues to LSO will be opened by LES’ ability to identify out of distribution areas, differentiability, and computational tractability.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Latent Space Optimisation (LSO) is prone to over-exploration, leading to invalid solutions. This paper proposes to regularise the acquisition function $\\mathcal{A}_{\\hat{f}}(\\mathbf{z})$ to restrict the selected query points $\\mathbf{z}^{(new)}$ to a subset leading to valid output sequences. This is done via the proposed score, ScaLES, which approximates the log-likelihood of valid samples through the log-determinant of the decoder's Jacobian (pseudo) inverse. This score is shown to be a good proxy measure both theoretically and empirically. Furthermore, ScaLES also demonstrates SOTA computational time compared to existing methods.

### Strengths
Quality
======
- The paper is theoretically sound; intuitive explanations and well-detailed derivations are provided in both the main paper and the Appendix. I carefully checked the derivations and did not find any issues. 
- The experiments provided are extensive, using several architectures, datasets, and hyperparameters. Additional ablation results are also provided. All this shows that ScaLES generalises well across various configurations.
- The limitations of ScaLES are discussed, and the impact of some theoretical assumptions in practical applications is well-detailed.

Clarity
=====
- The paper is well-written and easy to follow.
- As far as I know, the contextualisation with respect to related work is done adequately. Still, LSO is not my area of research, so I may not notice missing related work.

Originality
========
- As far as I know, the proposed method is novel. However, as I mentioned above, LSO is not my area of research, so I may not be aware of concurrent work close to the proposed method.

Significance
=========
- ScaLES is well-motivated theoretically and shows SOTA results across extensive experiments. This score can be of interest for the LSO community, both on the theoretical and practical side.

### Weaknesses
I really enjoyed reading this paper, and I only have a few minor comments.

Mathematical notation
=================
- In Alg. (1), the new samples are generally referred to as $\\cdot^{(new)}$, except $y$, which is referred to as $y^{new}$. I suggest renaming it $y^{(new)}$ for consistency.
- In Eq. (6), $L$ is not defined.
- l. 212, the vocabulary size and sequence length are denoted in two ways: D, L and $D, L$. Which notation is correct?
- In Eq. (19), shouldn't $\\mathbf{x}$ be  $\\mathbf{x}_{\\mathbf{z}}$ as in Eq. (9), or did I miss something there?
- l. 648, could the numerator of $p_{\\mathbf{z}}^{(i)}$ be rewritten with $\\exp(\cdot)$ instead of $e^{\cdot}$? the font size of the fraction is quite small, and it is hard to distinguish which subscripts go where.

Clarity
=====
- Are the results in Fig. 9 an average over all the datasets? If so, could we also have the details per dataset (especially for the challenging Expressions dataset)? If it is only over one dataset, could the dataset used be mentioned in the legend?
- In Sec.5, given the need to select a good value for $\\lambda$, I think the claim that "ScaLES has no hyperparameter" could be misleading. I would encourage the authors to reformulate this and mention the need for the $\lambda$ parameter to weight the score given by ScaLES.

Typo
====
- Appendix A is empty, consider removing it if not used.

### Questions
- [1] observed that almost all types of probabilistic generative models can spuriously assign a high likelihood to OOD data samples. How would this affect ScaLES?
- In 4.2, the authors mentionned that $ \\lambda > 0.5$ did not improve the optimisation process. Are the results worse in that case? If so, why?
- Can the authors provide some insights into why the Expressions dataset was more challenging for ScaLES? did $\\lambda > 0.05$ provided worse results or just no improvements?

References
=========
[1] Nalisnick, Eric, et al. "Do Deep Generative Models Know What They Don't Know?." International Conference on Learning Representations. 2019

### Soundness
4

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
This paper proposes a method for latent space optimization called ScaLES. The author's highlight that existing methods for latent space optimization often yield latents which correspond to invalid configurations in the observation space. To focus on latents within the "valid set", the authors propose to use the likelihood of a sample under the learned generative model, specifically using the decoder contribution to the likelihood. The authors show that incorporating this score into latent space optimization can be done in a computationally tractable manner and yields superior performance to existing methods on benchmark datasets.

### Strengths
* The paper address an important problem, i.e., finding effective methods for latent space optimization.
* The paper is straightforward to understand and well written.
* The paper does a very good job motivating the need for their method and contextualizing it relative to prior work.
* The method seems simple and straightforward to implement.
* The method shows promising performance in terms of identifying valid latent configurations yielding performance gains over existing methods.

### Weaknesses
**CPA Assumption**

A key aspect of the author's method is the assumption that all reasonable deep generative models can be well approximated by continuous piecewise affine (CPA) spline operators. I am confused exactly what the purpose is of making this approximation? If the sole purpose is to use this approximation to derive the likelihood of a sample under the generative model via Theorem 5, then the assumption seems unnecessary as such a formula exist via the well known change of variable formula or alternatively its extension to rectangular matrices [1]. Thus, unless I am missing something, the CPA assumption and Theorem 5, which constitute a large part of the method section, seem unnecessary.

**On Scalability**

The authors emphasize "scalability" as a key strength of the Scales method. Based on my understanding of the method, however, "scalability" does not seem to be the method's strong point. Scales requires computing the determinant of the Jacobian of a decoder. In the experiments the authors consider, the latent dimension is quite low, s.t. this computation can be done relatively efficiently, as shown in Table 1. It is known, however, that the computation of the log determinant of the decoder Jacobian scales cubicly with the latent dimension (as the authors note). I am not an expert in LSO, thus I do not know if most problems of interest require a significantly larger latent dimensionality. However, for such problems, the Scales method will not scale gracefully. Thus, I do not view "scalability" as the strong point of this method.

**Presentation**

I think the clarity of the presentation of the contributions and main ideas in this work can be improved. As far as I understand, the main contributions of this work are the following:
1. The authors propose the idea that the likelihood of a generative model can be used as a heuristic for determining whether a given latent will yield a valid observation.
2. The authors create a score for LSO based on this idea, using only the generator term in the likelihood formula.
3. The authors show that this score can be efficiently computed on the datasets considered and that it is able to yield latents which correspond to valid configurations more robustly than existing methods. 

Currently, I feel that the abstract and introduction are rather vague in terms of describing the core ideas and contribution of this work. To this end, I think a more direct description, along the lines of what is written above, would be easy to write and would improve the clarity of the paper.

**Validity of Scales Heuristic**

I am skeptical of the whether the likelihood of a generated latent under the decoder will always be a good heuristic for whether this latent will yield a valid observed configuration. For example, many configurations of interest are very likely OOD w.r.t. the empirical data distribution, i.e., a newly discovered molecule may have zero coverage under the empirical data distribution. Consequently, whether Scales is able to identify such a latent is contingent on whether or not the decoder is able to extrapolate to these unseen regions. In other words, the success of Scales seems intimately tied to the performance of the decoders considered. In this sense, I think calling the method a "general purpose regularization method for LSO" is a bit of an overclaim. I think a deeper discussion on this point, i.e., the validity of the Scales heuristic and its relationship to the generalization of the models in question, would improve the paper.

**Overclaims in Writing**

I feel that some of the writing, when motivating Scales feels like a bit of an overclaim that should be toned down or at least better expounded upon. For example, the authors motivate Scales as a "exact and theoretically motivated method" or "implemented ... exactly as theoretically derived". In what sense is the method "exact" and in what sense is it "theoretically motivated" or "derived"? The latter makes it sounds like there exist a theorem on the optimality of the method, when, in reality, I presume the authors are referring to their derivation of the likelihood (Thm 5). I do not think this constitutes theoretical motivation for the method and feels like an overclaim which makes the method sound more principled than it actually is, as I understand it.

Further, as previously discussed, I do not think calling the method "general purpose" or "scalable" is particularly accurate, and I would encourage the authors to explain what is meant by this, if they include this language.

**Baseline Method Descriptions**

A key baseline method that the authors compare against is the "Bayesian uncertainty score". The authors, however, do not give a self-contained description of this method in the main text or appendix which makes it difficult to contextualize some of the experimental findings. 

**Summary**

In summary, I think the core idea of this paper, i.e., that the likelihood of a generative model can serve as a good heuristic for identifying valid latents for LSO, is potentially interesting and the author's empirical results seem promising. Currently, however, I feel the manuscript obfuscates this relatively straightforward idea with superfluous theory and vague writing. Furthermore, I think several stated motivations of Scales, in particular scalability, feel like overclaims to me, for the reasons stated above.

**Bibliography**

1. http://benisrael.net/INTEGRAL-AMS.pdf

### Questions
**1.** What is the purpose of the CPA approximation?

**2.** What is the meaning of "scalability" in reference to Scales?

**3.** Is the latent dimension of the generative models considered in this work similar to what one would expect to use in more practical settings for LSO.

**4.** Lastly, I am curious if the authors have considered alternative instantiations of the Scales heuristic?

Specifically, a core idea of Scales, as I understand it, is that the generator likelihood, can serve as a good heuristic for the validity of latents in LSO. This likelihood consist of the prior likelihood term and a Jacobian determinant. Currently, the authors rely on the determinant term. I am curious if the prior likelihood could instead be used in some cases. For example, consider a case in which one does not force the latents to conform to a restricted prior such as a unit Gaussian as in a VAE, but instead just encodes data with a vanilla autoencoder. In this case, the latent distribution will have a complex shape which may better capture the overall likelihood of a generated latent. To measure this likelihood, one could model the unknown latent distribution via an exact likelihood method such as a normalizing flow. I am curious if the authors have intuition on whether the prior likelihood on its own, in such cases, could serve as an effective score for LSO. I ask this particularly because using the prior likelihood seems much more scalable than the Jacobian determinant, thus I am wondering if there are cases where this score could be effective.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposed a sequence density function for discrete sequential data $x$ in a latent variable model setting, i.e., VAEs. The authors further demonstrated the effectiveness of this density function in detecting out-of-distribution $x$, as well as in mitigating over-exploration during latent space optimization (LSO).

### Strengths
- The authors showed that the proposed method is more compute efficient and outperforms the baselines in LSO.
- The question of LSO for sequential data is important, and using VAEs to solve it is interesting.

### Weaknesses
VAEs are generative models, i.e., modelling $p(x,z)$ with the pre-defined prior $p(z)$ and the learned likelihood $p(x|z)$ (the decoder). When we train a VAE, we need to choose the class of model for $p(x|z)$ that correctly describes the data type of $x$. For example, in the case of  RGB images, people use discrete mixture of logistics to model the discrete pixel value range from 0 to 255.

From the paper, it is not mentioned what likelihood models are used in the decoder for such discrete sequential data type (maybe I missed some parts).

If I understand correctly, the authors proposed an estimator (denoted as $S(z)$) for $p(x|z)$. The part I am not sure about is: since we already have a free $p(x|z)$ from the VAE, can we simply use $p(x|z)$ to evaluate the density? What are the benefits on using ScaLES instead?

### Questions
1. Table 1, what is column ‘ScaLES/Uncertainty’? Is it the ratio or another method?
2. Typos:
    1. Page 2 line 097, “out-out-distribution”, —> “out-of-distribution”
    2. Page 8 line 396, “samples are decoded into the latent space”?
3. Other than VAEs, are there any method using other sequence generative models (e.g., Transformers) for this task?

### Soundness
3

### Presentation
2

### Contribution
2
